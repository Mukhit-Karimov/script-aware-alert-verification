"""Layer 3 / E6: risk fusion of content probabilities + normalizer report + (simulated) source metadata.

Decision: risk >= t_block -> block; t_review <= risk < t_block -> human_review; else publish.
Fusion models (LR, LightGBM) are cross-fitted on val (5-fold out-of-fold scores) for threshold tuning, then refit on
the full val and applied once to test (rule 2). Feature sets: content / source / fusion.
  python -m src.fusion --seed 13 [--attack A1 --budget 0.1]
"""
import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")  # the node is shared with training jobs
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import decision_metrics

CFG = yaml.safe_load(open("configs/E6.yaml"))
NORM_FEATS = ["n_invisible", "n_nfkc_changed", "n_homoglyph_fixed", "n_mixed_tokens", "n_mixed_protected",
              "kk_letter_ratio", "kk_letter_substitution_rate"]
SRC_FEATS = ["channel_official", "log_account_age", "repost_burst"]


def simulate_source(labels, seed):
    rng = np.random.default_rng(seed)
    f = CFG["features"]
    out = {k: np.zeros(len(labels)) for k in SRC_FEATS}
    for l in ("credible", "misinfo", "forged_alert"):
        m = np.asarray(labels) == l
        n = int(m.sum())
        out["channel_official"][m] = rng.random(n) < f["channel_official"][l]
        mu, sg = f["account_age_days"][l]
        out["log_account_age"][m] = np.log1p(rng.lognormal(mu, sg, n))
        k, sc = f["repost_burst"][l]
        out["repost_burst"][m] = rng.gamma(k, sc, n)
    return pd.DataFrame(out)


def features(split, seed, content_dir):
    pr = pd.read_parquet(f"{content_dir}/{seed}/predictions_{split}.parquet")
    nr = pd.read_parquet(f"{content_dir}/{seed}/norm_report_{split}.parquet").set_index("id").loc[pr.id]
    X = pd.DataFrame({"p_misinfo": pr.p_misinfo.values, "p_forged": pr.p_forged.values,
                      "p_noncred": 1 - pr.p_credible.values})
    for c in NORM_FEATS:
        X[c] = nr[c].values.astype(float)
    S = simulate_source(pr.y_true.values, CFG["seed"] + seed + (0 if split == "val" else 1))
    X = pd.concat([X, S], axis=1)
    return X, pr


def tune(scores, y):
    """t_review: largest threshold with missed forged (score < t) <= max_missed; t_block: smallest with FBR <= max_fb."""
    y = np.asarray(y)
    f, c = scores[y == "forged_alert"], scores[y == "credible"]
    t_review = float(np.quantile(f, CFG["thresholds"]["max_missed_forgery"], method="lower")) if len(f) else 0.5
    t_block = float(np.quantile(c, 1 - CFG["thresholds"]["max_false_block"], method="higher")) + 1e-9
    return t_review, max(t_block, t_review)


def decide(scores, t_review, t_block):
    return np.where(scores >= t_block, "block", np.where(scores >= t_review, "human_review", "publish"))


def make_model(kind, seed):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=seed, verbose=-1, n_jobs=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--content", default=CFG["content_source"], help="results/<exp>/<name> with val/test predictions + norm reports")
    ap.add_argument("--out", default="E6")
    a = ap.parse_args()
    cdir = "results/" + a.content
    Xv, pv = features("val", a.seed, cdir)
    Xt, pt = features("test", a.seed, cdir)
    yv = (pv.y_true != "credible").astype(int).values
    sets = {"content": ["p_misinfo", "p_forged", "p_noncred"] + NORM_FEATS, "source": SRC_FEATS,
            "fusion": ["p_misinfo", "p_forged", "p_noncred"] + NORM_FEATS + SRC_FEATS}
    res = {}
    od = f"results/{a.out}/{a.seed}"
    os.makedirs(od, exist_ok=True)
    for kind in ("lr", "lgbm"):
        for name, cols in sets.items():
            m = make_model(kind, a.seed)
            oof = cross_val_predict(m, Xv[cols], yv, cv=StratifiedKFold(5, shuffle=True, random_state=a.seed),
                                    method="predict_proba")[:, 1]
            tr, tb = tune(oof, pv.y_true.values)
            m.fit(Xv[cols], yv)
            st = m.predict_proba(Xt[cols])[:, 1]
            d = decide(st, tr, tb)
            real = ~pt.is_synthetic.values.astype(bool)
            key = f"{kind}.{name}"
            res[key] = {"t_review": tr, "t_block": tb, "val_oof": decision_metrics(pv.y_true.values, decide(oof, tr, tb)),
                        "test": decision_metrics(pt.y_true.values, d), "test_real_only": decision_metrics(pt.y_true.values[real], d[real])}
            pd.DataFrame({"id": pt.id, "y_true": pt.y_true, "risk": st, "decision": d, "is_synthetic": pt.is_synthetic,
                          "lang": pt.lang}).to_parquet(f"{od}/decisions_{key}.parquet", index=False)
            pd.DataFrame({"y_true": pv.y_true, "risk_oof": oof}).to_parquet(f"{od}/val_oof_{key}.parquet", index=False)
            if kind == "lr":
                coefs = m[-1].coef_[0]
                res[key]["coef_standardized"] = dict(zip(cols, map(float, coefs)))
            print(key, json.dumps(res[key]["test"]))
    res["note"] = "source features are SIMULATED (configs/E6.yaml) - scenario analysis"
    json.dump(res, open(f"{od}/metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
