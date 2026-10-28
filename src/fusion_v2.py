"""Layer 3 v2 (E6c): monotone, adversarially trained risk fusion with an explicit human-review band.

Differences from src/fusion.py (E6), all decided on validation data only:
  (a) monotone logistic regression: weights of the anomaly features of the normalizer report (invisible characters,
      NFKC changes, remapped homoglyphs, mixed-script tokens, kk_letter_substitution_rate) are constrained to be >= 0,
      so an anomaly can only raise risk (L-BFGS-B with bounds, L2 penalty C = 1, balanced class weights);
      anomaly counts enter as log1p;
  (b) training data = clean validation set + one attacked copy of every misinfo/forged_alert validation message
      (random attack A1-A4 with a random budget; D4 predictions and normalizer report of the attacked text;
      the copy keeps the simulated source features of its original);
  (c) thresholds from 5-fold out-of-fold scores (folds grouped by message): t_block = smallest score with
      <= 1% of clean credible validation messages blocked; t_publish = largest score with <= 1% of forged alerts
      (clean + attacked) published; the review band [t_publish, t_block) must hold <= 5% of clean validation messages.
Test data are used once, for evaluation: clean test, and attacked test fakes (E4 attacks on D4: A1 20%, A3 100%,
A7 official), each attacked fake replacing its clean version in a stream with all clean credible test messages.
  python -m src.fusion_v2            -> results/E6c/<seed>/{val_attacked.parquet,metrics.json}, results/E6c/summary*.csv
"""
import json
import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

from src.attacks.char_attacks import BUDGETS, apply
from src.attacks.run_attacks import Counted, load_model
from src.eval.metrics import decision_metrics
from src.fusion import CFG, NORM_FEATS, SRC_FEATS, decide, features, simulate_source
from src.normalize import normalize

SEEDS = [13, 42, 2026]
CONTENT = ["p_misinfo", "p_forged", "p_noncred"]
COUNTS = ["n_invisible", "n_nfkc_changed", "n_homoglyph_fixed", "n_mixed_tokens", "n_mixed_protected"]
MONOTONE = ["n_invisible", "n_nfkc_changed", "n_homoglyph_fixed", "n_mixed_tokens", "kk_letter_substitution_rate"]
COLS = CONTENT + NORM_FEATS + SRC_FEATS
TEST_ATTACKS = [("A1", "0.2"), ("A3", "1.0"), ("A7_official", "1")]
MAX_FB, MAX_MISSED, MAX_REVIEW = CFG["thresholds"]["max_false_block"], CFG["thresholds"]["max_missed_forgery"], 0.05
CDIR = "results/" + CFG["content_source"]  # E5/M6_D3+aware = D4


class MonotoneLR:
    """L2 logistic regression on standardized features; weights of `monotone` columns bounded below by 0."""

    def __init__(self, cols, monotone, C=1.0):
        self.cols, self.C = cols, C
        self.lower = np.array([0.0 if c in monotone else -np.inf for c in cols])

    def _x(self, X):
        Z = X[self.cols].astype(float).copy()
        for c in COUNTS:
            if c in Z:
                Z[c] = np.log1p(Z[c])
        return Z.values

    def fit(self, X, y):
        Z = self._x(X)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        Z = (Z - self.mu) / self.sd
        y = np.asarray(y, float)
        sw = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))

        def f(theta):
            w, b = theta[:-1], theta[-1]
            z = Z @ w + b
            p = 1 / (1 + np.exp(-z))
            loss = np.sum(sw * (np.logaddexp(0, z) - y * z)) + 0.5 / self.C * w @ w
            g = sw * (p - y)
            return loss, np.append(Z.T @ g + w / self.C, g.sum())
        bounds = [(lo if np.isfinite(lo) else None, None) for lo in self.lower] + [(None, None)]
        r = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B", bounds=bounds)
        self.w, self.b, self.converged = r.x[:-1], r.x[-1], bool(r.success)
        return self

    def score(self, X):
        return 1 / (1 + np.exp(-(((self._x(X) - self.mu) / self.sd) @ self.w + self.b)))


def attacked_val(seed, Xv, pv):
    """One attacked copy of every non-credible validation message, scored by D4 (cached)."""
    path = f"results/E6c/{seed}/val_attacked.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    import torch  # noqa: F401  (load_model needs it)
    val = pd.read_parquet("data/splits/val.parquet").set_index("id").loc[pv.id]
    rng = random.Random(seed)
    rows = []
    for pos, (i, r) in enumerate(val.iterrows()):
        if r.label == "credible":
            continue
        atk = rng.choice(["A1", "A2", "A3", "A4"])
        b = rng.choice(BUDGETS[atk])
        rows.append({"pos": pos, "id": i, "y_true": r.label, "attack": atk, "budget": b,
                     "text_adv": apply(atk, r.text, b, rng.randrange(2 ** 31))[0]})
    df = pd.DataFrame(rows)
    model = Counted(load_model("E5", "M6_D3", seed), "aware")
    P = np.asarray(model(df.text_adv.tolist()))
    df["p_credible"], df["p_misinfo"], df["p_forged"] = P[:, 0], P[:, 1], P[:, 2]
    rep = pd.DataFrame([normalize(t)[1] for t in df.text_adv])
    for c in NORM_FEATS:
        df[c] = rep[c].values.astype(float)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def as_features(base_X, base_pos, df):
    """Feature rows for attacked texts: content + report from `df`, source features copied from the clean rows."""
    X = base_X.iloc[base_pos].reset_index(drop=True).copy()
    X["p_misinfo"], X["p_forged"], X["p_noncred"] = df.p_misinfo.values, df.p_forged.values, 1 - df.p_credible.values
    for c in NORM_FEATS:
        X[c] = df[c].values.astype(float)
    return X


def thresholds(scores, y, clean):
    """t_block from clean credible, t_publish from all forged alerts (clean + attacked); review budget on clean val."""
    y, clean = np.asarray(y), np.asarray(clean)
    t_block = float(np.quantile(scores[clean & (y == "credible")], 1 - MAX_FB, method="higher")) + 1e-9
    t_pub = float(np.quantile(scores[y == "forged_alert"], MAX_MISSED, method="lower"))
    t_pub = min(t_pub, t_block)
    review = float(((scores >= t_pub) & (scores < t_block))[clean].mean())
    feasible = review <= MAX_REVIEW
    if not feasible:  # shrink the band to the budget: raise t_publish until <= 5% of clean val is in review
        s = np.sort(scores[clean & (scores < t_block)])
        t_pub = float(s[max(0, len(s) - int(np.floor(MAX_REVIEW * clean.sum())))])
        review = float(((scores >= t_pub) & (scores < t_block))[clean].mean())
    return t_pub, t_block, feasible, review


def main():
    rows, seed_metrics = [], {}
    for seed in SEEDS:
        Xv, pv = features("val", seed, CDIR)
        Xt, pt = features("test", seed, CDIR)
        va = attacked_val(seed, Xv, pv)
        Xa = as_features(Xv, va.pos.values, va)
        X = pd.concat([Xv, Xa], ignore_index=True)
        y_lab = np.concatenate([pv.y_true.values, va.y_true.values])
        y = (y_lab != "credible").astype(int)
        clean = np.r_[np.ones(len(Xv), bool), np.zeros(len(Xa), bool)]
        groups = np.concatenate([pv.id.values, va.id.values])
        oof = np.zeros(len(X))
        for tr, te in GroupKFold(5).split(X, y, groups):
            oof[te] = MonotoneLR(COLS, MONOTONE).fit(X.iloc[tr], y[tr]).score(X.iloc[te])
        t_pub, t_block, feasible, review_val = thresholds(oof, y_lab, clean)
        m = MonotoneLR(COLS, MONOTONE).fit(X, y)
        d_val = decide(oof, t_pub, t_block)
        val_m = {"clean": decision_metrics(y_lab[clean], d_val[clean]), "attacked_fakes_missed_forgery":
                 float((d_val[~clean][y_lab[~clean] == "forged_alert"] == "publish").mean())}
        st = m.score(Xt)
        d_clean = decide(st, t_pub, t_block)
        cred = (pt.y_true == "credible").values
        clean_m = decision_metrics(pt.y_true.values, d_clean)
        seed_metrics[seed] = {"t_publish": t_pub, "t_block": t_block, "review_budget_feasible": feasible,
                              "val_review_load_clean": review_val, "val": val_m, "test_clean": clean_m,
                              "converged": m.converged, "coef_standardized": dict(zip(COLS, map(float, m.w)))}
        rows.append({"seed": seed, "attack": "clean", "fakes_publish": float((d_clean[~cred] == "publish").mean()),
                     "fakes_review": float((d_clean[~cred] == "human_review").mean()),
                     "fakes_block": float((d_clean[~cred] == "block").mean()),
                     "missed_forgery_rate": clean_m["missed_forgery_rate"], "review_load": clean_m["review_load"],
                     "false_block_rate": clean_m["false_block_rate"]})
        idx = pd.Series(np.arange(len(pt)), index=pt.id.values)
        for atk, b in TEST_ATTACKS:
            pa = pd.read_parquet(f"results/E4/M6_D3+aware/{seed}/predictions_{atk}_{b}.parquet")
            rep = pd.DataFrame([normalize(t)[1] for t in pa.text_adv])
            for c in NORM_FEATS:
                pa[c] = rep[c].values.astype(float)
            Xatt = as_features(Xt, idx.loc[pa.id.values].values, pa)
            d_att = decide(m.score(Xatt), t_pub, t_block)
            ya = pa.y_true.values
            sm = decision_metrics(np.concatenate([pt.y_true.values[cred], ya]), np.concatenate([d_clean[cred], d_att]))
            rows.append({"seed": seed, "attack": atk, "fakes_publish": float((d_att == "publish").mean()),
                         "fakes_review": float((d_att == "human_review").mean()),
                         "fakes_block": float((d_att == "block").mean()),
                         "missed_forgery_rate": float((d_att[ya == "forged_alert"] == "publish").mean()),
                         "review_load": sm["review_load"], "false_block_rate": sm["false_block_rate"]})
        print(json.dumps({"seed": seed, **{k: v for k, v in seed_metrics[seed].items() if k != "coef_standardized"}}), flush=True)
        os.makedirs(f"results/E6c/{seed}", exist_ok=True)
        json.dump(seed_metrics[seed], open(f"results/E6c/{seed}/metrics.json", "w"), indent=1)
    df = pd.DataFrame(rows)
    df.to_csv("results/E6c/summary_by_seed.csv", index=False)
    num = [c for c in df.columns if c not in ("seed", "attack")]
    g = df.groupby("attack", sort=False)[num]
    g.mean().join(g.std(ddof=1), rsuffix="_std").reset_index().to_csv("results/E6c/summary.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
