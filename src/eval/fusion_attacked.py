"""E6b: attacked test messages through Layer 3 (D4 content + simulated source), thresholds NOT re-tuned.

For each seed the E6 fusion models are rebuilt exactly as in src/fusion.py (val cross-fit -> thresholds, refit on val);
the thresholds are checked against results/E6/<seed>/metrics.json. Then every misinfo/forged_alert test message that
E4 attacked is replaced by its attacked version: content features = D4 prediction on the attacked text + normalizer
report of the attacked text; source features = the same simulated values as the clean message (an attack on the text
does not change the sender). Two views:
  attacked_fakes: decision shares (publish / human_review / block) among the attacked fakes;
  stream: all clean credible test messages + the attacked fakes -> false block rate, review load, missed forgery rate.
Diagnostic variants "probs" (D4 probabilities only) and "probs_source" (+ simulated source) show the effect of the
normalizer-report features, whose weights are learned on clean validation data only.
  python -m src.eval.fusion_attacked          -> results/E6/attacked_summary.csv, results/E6/<seed>/attacked.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.eval.metrics import decision_metrics
from src.fusion import CFG, NORM_FEATS, SRC_FEATS, decide, features, make_model, tune
from src.normalize import normalize

SEEDS = [13, 42, 2026]
ATTACKS = [("A1", "0.2"), ("A3", "1.0"), ("A7_official", "1")]
CONTENT = ["p_misinfo", "p_forged", "p_noncred"]
SETS = {"content": CONTENT + NORM_FEATS, "fusion": CONTENT + NORM_FEATS + SRC_FEATS,
        # diagnostic variants without the normalizer report (thresholds tuned on val exactly as in E6)
        "probs": CONTENT, "probs_source": CONTENT + SRC_FEATS}


def main():
    cdir = "results/" + CFG["content_source"]
    rows = []
    for seed in SEEDS:
        Xv, pv = features("val", seed, cdir)
        Xt, pt = features("test", seed, cdir)
        yv = (pv.y_true != "credible").astype(int).values
        saved = json.load(open(f"results/E6/{seed}/metrics.json"))
        out = {}
        for name, cols in SETS.items():
            key = f"lr.{name}"
            m = make_model("lr", seed)
            oof = cross_val_predict(m, Xv[cols], yv, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                    method="predict_proba")[:, 1]
            tr, tb = tune(oof, pv.y_true.values)
            if key in saved:
                assert np.isclose(tr, saved[key]["t_review"]) and np.isclose(tb, saved[key]["t_block"]), f"{key} thresholds differ from E6"
            m.fit(Xv[cols], yv)
            d_clean = decide(m.predict_proba(Xt[cols])[:, 1], tr, tb)
            clean_m = decision_metrics(pt.y_true.values, d_clean)
            if key in saved:
                assert np.isclose(clean_m["missed_forgery_rate"], saved[key]["test"]["missed_forgery_rate"]), f"{key} clean test differs from E6"
            cred = (pt.y_true == "credible").values
            for atk, b in ATTACKS:
                pa = pd.read_parquet(f"results/E4/M6_D3+aware/{seed}/predictions_{atk}_{b}.parquet")
                idx = pd.Series(np.arange(len(pt)), index=pt.id.values)
                pos = idx.loc[pa.id.values].values
                Xa = Xt.iloc[pos].reset_index(drop=True).copy()  # same simulated source features as the clean message
                Xa["p_misinfo"], Xa["p_forged"], Xa["p_noncred"] = pa.p_misinfo.values, pa.p_forged.values, 1 - pa.p_credible.values
                rep = pd.DataFrame([normalize(t)[1] for t in pa.text_adv])
                for c in NORM_FEATS:
                    Xa[c] = rep[c].values.astype(float)
                d_att = decide(m.predict_proba(Xa[cols])[:, 1], tr, tb)
                y_att = pa.y_true.values
                stream_y = np.concatenate([pt.y_true.values[cred], y_att])
                stream_d = np.concatenate([d_clean[cred], d_att])
                sm = decision_metrics(stream_y, stream_d)
                r = {"seed": seed, "model": key, "attack": atk, "budget": float(b), "n_attacked": len(pa),
                     "t_review": tr, "t_block": tb, "review_band_empty": bool(np.isclose(tr, tb)),
                     "clean_missed_forgery_rate": clean_m["missed_forgery_rate"], "clean_false_block_rate": clean_m["false_block_rate"],
                     "n_attacked_forged": int((y_att == "forged_alert").sum()),
                     "fakes_publish": float((d_att == "publish").mean()),
                     "fakes_review": float((d_att == "human_review").mean()),
                     "fakes_block": float((d_att == "block").mean()),
                     "fakes_clean_publish": float((d_clean[pos] == "publish").mean()),
                     "missed_forgery_rate": float((d_att[y_att == "forged_alert"] == "publish").mean()),
                     "stream_review_load": sm["review_load"], "stream_false_block_rate": sm["false_block_rate"],
                     "stream_missed_forgery_rate": sm["missed_forgery_rate"]}
                rows.append(r)
                out[f"{key}.{atk}.{b}"] = r
                print(json.dumps(r), flush=True)
        json.dump(out, open(f"results/E6/{seed}/attacked.json", "w"), indent=1)
    df = pd.DataFrame(rows)
    df.to_csv("results/E6/attacked_by_seed.csv", index=False)
    g = df.groupby(["model", "attack", "budget"])
    num = [c for c in df.columns if c not in ("seed", "model", "attack", "budget", "review_band_empty")]
    summ = g[num].mean().join(g[num].std(ddof=1), rsuffix="_std").reset_index()
    summ.to_csv("results/E6/attacked_summary.csv", index=False)
    print(summ.to_string())


if __name__ == "__main__":
    main()
