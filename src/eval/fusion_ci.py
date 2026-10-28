"""Task 4: bootstrap 95% CIs (1000 resamples of test messages) for the Layer-3 decision metrics; thresholds unchanged.

Each resample draws the same message ids for all 3 seeds; the statistic is the mean over seeds.
Also reports the val (cross-fitted) value of each metric, to show the val -> test gap of the 1% target.
  python -m src.eval.fusion_ci  -> results/E6/fusion_ci.csv
"""
import glob
import json

import numpy as np
import pandas as pd

from src.eval.metrics import decision_metrics

METRICS = ["false_block_rate", "review_load", "missed_forgery_rate"]


def main(n_boot=1000):
    keys = sorted({f.split("decisions_")[1][:-8] for f in glob.glob("results/E6/*/decisions_*.parquet")})
    rows = []
    for key in keys:
        seeds = sorted(glob.glob(f"results/E6/*/decisions_{key}.parquet"))
        ds = [pd.read_parquet(f).set_index("id") for f in seeds]
        ids = ds[0].index.values
        ds = [d.loc[ids] for d in ds]
        val = [json.load(open(f.replace(f"decisions_{key}.parquet", "metrics.json")))[key]["val_oof"] for f in seeds]
        for sl, mask in [("test", np.ones(len(ids), bool)), ("test_real_only", ~ds[0].is_synthetic.astype(bool).values)]:
            idx = np.where(mask)[0]
            pm = [decision_metrics(d.y_true.values[idx], d.decision.values[idx]) for d in ds]
            point = {m: (np.mean([x[m] for x in pm]) if all(x[m] is not None for x in pm) else None) for m in METRICS}
            rng = np.random.default_rng(0)
            boot = {m: [] for m in METRICS}
            for _ in range(n_boot):
                b = rng.choice(idx, len(idx), replace=True)
                ms = [decision_metrics(d.y_true.values[b], d.decision.values[b]) for d in ds]
                for m in METRICS:
                    v = [x[m] for x in ms if x[m] is not None and not (isinstance(x[m], float) and np.isnan(x[m]))]
                    if v:
                        boot[m].append(np.mean(v))
            for m in METRICS:
                if point[m] is None or (isinstance(point[m], float) and np.isnan(point[m])) or not boot[m]:
                    continue
                rows.append({"model": key, "slice": sl, "metric": m, "value": point[m],
                             "ci_lo": float(np.percentile(boot[m], 2.5)), "ci_hi": float(np.percentile(boot[m], 97.5)),
                             "val_value": float(np.mean([v[m] for v in val])) if sl == "test" else np.nan,
                             "n": int(len(idx)), "n_seeds": len(ds)})
    out = pd.DataFrame(rows)
    out.to_csv("results/E6/fusion_ci.csv", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
