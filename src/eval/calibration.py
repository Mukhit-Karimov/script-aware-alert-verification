"""E7: calibration (ECE-15, Brier) before/after temperature scaling (T fitted on val by NLL) + coverage–risk.

Temperature is applied to log-probabilities (p_T ∝ p^(1/T)), which equals logit scaling for softmax models.
  python -m src.eval.calibration --pred-dir results/E1/M6 --name M6
Writes results/E7/<name>/<seed>/calibration.json and coverage_risk.parquet, reliability.parquet.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.eval.metrics import LABELS, brier, ece

PC = ["p_credible", "p_misinfo", "p_forged"]


def temp_scale(P, T):
    L = np.log(np.clip(P, 1e-12, 1)) / T
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def fit_T(P, y):
    yi = np.array([LABELS.index(l) for l in y])
    nll = lambda T: -np.mean(np.log(np.clip(temp_scale(P, T)[np.arange(len(yi)), yi], 1e-12, 1)))
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def reliability(y, P, n_bins=15):
    conf = P.max(1)
    corr = np.asarray(LABELS)[P.argmax(1)] == np.asarray(y)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1], right=True), 0, n_bins - 1)
    return pd.DataFrame([{"bin": b, "lo": bins[b], "hi": bins[b + 1], "n": int((idx == b).sum()),
                          "conf": float(conf[idx == b].mean()) if (idx == b).any() else np.nan,
                          "acc": float(corr[idx == b].mean()) if (idx == b).any() else np.nan} for b in range(n_bins)])


def coverage_risk(y, P):
    conf = P.max(1)
    err = np.asarray(LABELS)[P.argmax(1)] != np.asarray(y)
    order = np.argsort(-conf)
    n = len(y)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err[order]) / np.arange(1, n + 1)
    return pd.DataFrame({"coverage": cov, "risk": risk, "tau": conf[order]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    for d in sorted(glob.glob(f"{a.pred_dir}/*/")):
        seed = d.rstrip("/").split("/")[-1]
        if not (os.path.exists(f"{d}/predictions_val.parquet") and os.path.exists(f"{d}/predictions_test.parquet")):
            continue
        va, te = pd.read_parquet(f"{d}/predictions_val.parquet"), pd.read_parquet(f"{d}/predictions_test.parquet")
        T = fit_T(va[PC].values, va.y_true.values)
        out = {"T": T}
        for tag, P in [("before", te[PC].values), ("after", temp_scale(te[PC].values, T))]:
            out[tag] = {"ece": ece(te.y_true, P), "brier": brier(te.y_true, P)}
            real = ~te.is_synthetic.values.astype(bool)
            out[tag + "_real_only"] = {"ece": ece(te.y_true[real], P[real]), "brier": brier(te.y_true[real], P[real])}
            od = f"results/E7/{a.name}/{seed}"
            os.makedirs(od, exist_ok=True)
            reliability(te.y_true, P).to_parquet(f"{od}/reliability_{tag}.parquet", index=False)
            cr = coverage_risk(te.y_true.values, P)
            cr.to_parquet(f"{od}/coverage_risk_{tag}.parquet", index=False)
            out[tag]["risk_at_cov"] = {str(c): float(cr.risk.iloc[max(0, int(c * len(cr)) - 1)]) for c in (0.5, 0.8, 0.9, 0.95, 1.0)}
        json.dump(out, open(f"results/E7/{a.name}/{seed}/calibration.json", "w"), indent=1)
        print(a.name, seed, json.dumps({k: out[k] for k in ("T", "before", "after")}))


if __name__ == "__main__":
    main()
