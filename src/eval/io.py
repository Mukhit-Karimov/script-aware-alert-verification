"""Uniform result files: results/<exp>/<model>/<seed>/{predictions.parquet, metrics.json}."""
import json
import os

import numpy as np
import pandas as pd

from src.eval.metrics import LABELS, classification_metrics


def save_predictions(exp, model, seed, df, proba, attack="none", budget=0, extra=None, tag=None):
    """df: rows of the evaluated split (needs id, label, lang, is_synthetic); proba: (n, 3) in LABELS order."""
    proba = np.asarray(proba, dtype=float)
    pred = np.asarray(LABELS)[proba.argmax(1)]
    out = pd.DataFrame({
        "id": df.id.values, "y_true": df.label.values, "y_pred": pred,
        "p_credible": proba[:, 0], "p_misinfo": proba[:, 1], "p_forged": proba[:, 2],
        "lang": df.lang.values, "is_synthetic": df.is_synthetic.values, "attack": attack, "budget": budget,
    })
    d = os.path.join("results", exp, model, str(seed))
    os.makedirs(d, exist_ok=True)
    name = tag or "test"
    out.to_parquet(os.path.join(d, f"predictions_{name}.parquet"), index=False)
    m = metrics_with_slices(out)
    if extra:
        m.update(extra)
    with open(os.path.join(d, f"metrics_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    return out, m


_VER = {}


def _verified_ids(path="data/splits/test_real_verified.parquet"):
    """Ids of the manually audited real test subset (credible + verified real misinfo), if it exists."""
    if path not in _VER:
        _VER[path] = set(pd.read_parquet(path, columns=["id"]).id) if os.path.exists(path) else None
    return _VER[path]


def metrics_with_slices(pred):
    """All-data, real-only (rule 4), audited real subset and per-language slices."""
    P = pred[["p_credible", "p_misinfo", "p_forged"]].values
    m = {"all": classification_metrics(pred.y_true, pred.y_pred, P)}
    real = ~pred.is_synthetic.values.astype(bool)
    if real.any():
        m["real_only"] = classification_metrics(pred.y_true[real], pred.y_pred[real], P[real])
    ver = _verified_ids()
    if ver is not None:
        s = pred.id.isin(ver).values
        if s.any() and s.sum() < len(pred):
            m["real_verified"] = classification_metrics(pred.y_true[s], pred.y_pred[s], P[s])
    for lang in sorted(pred.lang.unique()):
        s = (pred.lang == lang).values
        m[f"lang_{lang}"] = classification_metrics(pred.y_true[s], pred.y_pred[s], P[s])
    return m
