"""Evaluation metrics (paper, Section 5). Labels are strings; LABELS fixes the column order of probability arrays."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

LABELS = ["credible", "misinfo", "forged_alert"]


def classification_metrics(y_true, y_pred, proba=None, labels=LABELS):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = [l for l in labels if l in set(y_true)]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    m = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(labels)},
    }
    if "forged_alert" in labels:
        m["recall_forged"] = m["per_class"]["forged_alert"]["recall"]
    if proba is not None:
        proba = np.asarray(proba, dtype=float)
        Y = np.stack([(y_true == l).astype(int) for l in labels], 1)
        cols = [i for i, l in enumerate(labels) if 0 < Y[:, i].sum() < len(Y)]
        m["roc_auc_macro"] = float(np.mean([roc_auc_score(Y[:, i], proba[:, i]) for i in cols])) if cols else None
        m["pr_auc_macro"] = float(np.mean([average_precision_score(Y[:, i], proba[:, i]) for i in cols])) if cols else None
        if "forged_alert" in labels:
            j = labels.index("forged_alert")
            m["fpr_at_tpr95_forged"] = fpr_at_tpr(Y[:, j], proba[:, j], 0.95) if j in cols else None
        m["ece"] = ece(y_true, proba, labels)
        m["brier"] = brier(y_true, proba, labels)
    return m


def fpr_at_tpr(y_bin, score, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(y_bin, score)
    return float(fpr[np.searchsorted(tpr, tpr_target, side="left")])


def ece(y_true, proba, labels=LABELS, n_bins=15):
    """Expected calibration error of the top-1 confidence, equal-width bins."""
    proba = np.asarray(proba, dtype=float)
    conf = proba.max(1)
    correct = np.asarray(labels)[proba.argmax(1)] == np.asarray(y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def brier(y_true, proba, labels=LABELS):
    """Multiclass Brier score: mean over samples of sum_k (p_k - y_k)^2."""
    Y = np.stack([(np.asarray(y_true) == l).astype(float) for l in labels], 1)
    return float(np.mean(np.sum((np.asarray(proba, dtype=float) - Y) ** 2, 1)))


def attack_metrics(y_true, pred_clean, pred_adv, target="credible"):
    """ASR = share of originally-correct examples that become `target` after attack."""
    y_true, pc, pa = map(np.asarray, (y_true, pred_clean, pred_adv))
    correct = pc == y_true
    n_correct = int(correct.sum())
    return {
        "n": int(len(y_true)),
        "clean_accuracy": float(correct.mean()) if len(y_true) else None,
        "robust_accuracy": float((pa == y_true).mean()) if len(y_true) else None,
        "asr": float((pa[correct] == target).mean()) if n_correct else None,
        "n_attacked": n_correct,
    }


def delta_f1(f1_clean, f1_adv):
    return f1_clean - f1_adv


def decision_metrics(y_true, decision):
    """Layer-3 metrics. decision in {publish, human_review, block}."""
    y_true, d = np.asarray(y_true), np.asarray(decision)
    cred, forged = y_true == "credible", y_true == "forged_alert"
    return {
        "false_block_rate": float((d[cred] == "block").mean()) if cred.any() else None,
        "review_load": float((d == "human_review").mean()) if len(d) else None,
        "missed_forgery_rate": float((d[forged] == "publish").mean()) if forged.any() else None,
    }
