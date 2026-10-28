"""E10 statistics: bootstrap CIs, McNemar (continuity-corrected), Friedman + Nemenyi, Holm correction."""
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

from src.eval.metrics import LABELS


def bootstrap_ci(y_true, y_pred, metric="macro_f1", n=1000, seed=0, alpha=0.05):
    """Percentile bootstrap over examples. metric: 'macro_f1' | 'accuracy' | callable(y_true, y_pred)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    if metric == "macro_f1":
        present = [l for l in LABELS if l in set(y_true)]
        f = lambda t, p: f1_score(t, p, labels=present, average="macro", zero_division=0)
    elif metric == "accuracy":
        f = lambda t, p: float((t == p).mean())
    else:
        f = metric
    point = f(y_true, y_pred)
    idx = rng.integers(0, len(y_true), (n, len(y_true)))
    boots = np.array([f(y_true[i], y_pred[i]) for i in idx])
    return float(point), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_rate(flags, n=1000, seed=0, alpha=0.05):
    """CI for a proportion such as ASR (flags: 1 = success)."""
    flags = np.asarray(flags, dtype=float)
    rng = np.random.default_rng(seed)
    boots = flags[rng.integers(0, len(flags), (n, len(flags)))].mean(1)
    return float(flags.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def mcnemar(correct_a, correct_b):
    """McNemar test with continuity correction on paired correctness vectors. Returns (chi2, p, b, c)."""
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    if b + c == 0:
        return 0.0, 1.0, b, c
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return float(chi2), float(stats.chi2.sf(chi2, 1)), b, c


def holm(pvals):
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


# Nemenyi critical values q_alpha (alpha = 0.05) for k = 2..10 (Demšar, 2006)
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def friedman_nemenyi(scores):
    """scores: (n_blocks, k_models), higher = better. Returns Friedman (stat, p), average ranks, Nemenyi CD."""
    scores = np.asarray(scores, float)
    n, k = scores.shape
    stat, p = stats.friedmanchisquare(*scores.T)
    ranks = np.mean([stats.rankdata(-row) for row in scores], axis=0)
    cd = _Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
    return float(stat), float(p), ranks, float(cd)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
