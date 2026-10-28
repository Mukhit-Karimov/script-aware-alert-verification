import numpy as np
import pytest

from src.eval.stats import bootstrap_ci, bootstrap_rate, friedman_nemenyi, holm, mcnemar


def test_mcnemar():
    a = [1, 1, 1, 1, 0, 0, 1, 1]
    b = [0, 0, 0, 1, 0, 1, 1, 1]
    chi2, p, b_, c = mcnemar(a, b)
    assert (b_, c) == (3, 1)
    assert chi2 == pytest.approx((abs(3 - 1) - 1) ** 2 / 4)
    assert mcnemar([1, 0], [1, 0])[1] == 1.0


def test_holm():
    adj = holm([0.01, 0.04, 0.03])
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_contains_point():
    y = ["credible"] * 50 + ["misinfo"] * 50
    p = ["credible"] * 45 + ["misinfo"] * 5 + ["misinfo"] * 50
    pt, lo, hi = bootstrap_ci(y, p, n=300)
    assert lo <= pt <= hi
    r, lo, hi = bootstrap_rate([1] * 30 + [0] * 70, n=300)
    assert r == 0.3 and lo < 0.3 < hi


def test_friedman():
    s = np.array([[0.9, 0.8, 0.7]] * 6)
    stat, p, ranks, cd = friedman_nemenyi(s)
    assert ranks.tolist() == [1, 2, 3]
    assert p < 0.05 and cd == pytest.approx(2.343 * np.sqrt(3 * 4 / 36))
