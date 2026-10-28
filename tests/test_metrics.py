import numpy as np
import pytest

from src.eval.metrics import attack_metrics, brier, classification_metrics, decision_metrics, ece

Y = ["credible", "misinfo", "forged_alert", "credible"]


def test_perfect():
    P = np.eye(3)[[0, 1, 2, 0]]
    m = classification_metrics(Y, Y, P)
    assert m["macro_f1"] == 1.0 and m["mcc"] == 1.0 and m["recall_forged"] == 1.0
    assert m["roc_auc_macro"] == 1.0 and m["pr_auc_macro"] == 1.0
    assert m["ece"] == pytest.approx(0.0) and m["brier"] == pytest.approx(0.0)


def test_one_error():
    pred = ["credible", "misinfo", "credible", "credible"]
    m = classification_metrics(Y, pred)
    assert m["accuracy"] == 0.75
    assert m["recall_forged"] == 0.0
    # credible: P=2/3 R=1 F=0.8; misinfo F=1; forged F=0
    assert m["macro_f1"] == pytest.approx((0.8 + 1 + 0) / 3)


def test_ece_brier_uniform():
    P = np.full((4, 3), 1 / 3)
    assert brier(Y, P) == pytest.approx(2 / 3)
    # argmax = credible (first col); correct on 2/4; conf 1/3 -> |0.5-1/3|
    assert ece(Y, P) == pytest.approx(0.5 - 1 / 3)


def test_attack():
    y = ["misinfo", "misinfo", "forged_alert", "forged_alert"]
    clean = ["misinfo", "misinfo", "forged_alert", "credible"]
    adv = ["credible", "misinfo", "misinfo", "credible"]
    a = attack_metrics(y, clean, adv)
    assert a["clean_accuracy"] == 0.75 and a["robust_accuracy"] == 0.25
    assert a["asr"] == pytest.approx(1 / 3)  # only flips to `credible` count


def test_decision():
    y = ["credible", "credible", "forged_alert", "forged_alert", "misinfo"]
    d = ["publish", "block", "publish", "human_review", "block"]
    m = decision_metrics(y, d)
    assert m == {"false_block_rate": 0.5, "review_load": 0.2, "missed_forgery_rate": 0.5}
