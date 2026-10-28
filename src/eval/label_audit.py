"""Task 2: automatic label audit with confident learning (cleanlab) on M1 out-of-fold probabilities.

5-fold CV on train+val (M1 with the C chosen on val in E1) -> out-of-fold probabilities -> cleanlab
find_label_issues (prune_by_noise_rate). Test examples are scored by M1 fitted on all of train+val with the same per-class
thresholds (mean self-confidence per class, as in confident learning); labels are never changed.
Sensitivity analysis: every E1 model's test metrics with all labels vs without suspicious test examples.
  python -m src.eval.label_audit
Writes results/label_audit.csv, results/label_audit_summary.json, results/E1_label_sensitivity.csv.
"""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.eval.metrics import LABELS, classification_metrics
from src.models.baselines import TfidfModel


def main():
    hp = json.load(open("results/E1/M1/13/config.json"))["hp"]
    tv = pd.concat([pd.read_parquet("data/splits/train.parquet"), pd.read_parquet("data/splits/val.parquet")], ignore_index=True)
    y = tv.label.map(LABELS.index).values
    oof = np.zeros((len(tv), len(LABELS)))
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=13).split(tv, y)):
        m = TfidfModel("M1", hp, 13, "none").fit(tv.text.iloc[a], tv.label.iloc[a])
        oof[b] = m.predict_proba(tv.text.iloc[b])
        print("fold", k, flush=True)
    from cleanlab.filter import find_label_issues
    issues = find_label_issues(labels=y, pred_probs=oof, filter_by="prune_by_noise_rate")
    thr = np.array([oof[y == j, j].mean() for j in range(len(LABELS))])  # per-class self-confidence thresholds

    te = pd.read_parquet("data/splits/test.parquet")
    full = TfidfModel("M1", hp, 13, "none").fit(tv.text, tv.label)
    pt = full.predict_proba(te.text)
    yt = te.label.map(LABELS.index).values
    # confident-learning rule: own-label prob below its class threshold and another class above its threshold
    above = pt >= thr
    above[np.arange(len(yt)), yt] = False
    sus_t = (pt[np.arange(len(yt)), yt] < thr[yt]) & above.any(1)

    def rows(d, P, yy, flag, split):
        pred = np.asarray(LABELS)[P.argmax(1)]
        conf = P[np.arange(len(yy)), yy]
        why = np.where(pred != d.label.values, "OOF prediction disagrees with label", "low self-confidence for its label")
        return pd.DataFrame({"id": d.id.values, "split": split, "text": d.text.values, "label": d.label.values,
                             "is_synthetic": d.is_synthetic.values, "pred": pred, "confidence": conf,
                             "reason": why})[flag]
    out = pd.concat([rows(tv, oof, y, issues, "train+val"), rows(te, pt, yt, sus_t, "test")], ignore_index=True)
    out.to_csv("results/label_audit.csv", index=False)

    summ = {"method": "cleanlab find_label_issues(prune_by_noise_rate) on 5-fold OOF M1; test: CL per-class thresholds",
            "cleanlab_version": __import__("cleanlab").__version__, "m1_C": hp, "class_thresholds": dict(zip(LABELS, thr.tolist())),
            "train_val": {"n": len(tv), "n_suspicious": int(issues.sum()),
                          "by_label": tv[issues].label.value_counts().to_dict(),
                          "share_by_label": {l: float(issues[tv.label == l].mean()) for l in LABELS},
                          "share_real_misinfo": float(issues[(tv.label == "misinfo") & ~tv.is_synthetic].mean())},
            "test": {"n": len(te), "n_suspicious": int(sus_t.sum()), "by_label": te[sus_t].label.value_counts().to_dict(),
                     "real_misinfo_suspicious": int((sus_t & (te.label == "misinfo") & ~te.is_synthetic).sum())}}
    # agreement with the LLM-assisted audit of the 119 real misinfo test messages
    try:
        aud = pd.read_csv("results/review/test_real_misinfo_audit.csv")
        s = set(te.id[sus_t])
        aud["cl_suspicious"] = aud.id.isin(s)
        summ["test"]["vs_manual_audit"] = {"n": len(aud), "manual_invalid": int((~aud.valid_rumour).sum()),
                                          "cl_flagged": int(aud.cl_suspicious.sum()),
                                          "both": int((aud.cl_suspicious & ~aud.valid_rumour).sum())}
    except Exception as e:  # noqa: BLE001
        summ["test"]["vs_manual_audit"] = f"not computed: {e}"
    json.dump(summ, open("results/label_audit_summary.json", "w"), indent=1, ensure_ascii=False)

    # sensitivity: metrics with all labels vs without suspicious test examples (no retraining)
    keep = set(te.id[~sus_t])
    real = set(te.id[~te.is_synthetic.astype(bool)])
    res = []
    for f in sorted(glob.glob("results/E1/*/*/predictions_test.parquet")):
        name, seed = f.split("/")[2], f.split("/")[3]
        p = pd.read_parquet(f)
        for sl, mask in [("test", np.ones(len(p), bool)), ("test_real", p.id.isin(real).values)]:
            for variant, m2 in [("all_labels", mask), ("without_suspicious", mask & p.id.isin(keep).values)]:
                d = p[m2]
                if d.y_true.nunique() < 2:
                    continue
                cm = classification_metrics(d.y_true.values, d.y_pred.values, d[["p_credible", "p_misinfo", "p_forged"]].values)
                res.append({"model": name, "seed": int(seed), "slice": sl, "variant": variant, "n": len(d),
                            "macro_f1": cm["macro_f1"], "mcc": cm["mcc"], "recall_forged": cm.get("recall_forged_alert", cm.get("recall_forged"))})
    pd.DataFrame(res).to_csv("results/E1_label_sensitivity.csv", index=False)
    print(json.dumps(summ, indent=1, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
