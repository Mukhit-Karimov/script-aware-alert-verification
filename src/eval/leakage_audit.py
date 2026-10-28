"""Why are clean scores ~0.99? Leakage checks between train and test (no labels or splits are changed).

1. near-duplicates: for each test message, max estimated Jaccard (MinHash, char 5-grams) to any train message
   at thresholds 0.5 / 0.7 / 0.85, by label and real/synthetic;
2. synthetic lineage: synthetic test messages whose source post (synth_parent) or sibling (same parent) is in train;
3. length shortcut: macro-F1 of a depth-2 tree on text length alone (val-trained, test-evaluated).
  python -m src.eval.leakage_audit  -> results/leakage_audit.json
"""
import json

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier


def mh(t, k=5, num_perm=64):
    m = MinHash(num_perm=num_perm)
    t = " ".join(t.lower().split())
    for i in range(max(1, len(t) - k + 1)):
        m.update(t[i:i + k].encode("utf-8"))
    return m


def main():
    tr, va, te = (pd.read_parquet(f"data/splits/{s}.parquet") for s in ["train", "val", "test"])
    out = {}
    trm = [mh(t) for t in tr.text]
    tem = [mh(t) for t in te.text]
    for thr in [0.5, 0.7, 0.85]:
        lsh = MinHashLSH(threshold=thr, num_perm=64)
        for i, m in enumerate(trm):
            lsh.insert(str(i), m)
        hit = np.array([len(lsh.query(m)) > 0 for m in tem])
        # cross-label near-duplicates (a test message close to a train message of another label)
        cross = np.array([any(tr.label.iloc[int(j)] != te.label.iloc[k] for j in lsh.query(m)) for k, m in enumerate(tem)])
        out[f"near_dup_jaccard_ge_{thr}"] = {
            "share_test": float(hit.mean()), "n": int(hit.sum()),
            "share_by_label": te.assign(h=hit).groupby("label").h.mean().to_dict(),
            "share_real": float(hit[~te.is_synthetic.astype(bool).values].mean()),
            "share_synthetic": float(hit[te.is_synthetic.astype(bool).values].mean()),
            "n_cross_label": int(cross.sum())}
    syn = te[te.is_synthetic.astype(bool) & te.synth_parent.notna()]
    tr_ids = set(tr.id)
    tr_parents = set(tr.synth_parent.dropna())
    out["synthetic_lineage"] = {
        "n_synthetic_test_with_parent": int(len(syn)),
        "parent_in_train": int(syn.synth_parent.isin(tr_ids).sum()),
        "sibling_in_train": int(syn.synth_parent.isin(tr_parents).sum()),
        "by_label_sibling_in_train": syn.assign(s=syn.synth_parent.isin(tr_parents)).groupby("label").s.mean().to_dict()}
    X = lambda d: np.c_[d.text.str.len(), d.text.str.count("\n")]  # noqa: E731
    clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X(va), va.label)
    out["length_only_depth2_tree_macro_f1_test"] = float(f1_score(te.label, clf.predict(X(te)), average="macro"))
    json.dump(out, open("results/leakage_audit.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
