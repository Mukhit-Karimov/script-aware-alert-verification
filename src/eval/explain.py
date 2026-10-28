"""E9 explainability.

  python -m src.eval.explain m1      # top-30 features per class of M1 (LR coefficients) -> results/E9/m1_top_features.csv
  python -m src.eval.explain shap    # SHAP Partition explainer for M6 (seed 13) on 20 test examples -> results/E9/shap.json
SHAP sample: 5 correct forged_alert, 5 errors, 5 attacked (A1 @10%) before/after pairs, 5 misinfo.
"""
import json
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd

from src.eval.metrics import LABELS


def m1_top(k=30):
    m = pickle.load(open("results/E1/M1/13/model.pkl", "rb"))
    names = np.r_[[f"w:{t}" for t in m.wv.get_feature_names_out()], [f"c:{t}" for t in m.cv.get_feature_names_out()]]
    coef = m.clf.coef_
    rows = []
    for ci, cls in enumerate(m.clf.classes_):
        for j in np.argsort(-coef[ci])[:k]:
            rows.append({"class": cls, "feature": names[j], "coef": float(coef[ci, j])})
    os.makedirs("results/E9", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/E9/m1_top_features.csv", index=False)
    print(pd.DataFrame(rows).groupby("class").head(8).to_string(index=False))


def shap_m6(seed=13):
    import shap
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.attacks.char_attacks import apply
    from src.models.transformer import predict_proba
    torch.set_num_threads(int(os.environ.get("NTHREADS", 28)))
    d = f"results/E1/M6/{seed}"
    tok = AutoTokenizer.from_pretrained(f"{d}/model")
    mdl = AutoModelForSequenceClassification.from_pretrained(f"{d}/model")
    pr = pd.read_parquet(f"{d}/predictions_test.parquet")
    te = pd.read_parquet("data/splits/test.parquet").set_index("id")
    rng = np.random.RandomState(0)
    pick = lambda q, n: list(pr[q].sample(min(n, q.sum()), random_state=rng).id)
    sel = {"correct_forged": pick((pr.y_true == "forged_alert") & (pr.y_pred == pr.y_true), 5),
           "errors": pick(pr.y_pred != pr.y_true, 5),
           "misinfo": pick((pr.y_true == "misinfo") & (pr.y_pred == pr.y_true), 5)}
    # attacked pairs: prefer examples where A1 @10% flips M6 to credible (from E4), else any correct non-credible
    flipped = []
    p = f"results/E4/M6/{seed}/predictions_A1_0.1.parquet"
    if os.path.exists(p):
        e4 = pd.read_parquet(p)
        flipped = list(e4[e4.y_pred == "credible"].sample(min(5, (e4.y_pred == "credible").sum()), random_state=0).id)
    if len(flipped) < 5:
        flipped += pick((pr.y_true != "credible") & (pr.y_pred == pr.y_true) & ~pr.id.isin(flipped), 5 - len(flipped))
    f = lambda texts: predict_proba(mdl, tok, list(texts))
    masker = shap.maskers.Text(tok)
    expl = shap.Explainer(f, masker, output_names=LABELS, algorithm="partition")
    items = [(g, i, te.loc[i, "text"][:600]) for g, ids in sel.items() for i in ids]
    for i in flipped:
        t = te.loc[i, "text"][:600]
        items.append(("attacked_before", i, t))
        items.append(("attacked_after", i, apply("A1", t, 0.10, random.Random(i).randrange(2 ** 31))[0]))
    out = []
    for g, i, t in items:
        sv = expl([t], max_evals=500)
        P = f([t])[0]
        out.append({"group": g, "id": i, "label": te.loc[i, "label"], "pred": LABELS[int(P.argmax())], "proba": P.tolist(),
                    "tokens": list(sv.data[0]), "shap": np.asarray(sv.values[0]).tolist()})
        print(g, i, out[-1]["pred"], flush=True)
    json.dump(out, open("results/E9/shap.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    {"m1": m1_top, "shap": shap_m6}[sys.argv[1]]()
