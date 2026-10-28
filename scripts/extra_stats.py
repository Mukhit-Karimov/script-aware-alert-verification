"""E10 McNemar M6+D4 vs M6 on attacked data, T8 attack examples, results/E4/samples.csv for author review.

McNemar is paired over all non-credible test messages: a model is "correct" on a message if it was classified correctly
before the attack and is still correct after it (same definition as robust_acc_noncredible).
  python scripts/extra_stats.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.eval.stats import holm, mcnemar  # noqa: E402

REF = {"A1": "0.1", "A2": "5", "A3": "0.5", "A4": "1.0", "A5": "0.2", "A6": "10", "A7_official": "1", "A7_neutral": "1"}


def correct_vec(name, seed, atk, ids):
    f = f"results/E4/{name}/{seed}/predictions_{atk}_{REF[atk]}.parquet"
    if not os.path.exists(f):
        return None
    p = pd.read_parquet(f).set_index("id")
    ok = (p.y_pred == p.y_true)
    return np.array([bool(ok.get(i, False)) for i in ids])


def mcnemar_attacked():
    te = pd.read_parquet("data/splits/test.parquet")
    ids = te[te.label != "credible"].id.tolist()
    rows = []
    par = pd.read_parquet("results/E4/A7_paraphrases.parquet")
    for atk in REF:
        ids_a = ids
        if atk.startswith("A7"):  # A7 was run on a 300-message subsample: pair only over kept paraphrases of this style
            kept = set(par[par.keep & (par["style"] == atk.split("_")[1])].id)
            ids_a = [i for i in ids if i in kept]
        for seed in [13, 42, 2026]:
            a, b = correct_vec("M6", seed, atk, ids_a), correct_vec("M6_D3+aware", seed, atk, ids_a)
            if a is None or b is None:
                continue
            chi2, p, n01, n10 = mcnemar(a, b)
            rows.append({"attack": atk, "budget": REF[atk], "seed": seed, "n": len(ids_a), "acc_D0": float(a.mean()),
                         "acc_D4": float(b.mean()), "D0_only_correct": n01, "D4_only_correct": n10, "chi2": chi2, "p": p})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p.values)
    df["stars"] = np.select([df.p_holm < 0.001, df.p_holm < 0.01, df.p_holm < 0.05], ["***", "**", "*"], "")
    df.to_csv("results/E10_mcnemar_D4_vs_D0_attacked.csv", index=False)
    print(df.to_string())


def samples():
    rng = 0
    s = [pd.read_csv(f).assign(model=f.split("/")[-3])
         for f in glob.glob("results/E4/*/*/samples_A[1-4].csv")]
    a14 = pd.concat(s).drop_duplicates(["original", "adversarial"]).sample(50, random_state=rng)
    a7 = pd.concat([pd.read_csv(f).assign(model=f.split("/")[-3]) for f in glob.glob("results/E4/*/*/samples_A7_*.csv")])
    a7 = a7.drop_duplicates(["original", "adversarial"]).assign(attack_success=True)
    n = min(50, len(a7))
    a7 = a7.sample(n, random_state=rng)
    if n < 50:  # fill with kept paraphrases that did not flip the model: meaning check still needed for A7
        par = pd.read_parquet("results/E4/A7_paraphrases.parquet")
        par = par[par.keep & ~par.text_adv.isin(a7.adversarial)].sample(50 - n, random_state=rng)
        a7 = pd.concat([a7, pd.DataFrame({"attack": "A7_" + par["style"], "budget": 1, "label": par.label,
                                          "original": par.text_orig, "adversarial": par.text_adv, "model": "",
                                          "attack_success": False})])
    out = pd.concat([a14.assign(attack_success=True), a7])
    out.assign(meaning_kept="", readable="").to_csv("results/E4/samples.csv", index=False)
    print("samples.csv", len(out), out.attack.value_counts().to_dict())


def t8():
    """5 attack examples for M6 seed 13 (A1, A2, A3, A4, A7_official): flipped to credible by D0, caught by D4."""
    te = pd.read_parquet("data/splits/test.parquet").set_index("id")
    clean = pd.read_parquet("results/E1/M6/13/predictions_test.parquet").set_index("id")
    rows = []
    for atk in ["A1", "A2", "A3", "A4", "A7_official"]:
        d0 = pd.read_parquet(f"results/E4/M6/13/predictions_{atk}_{REF[atk]}.parquet").set_index("id")
        d4 = pd.read_parquet(f"results/E4/M6_D3+aware/13/predictions_{atk}_{REF[atk]}.parquet").set_index("id")
        cand = d0[(d0.y_pred == "credible") & ~d0.is_synthetic.astype(bool)]
        if cand.empty:
            cand = d0[d0.y_pred == "credible"]
        cand = cand[cand.index.isin(d4.index)]
        if cand.empty:
            continue
        cand = cand.assign(L=te.loc[cand.index, "text"].str.len()).sort_values("L")
        i = cand.index[0]
        rows.append({"attack": atk, "budget": REF[atk], "id": i, "label": te.loc[i, "label"], "lang": te.loc[i, "lang"],
                     "is_synthetic": bool(te.loc[i, "is_synthetic"]), "original": te.loc[i, "text"],
                     "adversarial": d0.loc[i, "text_adv"], "pred_clean_D0": clean.loc[i, "y_pred"],
                     "pred_adv_D0": d0.loc[i, "y_pred"], "p_credible_adv_D0": float(d0.loc[i, "p_credible"]),
                     "pred_adv_D4": d4.loc[i, "y_pred"], "p_credible_adv_D4": float(d4.loc[i, "p_credible"])})
    df = pd.DataFrame(rows)
    os.makedirs("tables", exist_ok=True)
    df.to_csv("tables/T8_attack_examples.csv", index=False)

    def esc(t, n=110):
        t = t if len(t) <= n else t[:n] + "…"
        return t.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_") \
                .replace("#", r"\#").replace("$", r"\$")
    lines = [r"\begin{tabular}{p{1.2cm}p{4.6cm}p{4.6cm}ll}", r"\toprule",
             r"Attack & Original & Attacked & D0 & D4 \\", r"\midrule"]
    for r in df.itertuples():
        lines.append(f"{esc(r.attack)} & {esc(r.original)} & {esc(r.adversarial)} & {r.pred_clean_D0}$\\to${r.pred_adv_D0} "
                     f"& {r.pred_adv_D4} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("tables/T8_attack_examples.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(df[["attack", "label", "lang", "is_synthetic", "pred_adv_D0", "pred_adv_D4"]].to_string())


def budget_summary():
    """Mean ± std over seeds of robust accuracy / ASR for every model, defense, attack and budget (A1–A4)."""
    rows = []
    for f in glob.glob("results/E4/*/*/metrics_A[1-4].json"):
        name, seed = f.split("/")[2], f.split("/")[3]
        for r in json.load(open(f))["results"]:
            rows.append(dict(name=name, seed=seed, attack=r["attack"], budget=r["budget"], rob=r["robust_acc_noncredible"], asr=r["asr"]))
    d = pd.DataFrame(rows)
    g = d.groupby(["name", "attack", "budget"]).agg(rob=("rob", "mean"), rob_std=("rob", "std"), asr=("asr", "mean"),
                                                   asr_std=("asr", "std"), n_seeds=("seed", "nunique")).reset_index()
    g.to_csv("results/E5/budget_summary_all.csv", index=False)


if __name__ == "__main__":
    # "samples" rebuilds results/E4/samples.csv from scratch and drops the automatic check columns: run it only on purpose
    steps = sys.argv[1:] or ["mcnemar", "t8", "budget"]
    for st in steps:
        {"mcnemar": mcnemar_attacked, "samples": samples, "t8": t8, "budget": budget_summary}[st]()
