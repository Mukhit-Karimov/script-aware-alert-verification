"""E1 outputs: T2 (models: macro-F1 ± std over seeds, Recall(forged), PR-AUC, MCC, bootstrap CI, significance),
F2 (dot plot with 95% CI), F3 (normalized confusion matrices M1 vs M6). Reads results/E1/*/<seed>/predictions_test.parquet.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.eval.metrics import LABELS
from src.eval.stats import bootstrap_ci, holm, mcnemar, stars
from src.viz.style import FULL_W, HALF_W, LABEL_NAME, MODEL_NAME, MODEL_STYLE, save, setup

ORDER = ["M1", "M2", "M3", "M4", "M5", "M6", "M8", "M9", "M10"]  # M7 excluded (time limit)


def base(name):
    return name.split("_")[0]


def collect(split="test"):
    rows = []
    for d in sorted(glob.glob("results/E1/*/*/")):
        name, seed = d.rstrip("/").split("/")[-2:]
        f = f"{d}/metrics_{split}.json"
        if base(name) not in ORDER or not os.path.exists(f):
            continue
        m = json.load(open(f))
        for sl in ("all", "real_verified"):
            if sl in m:
                s = m[sl]
                rows.append({"model": base(name), "name": name, "seed": int(seed), "slice": sl, "macro_f1": s["macro_f1"],
                             "recall_forged": s.get("recall_forged"), "pr_auc": s.get("pr_auc_macro"), "mcc": s["mcc"],
                             "n": s["n"], "invalid_share": m.get("invalid_share")})
    return pd.DataFrame(rows)


def pooled_ci(model, sl, split="test"):
    """Bootstrap CI on predictions pooled over seeds (each seed's prediction is a resample unit of its own)."""
    ps = [pd.read_parquet(p) for p in sorted(glob.glob(f"results/E1/{model}*/*/predictions_{split}.parquet"))
          if base(p.split("/")[2]) == model]
    if not ps:
        return np.nan, np.nan, np.nan
    p = ps[0]  # CI from the first seed (13); the seed spread is reported separately as ± std
    if sl == "real_verified":
        ver = set(pd.read_parquet("data/splits/test_real_verified.parquet", columns=["id"]).id)
        p = p[p.id.isin(ver)]
    return bootstrap_ci(p.y_true.values, p.y_pred.values, n=1000, seed=0)


def table_t2(df):
    out = []
    for sl in ("all", "real_verified"):
        for m in ORDER:
            d = df[(df.model == m) & (df.slice == sl)]
            if d.empty:
                continue
            pt, lo, hi = pooled_ci(m, sl)
            out.append({"slice": sl, "model": m, "name": MODEL_NAME[m], "n_seeds": d.seed.nunique(),
                        "macro_f1_mean": d.macro_f1.mean(), "macro_f1_std": d.macro_f1.std(ddof=1) if len(d) > 1 else 0.0,
                        "ci_lo": lo, "ci_hi": hi, "recall_forged": d.recall_forged.mean(), "pr_auc": d.pr_auc.mean(),
                        "mcc": d.mcc.mean(), "n_test": int(d.n.iloc[0]), "invalid_share": d.invalid_share.mean()})
    t = pd.DataFrame(out)
    # label-audit sensitivity (task 2): macro-F1 without the confident-learning suspicious test examples
    if os.path.exists("results/E1_label_sensitivity.csv"):
        sv = pd.read_csv("results/E1_label_sensitivity.csv")
        sv = sv[sv.variant == "without_suspicious"].assign(model=lambda d: d.model.map(base),
                                                           slice=lambda d: d.slice.map({"test": "all", "test_real": "real_only"}))
        sv = sv.groupby(["model", "slice"]).agg(macro_f1_no_suspicious=("macro_f1", "mean"), n_no_suspicious=("n", "first")).reset_index()
        t = t.merge(sv, on=["model", "slice"], how="left")
    # McNemar M6 vs every other model on seed-13 predictions (all test), Holm-corrected
    p6 = pd.read_parquet("results/E1/M6/13/predictions_test.parquet").set_index("id") if os.path.exists("results/E1/M6/13/predictions_test.parquet") else None
    tests = []
    if p6 is not None:
        for m in ORDER:
            if m == "M6":
                continue
            fs = [p for p in glob.glob(f"results/E1/{m}*/13/predictions_test.parquet") if base(p.split("/")[2]) == m]
            if not fs:
                continue
            q = pd.read_parquet(fs[0]).set_index("id")
            ids = q.index.intersection(p6.index)
            chi2, p, b, c = mcnemar((p6.loc[ids].y_pred == p6.loc[ids].y_true).values, (q.loc[ids].y_pred == q.loc[ids].y_true).values)
            tests.append({"model": m, "chi2": chi2, "p": p, "n_paired": len(ids), "m6_only_correct": b, "other_only_correct": c})
    tt = pd.DataFrame(tests)
    if len(tt):
        tt["p_holm"] = holm(tt.p.values)
        tt["stars"] = tt.p_holm.map(stars)
    os.makedirs("tables", exist_ok=True)
    t.to_csv("tables/T2_models.csv", index=False)
    tt.to_csv("tables/T2_mcnemar_vs_M6.csv", index=False)
    write_t2_tex(t, tt)
    return t, tt


def write_t2_tex(t, tt):
    st = dict(zip(tt.model, tt.stars)) if len(tt) else {}
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model & Macro-F1 (all) & 95\% CI & Macro-F1 (real) & Recall(forged) & MCC & Macro-F1 w/o suspicious \\", r"\midrule"]
    a, r = t[t.slice == "all"].set_index("model"), t[t.slice == "real_verified"].set_index("model")
    best = a.macro_f1_mean.max()
    for m in a.index:
        x = a.loc[m]
        f1 = f"{x.macro_f1_mean:.3f} $\\pm$ {x.macro_f1_std:.3f}"
        if np.isclose(x.macro_f1_mean, best):
            f1 = r"\textbf{" + f1 + "}"
        rr = f"{r.loc[m].macro_f1_mean:.3f}" if m in r.index else "--"
        ns = f"{x.macro_f1_no_suspicious:.3f}" if "macro_f1_no_suspicious" in x and pd.notna(x.macro_f1_no_suspicious) else "--"
        lines.append(f"{x['name']} ({m}){st.get(m, '')} & {f1} & [{x.ci_lo:.3f}, {x.ci_hi:.3f}] & {rr} & {x.recall_forged:.3f} & "
                     f"{x.mcc:.3f} & {ns} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("tables/T2_models.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")


def fig_f2(t):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), sharey=True)
    a = t[t.slice == "all"].sort_values("macro_f1_mean")
    for ax, sl, title in zip(axes, ("all", "real_verified"), ("All test", "Real test (audited)")):
        d = t[t.slice == sl].set_index("model").loc[[m for m in a.model if m in set(t[t.slice == sl].model)]]
        for i, (m, r) in enumerate(d.iterrows()):
            c, mk, _ = MODEL_STYLE[m]
            ax.errorbar(r.macro_f1_mean, i, xerr=[[r.macro_f1_mean - r.ci_lo], [r.ci_hi - r.macro_f1_mean]], fmt=mk,
                        color=c, ecolor=c, elinewidth=1.2, capsize=2, markersize=5)
        ax.set_yticks(range(len(d)), [f"{MODEL_NAME[m]} ({m})" for m in d.index])
        ax.set_xlabel("Macro-F1 (95% bootstrap CI)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    save(fig, "F2_models", "Dot plot of test macro-F1 with 95% bootstrap confidence intervals for every model, sorted by "
         "macro-F1, shown separately for the full test set and the real-only test subset.")


def fig_f3():
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.8, 2.5))
    for ax, m in zip(axes, ("M1", "M6")):
        p = f"results/E1/{m}/13/predictions_test.parquet"
        if not os.path.exists(p):
            ax.axis("off"); continue
        d = pd.read_parquet(p)
        cm = confusion_matrix(d.y_true, d.y_pred, labels=LABELS, normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 0.6 else "#222222")
        names = [LABEL_NAME[l] for l in LABELS]
        ax.set_xticks(range(3), names, rotation=20)
        ax.set_yticks(range(3), names if m == "M1" else [""] * 3)
        ax.set_xlabel("Predicted")
        if m == "M1":
            ax.set_ylabel("True")
        ax.set_title(f"{MODEL_NAME[m]} ({m})", loc="left")
        ax.grid(False)
    save(fig, "F3_confusion", "Two row-normalized confusion matrices (TF-IDF+LR and XLM-R, seed 13) over credible, misinfo "
         "and forged alert, showing where the classes are confused.")


def main():
    df = collect()
    t, tt = table_t2(df)
    print(t.round(4).to_string(index=False)); print(tt.round(4).to_string(index=False))
    fig_f2(t)
    fig_f3()


if __name__ == "__main__":
    main()
