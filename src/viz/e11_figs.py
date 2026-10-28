"""E11 (source-marker ablation): T9 original text vs stripped text, F20 dumbbell original -> stripped.

Sources: results/E1|E5 (original clean), results/E11_strip (stripped clean + E4 attacks), results/E11_orig/E4 (original
models attacked at the E11 budgets), results/E11_src (source split: credible of other channels only in test).
  python -m src.viz.e11_figs
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz.style import DEF_STYLE, FULL_W, MODEL_STYLE, OI, save, setup

CFG = {"M1": ("M1", "TF-IDF+LR (D0)"), "M6": ("M6", "XLM-R (D0)"), "D4": ("M6_D3+aware", "XLM-R + D4")}
CLEAN_SRC = {"orig": {"M1": "results/E1/M1", "M6": "results/E1/M6", "D4": "results/E5/M6_D3+aware"},
             "strip": {"M1": "results/E11_strip/M1", "M6": "results/E11_strip/M6", "D4": "results/E11_strip/M6_D3+aware"}}
ATK_SRC = {"orig": "results/E11_orig/E4", "strip": "results/E11_strip/E4"}
BUDGETS = [("A1", 0.1), ("A1", 0.2), ("A2", 10), ("A2", 20), ("A3", 0.1), ("A3", 0.2), ("A3", 1.0), ("A4", 0.1), ("A4", 0.2), ("A4", 1.0)]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def clean_rows():
    rows = []
    for var, d in CLEAN_SRC.items():
        for cfg, path in d.items():
            for f in glob.glob(f"{path}/*/metrics_test.json"):
                m = jl(f)
                rows.append({"variant": var, "cfg": cfg, "seed": f.split("/")[-2], "macro_f1": m["all"]["macro_f1"],
                             "recall_forged": m["all"].get("recall_forged"), "mcc": m["all"]["mcc"],
                             "macro_f1_real_verified": m.get("real_verified", {}).get("macro_f1")})
    return pd.DataFrame(rows)


def attack_rows():
    rows = []
    for var, root in ATK_SRC.items():
        for cfg, (name, _) in CFG.items():
            for f in glob.glob(f"{root}/{name}/*/metrics_A*.json"):
                for r in jl(f)["results"]:
                    rows.append({"variant": var, "cfg": cfg, "seed": f.split("/")[-2], "attack": r["attack"],
                                 "budget": float(r["budget"]), "robust_acc": r["robust_acc_noncredible"], "asr": r["asr"]})
    return pd.DataFrame(rows)


def src_rows():
    rows = []
    for m in ["M1", "M6"]:
        for f in glob.glob(f"results/E11_src/{m}/*/metrics_test.json"):
            j = jl(f)
            pc = j["all"]["per_class"]
            rows.append({"model": m, "seed": f.split("/")[-2], "macro_f1": j["all"]["macro_f1"], "mcc": j["all"]["mcc"],
                         "recall_credible": pc["credible"]["recall"], "recall_misinfo": pc["misinfo"]["recall"],
                         "recall_forged": pc["forged_alert"]["recall"], "n": pc["credible"]["support"] + pc["misinfo"]["support"] + pc["forged_alert"]["support"]})
    return pd.DataFrame(rows)


def agg(df, keys, cols):
    g = df.groupby(keys)[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).join(g.size().rename("n_seeds"))
    return out.reset_index()


def main():
    setup()
    os.makedirs("results/E11", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    c = clean_rows()
    a = attack_rows()
    s = src_rows()
    ca = agg(c, ["cfg", "variant"], ["macro_f1", "recall_forged", "mcc", "macro_f1_real_verified"]) if len(c) else pd.DataFrame()
    aa = agg(a, ["cfg", "variant", "attack", "budget"], ["robust_acc", "asr"]) if len(a) else pd.DataFrame()
    ca.to_csv("results/E11/clean_summary.csv", index=False)
    aa.to_csv("results/E11/attack_summary.csv", index=False)
    if len(s):
        agg(s, ["model"], ["macro_f1", "mcc", "recall_credible", "recall_misinfo", "recall_forged"]).to_csv(
            "results/E11/source_split_summary.csv", index=False)

    # T9: one row per (config, metric); original vs stripped (mean ± std over seeds)
    t = []
    fmt = lambda m, sd: "—" if pd.isna(m) else (f"{m:.3f}" if pd.isna(sd) else f"{m:.3f} ± {sd:.3f}")  # noqa: E731
    for cfg, (_, label) in CFG.items():
        for met, nice in [("macro_f1", "macro-F1 (clean)"), ("macro_f1_real_verified", "macro-F1 real_verified"),
                          ("recall_forged", "Recall(forged)"), ("mcc", "MCC")]:
            r = {v: ca[(ca.cfg == cfg) & (ca.variant == v)] for v in ["orig", "strip"]}
            if any(len(x) == 0 for x in r.values()):
                continue
            t.append({"config": label, "metric": nice, **{v: fmt(r[v][f"{met}_mean"].iloc[0], r[v][f"{met}_std"].iloc[0]) for v in r},
                      **{f"{v}_mean": r[v][f"{met}_mean"].iloc[0] for v in r}})
        for atk, b in (BUDGETS if len(aa) else []):
            for met, nice in [("robust_acc", "robust acc"), ("asr", "ASR")]:
                r = {v: aa[(aa.cfg == cfg) & (aa.variant == v) & (aa.attack == atk) & np.isclose(aa.budget, b)] for v in ["orig", "strip"]}
                if any(len(x) == 0 for x in r.values()):
                    continue
                t.append({"config": label, "metric": f"{nice} {atk}@{b:g}",
                          **{v: fmt(r[v][f"{met}_mean"].iloc[0], r[v][f"{met}_std"].iloc[0]) for v in r},
                          **{f"{v}_mean": r[v][f"{met}_mean"].iloc[0] for v in r}})
    T9 = pd.DataFrame(t)
    if len(T9):
        T9["delta"] = T9.strip_mean - T9.orig_mean
        T9.to_csv("tables/T9_source_ablation.csv", index=False)
        lines = [r"\begin{tabular}{llll}", r"\toprule", r"Configuration & Metric & Original text & Markers stripped \\", r"\midrule"]
        for r in T9.itertuples():
            lines.append(f"{r.config} & {r.metric.replace('_', r'\_')} & {r.orig} & {r.strip} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open("tables/T9_source_ablation.tex", "w").write("\n".join(lines) + "\n")

    # F20: dumbbell original -> stripped; left: clean macro-F1, right: robust accuracy under attack
    if len(T9):
        sel_c = T9[T9.metric == "macro-F1 (clean)"]
        sel_a = T9[T9.metric.isin(["robust acc A1@0.1", "robust acc A1@0.2", "robust acc A3@1", "robust acc A4@1"])]
        fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"width_ratios": [1, 1.6]})
        for ax, sel, xl in [(axes[0], sel_c, "Macro-F1 on clean test"), (axes[1], sel_a, "Robust accuracy (misinfo + forged)")]:
            labels = [f"{r.config}" if ax is axes[0] else f"{r.config} · {r.metric.replace('robust acc ', '')}" for r in sel.itertuples()]
            for i, r in enumerate(sel.itertuples()):
                key = "D4" if "D4" in r.config else ("M1" if "LR" in r.config else "M6")
                col = DEF_STYLE["D4"][0] if key == "D4" else MODEL_STYLE[key][0]
                ax.plot([r.orig_mean, r.strip_mean], [i, i], color="#BBBBBB", lw=2, zorder=1)
                ax.scatter(r.orig_mean, i, s=30, facecolor="white", edgecolor=col, lw=1.4, zorder=2, label="original text" if i == 0 else None)
                ax.scatter(r.strip_mean, i, s=30, color=col, zorder=3, label="markers stripped" if i == 0 else None)
            ax.set_yticks(range(len(sel)), labels)
            ax.set_xlabel(xl)
            ax.invert_yaxis()
        lo = min(sel_c.orig_mean.min(), sel_c.strip_mean.min())
        axes[0].set_xlim(max(0.0, lo - 0.01), 1.0)
        axes[1].set_xlim(0, 1.02)
        h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#444444", label="original text"),
             plt.Line2D([], [], marker="o", ls="", color="#444444", label="markers stripped")]
        fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
        fig.tight_layout(w_pad=1.5)
        save(fig, "F20_source_ablation", "Dumbbell chart comparing each configuration on the original text (hollow marker) "
             "and after stripping source markers (filled marker): clean macro-F1 on the left, robust accuracy under character "
             "attacks on the right; short bars mean the result does not depend on channel markers.")
    print(T9.to_string() if len(T9) else "no E11 results yet")


if __name__ == "__main__":
    main()
