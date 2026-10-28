"""E4/E5 outputs from results/E4/<name>/<seed>/metrics_<attack>.json and results/E5/*/<seed>/metrics_test.json.

T4: model x attack (reference budget) - clean acc, robust acc, ASR (mean ± std over seeds), change rate.
T5: defenses D0–D4 x attacks - robust accuracy + clean macro-F1.
F5: robust accuracy vs budget, panels A1–A4, one line per model (mean over seeds, band = min–max of seeds... see note).
F6: ASR heatmap model x attack at the reference budget.  F8: dumbbell D0 -> D4 per attack (M6).  F9: clean F1 per defense.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.eval.stats import bootstrap_rate
from src.viz.style import DEF_STYLE, FULL_W, MODEL_NAME, MODEL_STYLE, save, setup

REF = {"A1": 0.10, "A2": 5, "A3": 0.5, "A4": 1.0, "A5": 0.2, "A6": 10, "A7_official": 1, "A7_neutral": 1}
ATK_NAME = {"A1": "A1 homoglyph", "A2": "A2 invisible", "A3": "A3 translit.", "A4": "A4 kk→ru letters", "A5": "A5 typos",
            "A6": "A6 greedy", "A7_official": "A7 para. (official)", "A7_neutral": "A7 para. (neutral)"}
BUDGET_LABEL = {"A1": "Chars changed (%)", "A2": "Insertions (count)", "A3": "Words translit. (%)",
                "A4": "Words changed (%)"}
DEFENSES = {"D0": "M6", "D1": "M6+naive", "D2": "M6+aware", "D3": "M6_D3", "D4": "M6_D3+aware"}
DEF_NAME = {"D0": "D0 none", "D1": "D1 naive norm.", "D2": "D2 lang-aware norm.", "D3": "D3 adv. training", "D4": "D4 = D2+D3"}


def collect():
    rows = []
    for f in glob.glob("results/E4/*/*/metrics_*.json"):
        name, seed = f.split("/")[2], int(f.split("/")[3])
        m = json.load(open(f))
        for r in m["results"]:
            rows.append({"name": name, "seed": seed, "clean_acc": m["clean_acc_noncredible"], **r})
    return pd.DataFrame(rows)


def asr_ci(name, seed, atk, b):
    p = f"results/E4/{name}/{seed}/predictions_{atk}_{b}.parquet"
    if not os.path.exists(p):
        return np.nan, np.nan
    d = pd.read_parquet(p)
    _, lo, hi = bootstrap_rate((d.y_pred == "credible").values, n=1000)
    return lo, hi


VER = set(pd.read_parquet("data/splits/test_real_verified.parquet", columns=["id"]).id) if os.path.exists("data/splits/test_real_verified.parquet") else set()


def asr_verified(name, atk, b):
    """ASR over the audited real misinfo (mean over available seeds) and its n."""
    vals, n = [], 0
    for p in glob.glob(f"results/E4/{name}/*/predictions_{atk}_{b}.parquet"):
        d = pd.read_parquet(p)
        d = d[d.id.isin(VER)]
        if len(d):
            vals.append(float((d.y_pred == "credible").mean())); n = len(d)
    return (float(np.mean(vals)) if vals else np.nan), n


def table_t4(df):
    out = []
    names = [n for n in ["M1", "M4", "M6", "M8", "M9_qwen3.5-35b-a3b"] if n in set(df.name)]
    for n in names:
        for atk, b in REF.items():
            d = df[(df.name == n) & (df.attack == atk) & np.isclose(df.budget.astype(float), float(b))]
            if d.empty:
                continue
            lo, hi = asr_ci(n, 13, atk, b)
            av, nv = asr_verified(n, atk, b)
            out.append({"asr_real_verified": av, "n_real_verified": nv,"model": n.split("_")[0], "attack": atk, "budget": b, "n_seeds": d.seed.nunique(),
                        "clean_acc": d.clean_acc.mean(), "robust_acc": d.robust_acc_noncredible.mean(),
                        "robust_acc_std": d.robust_acc_noncredible.std(ddof=1) if len(d) > 1 else 0.0,
                        "asr": d.asr.mean(), "asr_std": d.asr.std(ddof=1) if len(d) > 1 else 0.0,
                        "asr_ci_lo_seed13": lo, "asr_ci_hi_seed13": hi, "asr_real": d.asr_real.mean(),
                        "n_real": d.n_real.iloc[0], "asr_kk": d.asr_lang_kk.mean(), "asr_ru": d.asr_lang_ru.mean(),
                        "frac_chars": d.mean_frac_chars.mean(), "frac_words": d.mean_frac_words.mean(),
                        "queries": d.queries_per_example.mean(), "n_attacked": d.n_evaluated.iloc[0]})
    t = pd.DataFrame(out)
    t.to_csv("tables/T4_attacks.csv", index=False)
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Model & Attack (budget) & Clean acc. & Robust acc. & ASR & ASR real \\", r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{MODEL_NAME.get(r.model, r.model)} & {ATK_NAME[r.attack]} ({r.budget}) & {r.clean_acc:.3f} & "
                     f"{r.robust_acc:.3f} $\\pm$ {r.robust_acc_std:.3f} & {r.asr:.3f} & "
                     f"{'--' if pd.isna(r.asr_real_verified) else f'{r.asr_real_verified:.3f}'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("tables/T4_attacks.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return t


def fig_f5(df):
    setup()
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, float(os.environ.get("FIG_F5_H", 2.2))), sharey=True)  # FIG_F5_H: paper build
    for ax, atk in zip(axes, ["A1", "A2", "A3", "A4"]):
        for n in ["M1", "M4", "M6", "M8"]:
            d = df[(df.name == n) & (df.attack == atk)]
            if d.empty:
                continue
            g = d.groupby("budget").robust_acc_noncredible
            mean, lo, hi = g.mean(), g.min(), g.max()
            clean = d.clean_acc.mean()
            xs = np.r_[0, mean.index.astype(float)]
            xs_plot = xs * (100 if atk in ("A1", "A3", "A4") else 1)
            c, mk, ls = MODEL_STYLE[n]
            ax.plot(xs_plot, np.r_[clean, mean.values], color=c, marker=mk, ls=ls, markersize=3.5, label=MODEL_NAME[n])
            ax.fill_between(xs_plot[1:], lo.values, hi.values, color=c, alpha=0.18, linewidth=0)
        ax.set_title(ATK_NAME[atk], loc="left")
        ax.set_xlabel(BUDGET_LABEL[atk], fontsize=8)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Robust accuracy (misinfo + forged)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    save(fig, "F5_budget_curves", "Four panels (homoglyphs, invisible characters, transliteration, Kazakh-to-Russian letter "
         "substitution) plotting robust accuracy on misinfo and forged alerts against attack budget, one line per model; "
         "bands span the three seeds; budget 0 is clean accuracy.")


def fig_f6(df):
    setup()
    names = [n for n in ["M1", "M4", "M6", "M8", "M9_qwen3.5-35b-a3b"] if n in set(df.name)]
    atks = [a for a in REF if not df[df.attack == a].empty]
    M = np.full((len(names), len(atks)), np.nan)
    for i, n in enumerate(names):
        for j, a in enumerate(atks):
            d = df[(df.name == n) & (df.attack == a) & np.isclose(df.budget.astype(float), float(REF[a]))]
            if len(d):
                M[i, j] = d.asr.mean()
    fig, ax = plt.subplots(figsize=(FULL_W, 0.42 * len(names) + 1.0))
    ax.imshow(M, cmap="Oranges", vmin=0, vmax=0.3, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "n/a" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if (not np.isnan(M[i, j]) and M[i, j] > 0.2) else "#222222")
    ax.set_xticks(range(len(atks)), [ATK_NAME[a].replace("para. ", "").replace(" ", "\n", 1) + f"\n({REF[a]})" for a in atks],
                  rotation=0, fontsize=8)
    ax.set_yticks(range(len(names)), [MODEL_NAME[n.split('_')[0]] for n in names])
    ax.grid(False)
    save(fig, "F6_asr_heatmap", "Heatmap of attack success rate (share of correctly classified misinfo/forged messages "
         "flipped to credible) for each model and attack at a reference budget; darker cells are more dangerous attacks.")


def clean_f1_defenses():
    rows = []
    for d, name in DEFENSES.items():
        exp = "E1" if name == "M6" else "E5"
        for f in glob.glob(f"results/{exp}/{name}/*/metrics_test.json"):
            m = json.load(open(f))
            rows.append({"defense": d, "seed": int(f.split("/")[3]), "macro_f1": m["all"]["macro_f1"],
                         "macro_f1_real": m.get("real_only", {}).get("macro_f1")})
    return pd.DataFrame(rows)


def table_t5_f8_f9(df):
    cf = clean_f1_defenses()
    rows = []
    for d, name in DEFENSES.items():
        for atk, b in REF.items():
            x = df[(df.name == name) & (df.attack == atk) & np.isclose(df.budget.astype(float), float(b))]
            if len(x):
                rows.append({"defense": d, "attack": atk, "budget": b, "robust_acc": x.robust_acc_noncredible.mean(),
                             "robust_acc_std": x.robust_acc_noncredible.std(ddof=1) if len(x) > 1 else 0.0,
                             "asr": x.asr.mean(), "n_seeds": x.seed.nunique()})
    t = pd.DataFrame(rows)
    c = cf.groupby("defense").agg(clean_f1=("macro_f1", "mean"), clean_f1_std=("macro_f1", "std"),
                                  clean_f1_real=("macro_f1_real", "mean")).reset_index()
    t.to_csv("tables/T5_defenses.csv", index=False)
    c.to_csv("tables/T5_defenses_clean.csv", index=False)
    if len(t):
        piv = t.pivot(index="defense", columns="attack", values="robust_acc")
        lines = [r"\begin{tabular}{l" + "r" * (len(piv.columns) + 1) + "}", r"\toprule",
                 "Defense & Clean F1 & " + " & ".join(ATK_NAME[a] for a in piv.columns) + r" \\", r"\midrule"]
        cc = c.set_index("defense")
        for dfn in piv.index:
            cl = f"{cc.loc[dfn].clean_f1:.3f}" if dfn in cc.index else "--"
            lines.append(f"{DEF_NAME[dfn]} & {cl} & " + " & ".join("--" if pd.isna(v) else f"{v:.3f}" for v in piv.loc[dfn]) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open("tables/T5_defenses.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    setup()
    if len(t):  # F8 dumbbell
        atks = [a for a in REF if not t[(t.attack == a) & t.defense.isin(["D0", "D4"])].empty]
        fig, ax = plt.subplots(figsize=(FULL_W * 0.6, float(os.environ.get("FIG_ROW_H", 0.3)) * len(atks) + 0.8))  # FIG_ROW_H: paper build
        for i, a in enumerate(atks):
            r = t[t.attack == a].set_index("defense").robust_acc
            if "D0" in r and "D4" in r:
                ax.plot([r["D0"], r["D4"]], [i, i], color="#BBBBBB", lw=2, zorder=1)
            for d in ("D0", "D2", "D3", "D4"):
                if d in r:
                    col, mk = DEF_STYLE[d]
                    ax.scatter(r[d], i, color=col, marker=mk, s=28 if d in ("D0", "D4") else 16, zorder=3,
                               label=DEF_NAME[d] if i == 0 else None, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(atks)), [f"{ATK_NAME[a]} ({REF[a]})" for a in atks])
        ax.set_xlabel("Robust accuracy (misinfo + forged)")
        ax.set_xlim(0, 1.02)
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(1.0, 0.0))
        ax.grid(axis="y", visible=False)
        save(fig, "F8_defenses_dumbbell", "Dumbbell chart: for each attack, robust accuracy of XLM-R without defense (D0) and "
             "with the full method (D4), with the normalization-only (D2) and adversarial-training-only (D3) variants as smaller points.")
    if len(cf):  # F9
        fig, ax = plt.subplots(figsize=(FULL_W * 0.5, 1.9))
        g = cf.groupby("defense").macro_f1
        ds = [d for d in DEFENSES if d in g.groups]
        for i, d in enumerate(ds):
            v = g.get_group(d)
            col, mk = DEF_STYLE[d]
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, fmt=mk, color=col, capsize=2)
        ax.set_xticks(range(len(ds)), [DEF_NAME[d].split(" ", 1)[0] for d in ds])
        ax.set_ylabel("Clean macro-F1 (mean ± sd, 3 seeds)")
        save(fig, "F9_defense_clean_cost", "Clean-test macro-F1 of XLM-R under each defense D0–D4 (mean and standard "
             "deviation over three seeds), showing the cost of each defense on unattacked messages.")
    return t, c


def main():
    os.makedirs("tables", exist_ok=True)
    df = collect()
    if df.empty:
        print("no E4 results yet"); return
    print(table_t4(df).round(3).to_string(index=False))
    fig_f5(df); fig_f6(df)
    t, c = table_t5_f8_f9(df)
    print(c.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
