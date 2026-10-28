"""T1 (corpus statistics) and F1 (corpus composition) from results/data_stats.json + data/processed/corpus.parquet."""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.viz.style import FULL_W, LABEL_COLOR, LABEL_NAME, save, setup

LABELS = ["credible", "misinfo", "forged_alert"]
LANGS = ["kk", "ru", "mixed"]


def table_t1(c):
    rows = []
    for l in LABELS:
        for g in LANGS:
            d = c[(c.label == l) & (c.lang == g)]
            if not len(d):
                continue
            rows.append({"label": LABEL_NAME[l], "lang": g, "n": len(d), "real": int((~d.is_synthetic).sum()),
                         "synthetic": int(d.is_synthetic.sum()), "mean_len_chars": round(d.text.str.len().mean(), 1)})
    t = pd.DataFrame(rows)
    tot = {"label": "total", "lang": "", "n": len(c), "real": int((~c.is_synthetic).sum()),
           "synthetic": int(c.is_synthetic.sum()), "mean_len_chars": round(c.text.str.len().mean(), 1)}
    t = pd.concat([t, pd.DataFrame([tot])], ignore_index=True)
    os.makedirs("tables", exist_ok=True)
    t.to_csv("tables/T1_corpus.csv", index=False)
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Label & Lang & $N$ & Real & Synthetic & Mean length (chars) \\", r"\midrule"]
    for _, r in t.iterrows():
        if r.label == "total":
            lines.append(r"\midrule")
        lines.append(f"{r.label} & {r.lang} & {r.n:,} & {r.real:,} & {r.synthetic:,} & {r.mean_len_chars} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("tables/T1_corpus.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return t


def fig_f1(c):
    setup()
    fig, ax = plt.subplots(figsize=(FULL_W * 0.62, 2.3))
    x = np.arange(len(LANGS))
    w = 0.26
    for i, l in enumerate(LABELS):
        real = [int(((c.label == l) & (c.lang == g) & ~c.is_synthetic).sum()) for g in LANGS]
        syn = [int(((c.label == l) & (c.lang == g) & c.is_synthetic).sum()) for g in LANGS]
        xs = x + (i - 1) * w
        ax.bar(xs, real, w * 0.92, color=LABEL_COLOR[l], edgecolor="white", linewidth=0.8,
               label=f"{LABEL_NAME[l]} (real)" if sum(real) else None)
        ax.bar(xs, syn, w * 0.92, bottom=real, color="white", edgecolor=LABEL_COLOR[l], hatch="////", linewidth=0.8,
               label=f"{LABEL_NAME[l]} (synthetic)" if sum(syn) else None)
        for k, (xi, r, s) in enumerate(zip(xs, real, syn)):
            if r + s:
                # forged-alert labels sit above the neighbouring misinfo label so the two never collide
                y = r + s if i != 2 else max(r + s, prev[k] + 650)
                ax.text(xi, y + 120, f"{r + s:,}", ha="center", va="bottom", fontsize=8, color="#333333")
        prev = [a + b for a, b in zip(real, syn)]
    ax.set_xticks(x, ["Kazakh (kk)", "Russian (ru)", "mixed"])
    ax.set_ylabel("Messages")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=1, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    save(fig, "F1_corpus", "Grouped bar chart of corpus size by label (credible, misinfo, forged alert) and language "
         "(Kazakh, Russian, mixed); solid bars are real messages and hatched bars are synthetic ones. Credible messages "
         "dominate in both main languages, all forged alerts are synthetic, and mixed-language texts are rare.")


def main():
    c = pd.read_parquet("data/processed/corpus.parquet")
    c = c[c.lang.isin(LANGS)]
    print(table_t1(c).to_string(index=False))
    fig_f1(c)


if __name__ == "__main__":
    main()
