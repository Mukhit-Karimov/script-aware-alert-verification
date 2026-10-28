"""Shared figure style: Okabe–Ito, fixed entity->color/marker, PDF + 300-dpi PNG + alt text.

Palette validated with the dataviz validator (labels: all checks pass; contrast WARN -> direct labels/markers).
Black is not used as a categorical hue (fails lightness/chroma); it is reserved for reference lines.
"""
import os

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

matplotlib.use("Agg")

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000", "gray": "#8C8C8C"}

LABEL_COLOR = {"credible": OI["blue"], "misinfo": OI["orange"], "forged_alert": OI["purple"]}
LABEL_NAME = {"credible": "credible", "misinfo": "misinfo", "forged_alert": "forged alert"}

# model -> (color, marker, linestyle); M6 is the main model and always blue
MODEL_STYLE = {
    "M1": (OI["orange"], "s", "--"), "M2": (OI["gray"], "v", ":"), "M3": (OI["gray"], "<", ":"),
    "M4": (OI["green"], "^", "-."), "M5": (OI["sky"], "D", "--"), "M6": (OI["blue"], "o", "-"),
    "M7": (OI["gray"], ">", ":"), "M8": (OI["purple"], "P", "-."), "M9": (OI["vermillion"], "X", (0, (3, 1, 1, 1))),
    "M10": (OI["gray"], "*", ":"),
}
MODEL_NAME = {"M1": "TF-IDF+LR", "M2": "TF-IDF+SVM", "M3": "TF-IDF+CNB", "M4": "CNN-BiLSTM-Att", "M5": "mBERT",
              "M6": "XLM-R", "M7": "mDeBERTa-v3", "M8": "Kaz-RoBERTa", "M9": "LLM zero-shot", "M10": "LLM few-shot"}
# defense -> color (consistent across F8/F9/F13); D4 = full method
DEF_STYLE = {"D0": (OI["gray"], "o"), "D1": (OI["orange"], "s"), "D2": (OI["sky"], "^"),
             "D3": (OI["purple"], "D"), "D4": (OI["vermillion"], "P")}

# FIG_FULL_W / FIG_DIR: paper build (acmart manuscript text width 5.95 in) without touching figures/
FULL_W, HALF_W = float(os.environ.get("FIG_FULL_W", 6.5)), 3.2
FIG_DIR = os.environ.get("FIG_DIR", "figures")


def setup():
    fonts = {f.name for f in fm.fontManager.ttflist}
    family = "Linux Libertine O" if "Linux Libertine O" in fonts else "Linux Libertine" if "Linux Libertine" in fonts else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.edgecolor": "#555555", "lines.linewidth": 1.6, "lines.markersize": 5, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    return family


def save(fig, name, alt_text):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(f"{FIG_DIR}/{name}.pdf")
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=300)
    with open(f"{FIG_DIR}/{name}.txt", "w", encoding="utf-8") as f:
        f.write(alt_text.strip() + "\n")
    plt.close(fig)
