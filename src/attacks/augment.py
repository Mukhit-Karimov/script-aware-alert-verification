"""D3 training data: train + attacked copies of 30% of train rows (A1–A5, random attack and budget), labels unchanged.

A5 here perturbs randomly chosen words (no model queries), so the augmented set is model-independent.
  python -m src.attacks.augment --seed 13    -> data/splits/train_adv_13.parquet
"""
import os
import argparse
import random
import re

import pandas as pd

from src.attacks.a5_wordbug import perturb_word
from src.attacks.char_attacks import BUDGETS, apply

SPLITS = os.environ.get("SPLITS", "data/splits")  # E11: data/splits_strip, data/splits_src

_W = re.compile(r"\w+", re.UNICODE)


def a5_random(text, budget, rng):
    words = list(_W.finditer(text))
    k = max(1, round(budget * len(words))) if words else 0
    for m in sorted(rng.sample(words, min(k, len(words))), key=lambda m: -m.start()):
        text = text[:m.start()] + perturb_word(m.group(), rng) + text[m.end():]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--frac", type=float, default=0.30)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tr = pd.read_parquet(f"{SPLITS}/train.parquet")
    sub = tr.sample(frac=a.frac, random_state=a.seed).copy()
    texts, kinds = [], []
    for t in sub.text:
        atk = rng.choice(["A1", "A2", "A3", "A4", "A5"])
        if atk == "A5":
            b = rng.choice([0.05, 0.10, 0.20])
            texts.append(a5_random(t, b, rng))
        else:
            b = rng.choice(BUDGETS[atk])
            texts.append(apply(atk, t, b, rng.randrange(2 ** 31))[0])
        kinds.append(f"{atk}:{b}")
    sub["text"] = texts
    sub["id"] = sub.id + "_adv"
    sub["augment"] = kinds
    out = pd.concat([tr.assign(augment="none"), sub], ignore_index=True)
    out.to_parquet(f"{SPLITS}/train_adv_{a.seed}.parquet", index=False)
    print(len(tr), "->", len(out), sub.augment.str.split(":").str[0].value_counts().to_dict())


if __name__ == "__main__":
    main()
