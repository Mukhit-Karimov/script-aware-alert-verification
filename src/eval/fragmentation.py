"""E9: subword fragmentation before/after character attacks (mechanism of the vulnerability) -> F7.

For test misinfo/forged messages: mean number of subword tokens per whitespace word, clean vs attacked (A1–A4 at the
reference budget), per tokenizer (XLM-R, mBERT, mDeBERTa, Kaz-RoBERTa), and only over the words the attack changed.
  python -m src.eval.fragmentation
"""
import json
import os
import random
import re

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.attacks.char_attacks import apply

TOKS = {"M6": "models/xlm-roberta-base", "M5": "models/bert-base-multilingual-cased",
        "M7": "models/mdeberta-v3-base", "M8": "models/kaz-roberta-conversational"}
REF = {"A1": 0.10, "A2": 5, "A3": 0.5, "A4": 1.0}


def main():
    te = pd.read_parquet("data/splits/test.parquet")
    te = te[te.label != "credible"]
    out, rows = {}, []
    for m, path in TOKS.items():
        tok = AutoTokenizer.from_pretrained(path)
        ntok = lambda w: len(tok.tokenize(w)) if w else 0
        out[m] = {}
        for atk, b in REF.items():
            per_word_clean, per_word_adv, changed_clean, changed_adv, unk = [], [], [], [], []
            for rid, t in zip(te.id, te.text):
                adv, _ = apply(atk, t, b, random.Random(rid).randrange(2 ** 31))
                wc, wa = t.split(), adv.split()
                if len(wc) != len(wa):  # attacks are word-aligned; skip the rare misaligned case
                    continue
                for x, y in zip(wc, wa):
                    a, bb = ntok(x), ntok(y)
                    per_word_clean.append(a); per_word_adv.append(bb)
                    if x != y:
                        changed_clean.append(a); changed_adv.append(bb)
                unk.append(sum(tk == tok.unk_token for tk in tok.tokenize(adv)))
            r = {"subtok_per_word_clean": float(np.mean(per_word_clean)), "subtok_per_word_adv": float(np.mean(per_word_adv)),
                 "subtok_per_changed_word_clean": float(np.mean(changed_clean)) if changed_clean else None,
                 "subtok_per_changed_word_adv": float(np.mean(changed_adv)) if changed_adv else None,
                 "n_changed_words": len(changed_clean), "mean_unk_per_text_adv": float(np.mean(unk))}
            out[m][atk] = r
            rows.append({"tokenizer": m, "attack": atk, "budget": b, **r})
            print(m, atk, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    os.makedirs("results/E9", exist_ok=True)
    json.dump(out, open("results/E9/fragmentation.json", "w"), indent=1)
    pd.DataFrame(rows).to_csv("results/E9/fragmentation.csv", index=False)


if __name__ == "__main__":
    main()
