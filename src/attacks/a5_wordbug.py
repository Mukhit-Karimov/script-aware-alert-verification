"""A5: DeepWordBug-style black-box attack (Gao et al., 2018), own implementation.

TextAttack is installed with --no-deps (its dependencies do not build on Python 3.12), so the recipe is re-implemented:
  1. word importance = drop in p(true class) when the word is deleted (leave-one-out, one query per word);
  2. the most important words are perturbed with one character edit each (swap adjacent / substitute with a
     keyboard neighbour / delete / insert), up to budget * n_words words; stop early once the label flips.
"""
import re

import numpy as np

_W = re.compile(r"\w+", re.UNICODE)
# neighbouring keys on the Russian ЙЦУКЕН and Latin QWERTY layouts (Kazakh letters use their Russian base key)
_ROWS = ["йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю", "qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGH = {}
for row in _ROWS:
    for i, c in enumerate(row):
        NEIGH[c] = (row[i - 1] if i else "") + (row[i + 1] if i + 1 < len(row) else "")


def perturb_word(w, rng):
    if len(w) < 2:
        return w
    op = rng.choice(["swap", "sub", "del", "ins"])
    i = rng.randrange(1, len(w)) if op == "swap" else rng.randrange(len(w))
    if op == "swap":
        return w[:i - 1] + w[i] + w[i - 1] + w[i + 1:]
    if op == "del":
        return w[:i] + w[i + 1:]
    nb = NEIGH.get(w[i].lower(), "")
    c = rng.choice(nb) if nb else w[i]
    if w[i].isupper():
        c = c.upper()
    return w[:i] + c + w[i + 1:] if op == "sub" else w[:i] + c + w[i:]


def attack(text, true_idx, model, budget, rng, target_idx=0):
    words = list(_W.finditer(text))
    if not words:
        return text, {"n_words_changed": 0, "frac_words": 0.0}
    base = model([text])[0, true_idx]
    loo = [text[:m.start()] + text[m.end():] for m in words]
    scores = base - model(loo)[:, true_idx]
    order = np.argsort(-scores)
    k = max(1, int(np.ceil(budget * len(words))))
    cur, changed = text, 0
    spans = [(m.start(), m.end()) for m in words]
    for j in order[:k]:
        s, e = spans[j]
        new = perturb_word(cur[s:e], rng)
        delta = len(new) - (e - s)
        cur = cur[:s] + new + cur[e:]
        spans = [(a + delta if a > s else a, b + delta if b > s else b) if idx != j else (s, s + len(new))
                 for idx, (a, b) in enumerate(spans)]
        changed += 1
        if model([cur])[0].argmax() == target_idx:
            break
    return cur, {"n_words_changed": changed, "frac_words": changed / len(words)}
