"""Model-agnostic character-level perturbations A1–A4 (random, budgeted).

Each function returns (attacked_text, info) where info has n_chars_changed / n_words_changed.
Budget semantics (also the X axis of F5):
  A1: fraction of all characters of the text replaced by a cross-script homoglyph;
  A2: number of invisible characters inserted strictly inside words;
  A3: fraction of Cyrillic words fully transliterated to Kazakh Latin-2021;
  A4: fraction of words *containing Kazakh-specific letters* where those letters are replaced by Russian ones.
"""
import math
import random
import re

from src.charmaps import CYR2LAT, INVISIBLE_ATTACK, KK2RU, KK_LATIN_2021, LAT2CYR

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_CYR_WORD_RE = re.compile(r"[Ѐ-ӿ]+")


def _n(budget, total):
    return min(total, max(1, math.ceil(budget * total))) if total else 0


def a1_homoglyph(text, budget, rng):
    """Cyrillic->Latin (or Latin->Cyrillic) homoglyph substitution."""
    pos = [i for i, ch in enumerate(text) if ch in CYR2LAT or ch in LAT2CYR]
    k = min(len(pos), _n(budget, len(text)))
    chars = list(text)
    for i in rng.sample(pos, k):
        chars[i] = CYR2LAT.get(chars[i]) or LAT2CYR[chars[i]]
    return "".join(chars), {"n_chars_changed": k, "frac_chars": k / max(1, len(text))}


def a2_invisible(text, n_insert, rng):
    """Insert zero-width / format characters between two letters of a word."""
    slots = [m.start() + j for m in _WORD_RE.finditer(text) for j in range(1, len(m.group()))]
    if not slots:
        return text, {"n_chars_changed": 0, "frac_chars": 0.0}
    picks = sorted((rng.choice(slots) for _ in range(int(n_insert))), reverse=True)
    for p in picks:
        text = text[:p] + rng.choice(INVISIBLE_ATTACK) + text[p:]
    return text, {"n_chars_changed": len(picks), "frac_chars": len(picks) / max(1, len(text))}


def _translit(word):
    out = []
    for ch in word:
        low = ch.lower()
        t = KK_LATIN_2021.get(low, ch)
        out.append(t.capitalize() if ch != low and t else t)
    return "".join(out)


def a3_translit(text, budget, rng):
    """Partial transliteration of Cyrillic words to Kazakh Latin (2021)."""
    words = list(_CYR_WORD_RE.finditer(text))
    k = _n(budget, len(words))
    chosen = sorted(rng.sample(range(len(words)), k), reverse=True)
    n_chars = 0
    for idx in chosen:
        m = words[idx]
        new = _translit(m.group())
        n_chars += len(m.group())
        text = text[:m.start()] + new + text[m.end():]
    return text, {"n_words_changed": k, "frac_words": k / max(1, len(words)), "n_chars_changed": n_chars}


def a4_kk2ru(text, budget, rng):
    """Replace Kazakh-specific letters with the nearest Russian letters in a fraction of eligible words."""
    words = [m for m in _WORD_RE.finditer(text) if any(c in KK2RU for c in m.group())]
    k = _n(budget, len(words))
    chosen = sorted(rng.sample(range(len(words)), k), reverse=True)
    n_chars = 0
    for idx in chosen:
        m = words[idx]
        new = "".join(KK2RU.get(c, c) for c in m.group())
        n_chars += sum(a != b for a, b in zip(m.group(), new))
        text = text[:m.start()] + new + text[m.end():]
    n_all = len(_WORD_RE.findall(text))
    return text, {"n_words_changed": k, "frac_words": k / max(1, n_all), "n_chars_changed": n_chars,
                  "n_eligible_words": len(words)}


ATTACKS = {"A1": a1_homoglyph, "A2": a2_invisible, "A3": a3_translit, "A4": a4_kk2ru}
BUDGETS = {"A1": [0.01, 0.02, 0.05, 0.10, 0.20], "A2": [1, 3, 5, 10, 20],
           "A3": [0.10, 0.25, 0.50, 1.00], "A4": [0.10, 0.25, 0.50, 1.00]}


def apply(attack, text, budget, seed):
    return ATTACKS[attack](text, budget, random.Random(seed))
