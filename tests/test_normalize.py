import random
import unicodedata

from src.attacks.char_attacks import a1_homoglyph, a2_invisible, a3_translit, a4_kk2ru
from src.normalize import normalize, normalize_naive

KK = "Алматы қаласында жер сілкінісі болды, тұрғындар ғимараттан шығыңыз. Әкімдік хабарламасы өте маңызды, үйде һәм"
RU = "Внимание! В городе Алматы произошло землетрясение, сохраняйте спокойствие."


def test_kazakh_letters_preserved():
    letters = "әғқңөұүһіӘҒҚҢӨҰҮҺІ"
    out, rep = normalize(letters)
    assert out == letters
    assert all(unicodedata.normalize("NFKC", c) == c for c in letters)
    assert normalize(KK)[0] == KK


def test_clean_text_unchanged():
    for t in (KK, RU, "Qazaqstan Respublikasy Tötenşe jağdailar ministrligi", "Подробнее: https://www.gov.kz/memleket"):
        out, rep = normalize(t)
        assert out == t, (t, out, rep)
        assert rep["n_homoglyph_fixed"] == 0


def test_invisible_removed():
    t = "зем​лет­рясе﻿ние⁠ в‍ Ал‌маты"
    out, rep = normalize(t)
    assert out == "землетрясение в Алматы"
    assert rep["n_invisible"] == 6


def test_homoglyph_restored_ru_and_kk():
    for t in (RU, KK):
        for seed in range(5):
            adv, _ = a1_homoglyph(t, 0.1, random.Random(seed))
            assert adv != t
            out, rep = normalize(adv)
            assert out == t, (adv, out)
            assert rep["n_homoglyph_fixed"] > 0


def test_a2_restored():
    adv, _ = a2_invisible(KK, 10, random.Random(0))
    assert normalize(adv)[0] == KK


def test_kazakh_latin_kept_but_naive_breaks_it():
    t = "Qazaqstan halqyna eskertu: jer silkinisi bolady"
    assert normalize(t)[0] == t
    assert normalize_naive(t) != t


def test_a3_translit_is_latin_and_kept():
    adv, info = a3_translit("қала өрт", 1.0, random.Random(0))
    assert adv == "qala ört"
    assert normalize(adv)[0] == adv


def test_a4_flagged_not_corrected():
    adv, _ = a4_kk2ru(KK, 1.0, random.Random(0))
    out, rep = normalize(adv)
    assert out == adv  # never corrected blindly
    assert rep["kk_letter_substitution_rate"] > 0.5
    assert normalize(KK)[1]["kk_letter_substitution_rate"] == 0.0


def test_url_protected():
    t = "Подробнее: https://mchs.gov.kz/ru/news"
    assert normalize(t)[0] == t
