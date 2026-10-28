"""Shared character tables: homoglyphs (Appendix B), invisibles, Kazakh letters, Latin-2021 (Appendix C)."""

# Appendix B: Cyrillic -> visually identical Latin.
CYR2LAT = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "һ": "h",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I",
}
LAT2CYR = {v: k for k, v in CYR2LAT.items()}
HOMOGLYPHS = set(CYR2LAT) | set(LAT2CYR)

# Invisible / format characters used by attack A2.
INVISIBLE_ATTACK = ["​", "‌", "‍", "⁠", "﻿", "­"]
# Additional invisibles removed by the normalizer (not all are category Cf).
INVISIBLE_EXTRA = {"͏", "ᅟ", "ᅠ", "ㅤ", "ﾠ", "᠎"} | {chr(c) for c in range(0xFE00, 0xFE10)}

KK_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
# Attack A4: Kazakh-specific letters -> nearest Russian letters.
KK2RU = {
    "ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о", "ұ": "у", "ү": "у", "һ": "х", "і": "и",
    "Ә": "А", "Ғ": "Г", "Қ": "К", "Ң": "Н", "Ө": "О", "Ұ": "У", "Ү": "У", "Һ": "Х", "І": "И",
}

# Kazakh Latin letters with diacritics (2021) — a token containing them is legitimate Latin.
KK_LATIN_SPECIFIC = set("äğqñöūüışİÄĞQÑÖŪÜŞ")

# Appendix C, checked against the April-2021 revision (31 letters; ң -> ñ, no Ç).
# Letters absent from the official alphabet (ц ч щ ю я ё э ъ ь) use common digraphs: documented deviation.
KK_LATIN_2021 = {
    "а": "a", "ә": "ä", "б": "b", "в": "v", "г": "g", "ғ": "ğ", "д": "d", "е": "e", "ж": "j",
    "з": "z", "и": "i", "й": "i", "к": "k", "қ": "q", "л": "l", "м": "m", "н": "n", "ң": "ñ",
    "о": "o", "ө": "ö", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "ū", "ү": "ü",
    "ф": "f", "х": "h", "һ": "h", "ш": "ş", "ы": "y", "і": "ı",
    # non-official (deviation, see results/E4/translit_map.json)
    "ц": "ts", "ч": "ch", "щ": "şş", "ю": "iu", "я": "ia", "ё": "io", "э": "e", "ъ": "", "ь": "",
}
KK_LATIN_2021_DEVIATIONS = ["ц", "ч", "щ", "ю", "я", "ё", "э", "ъ", "ь"]
