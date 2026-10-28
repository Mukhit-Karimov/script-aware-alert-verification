"""Layer 1: language-aware normalization (D2) and the naive baseline (D1).

normalize(text) -> (normalized_text, report). The report is used as features by Layer 3.
"""
import re
import unicodedata

from src.charmaps import (CYR2LAT, HOMOGLYPHS, INVISIBLE_EXTRA, KK2RU, KK_LATIN_SPECIFIC,
                          KK_LETTERS, LAT2CYR)

_TOKEN_RE = re.compile(r"\S+")
_PROTECTED_RE = re.compile(r"(https?://|www\.|@|\.(kz|ru|com|org|net|gov)\b|\.[a-z]{2,4}/)", re.I)

# Frequent Kazakh words that contain Kazakh-specific letters. Used only to *detect* A4-style
# substitution (ә->а, қ->к, ...); the text is never corrected with it.
KK_SEED_LEXICON = [
    "және", "үшін", "бұл", "туралы", "бойынша", "қазір", "кейін", "қала", "қаласы", "ақпарат",
    "жағдай", "жағдайлар", "төтенше", "өрт", "қызмет", "қызметі", "тұрғындар", "тұрғындары",
    "үй", "үйлер", "көшесі", "ауданы", "облысы", "сілкінісі", "сілкініс", "құтқару", "әкімдік",
    "әкімдігі", "қауіп", "қауіпті", "қауіпсіздік", "ескерту", "хабарлама", "хабарламасы",
    "барлық", "бірақ", "біз", "сіз", "сізге", "өтінеміз", "болып", "бүгін", "ертең", "түнде",
    "таңертең", "сағат", "министрлігі", "дейін", "қарай", "әлі", "көп", "күн", "арқылы", "тиіс",
    "керек", "өз", "үшін", "жөнінде", "мұнда", "сондай", "қатты", "күшті", "нөсер", "тасқын",
    "көктемгі", "ғимарат", "ғимараттан", "шығу", "шығыңыз", "қалдырыңыз", "сақ", "болыңыз",
    "жаңа", "жаңалық", "ресми", "дереккөз", "мәлімет", "мәліметі", "жалған", "сенбеңіз",
]


def _is_cyr(ch):
    return "Ѐ" <= ch <= "ӿ" or "Ԁ" <= ch <= "ԯ"


def _is_lat(ch):
    return ch.isalpha() and ch.isascii() or "À" <= ch <= "ɏ"


def _script_counts(s):
    """Counts of unambiguous / homoglyph letters per script."""
    c = {"cyr_u": 0, "lat_u": 0, "cyr_h": 0, "lat_h": 0}
    for ch in s:
        if _is_cyr(ch):
            c["cyr_h" if ch in HOMOGLYPHS else "cyr_u"] += 1
        elif _is_lat(ch):
            c["lat_h" if ch in HOMOGLYPHS else "lat_u"] += 1
    return c


def remove_invisible(text):
    out, n = [], 0
    for ch in text:
        if unicodedata.category(ch) == "Cf" or ch in INVISIBLE_EXTRA:
            n += 1
        else:
            out.append(ch)
    return "".join(out), n


def _nfkc_safe(text):
    """NFKC applied char-by-char so that Kazakh letters can never be decomposed/merged away."""
    text = unicodedata.normalize("NFC", text)
    return "".join(ch if ch in KK_LETTERS else unicodedata.normalize("NFKC", ch) for ch in text)


def _to_script(token, script):
    table = LAT2CYR if script == "cyr" else CYR2LAT
    out, n = [], 0
    for ch in token:
        if ch in table and (_is_lat(ch) if script == "cyr" else _is_cyr(ch)):
            out.append(table[ch]); n += 1
        else:
            out.append(ch)
    return "".join(out), n


def kk_features(text, lexicon=KK_SEED_LEXICON):
    """kk_letter_ratio and kk_letter_substitution_rate (share of lexicon hits seen in A4-substituted form)."""
    letters = [ch for ch in text if _is_cyr(ch)]
    kk_ratio = sum(ch in KK_LETTERS for ch in letters) / len(letters) if letters else 0.0
    lex = {w for w in lexicon if any(c in KK_LETTERS for c in w)}
    sub = {"".join(KK2RU.get(c, c) for c in w) for w in lex} - lex
    n_orig = n_sub = 0
    for tok in re.findall(r"\w+", text.lower()):
        if tok in lex:
            n_orig += 1
        elif tok in sub:
            n_sub += 1
    rate = n_sub / (n_orig + n_sub) if (n_orig + n_sub) else 0.0
    return kk_ratio, rate, n_orig + n_sub


def normalize(text, lexicon=KK_SEED_LEXICON, context_threshold=0.8):
    """D2: language-aware normalization. Returns (text, report)."""
    rep = {}
    text, rep["n_invisible"] = remove_invisible(text)
    nf = _nfkc_safe(text)
    rep["n_nfkc_changed"] = sum(a != b for a, b in zip(text, nf)) + abs(len(text) - len(nf))
    text = nf

    tot = _script_counts(text)
    denom = tot["cyr_u"] + tot["lat_u"]
    ctx = None
    if denom:
        if tot["cyr_u"] / denom >= context_threshold:
            ctx = "cyr"
        elif tot["lat_u"] / denom >= context_threshold:
            ctx = "lat"

    rep.update(n_homoglyph_fixed=0, n_mixed_tokens=0, n_mixed_protected=0, n_kk_latin_tokens=0)
    out, last = [], 0
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        out.append(text[last:m.start()]); last = m.end()
        c = _script_counts(tok)
        has_cyr, has_lat = c["cyr_u"] + c["cyr_h"] > 0, c["lat_u"] + c["lat_h"] > 0
        if _PROTECTED_RE.search(tok):
            rep["n_mixed_protected"] += int(has_cyr and has_lat)
            out.append(tok); continue
        if has_lat and not has_cyr and any(ch in KK_LATIN_SPECIFIC for ch in tok):
            rep["n_kk_latin_tokens"] += 1
            out.append(tok); continue
        if c["cyr_u"] and not c["lat_u"]:
            target = "cyr"
        elif c["lat_u"] and not c["cyr_u"]:
            target = "lat"
        elif c["cyr_u"] and c["lat_u"]:
            target = "cyr" if c["cyr_u"] > c["lat_u"] else "lat" if c["lat_u"] > c["cyr_u"] else ctx
        else:  # token made only of homoglyph letters: decide by context
            target = ctx
        if has_cyr and has_lat:
            rep["n_mixed_tokens"] += 1
        if target is None or (target == "cyr" and not has_lat) or (target == "lat" and not has_cyr):
            out.append(tok); continue
        new, n = _to_script(tok, target)
        rep["n_homoglyph_fixed"] += n
        out.append(new)
    out.append(text[last:])
    text = "".join(out)

    rep["kk_letter_ratio"], rep["kk_letter_substitution_rate"], rep["kk_lexicon_hits"] = kk_features(text, lexicon)
    rep["anomaly_classes"] = [k for k in ("n_invisible", "n_nfkc_changed", "n_homoglyph_fixed",
                                          "n_mixed_tokens", "n_mixed_protected") if rep[k]]
    if rep["kk_letter_substitution_rate"] > 0:
        rep["anomaly_classes"].append("kk_letter_substitution")
    return text, rep


def normalize_naive(text):
    """D1: NFKC + remove invisibles + map every homoglyph to Cyrillic (breaks legitimate Latin)."""
    text, _ = remove_invisible(text)
    text = unicodedata.normalize("NFKC", text)
    return "".join(LAT2CYR.get(ch, ch) for ch in text)
