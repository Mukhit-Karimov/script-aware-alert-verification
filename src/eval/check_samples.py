"""Automatic meaning/readability check of attack samples (results/E4/samples.csv) — replaces manual review.

A1–A4 (character-level): meaning_kept = 1 by construction.
  readable_strict = 1 if D2-style cleaning (invisible chars removed, homoglyphs mapped to one alphabet) restores the original
                    exactly (by design it can only hold for A1/A2);
  readable_folded = 1 if the texts coincide after alphabet folding (invisibles removed, Kazakh letters folded to their
                    Russian look-alikes, Cyrillic transliterated to Latin 2021), i.e. they differ only in alphabet.
A7 (paraphrase): three independent checks
  - LLM judge (local Qwen3.5-35B-A3B — KAZ-LLM unavailable), temperature 0, prompt configs/judge_prompt.txt;
  - LaBSE cosine to the original (from results/E4/A7_paraphrases.parquet);
  - NLI bidirectional entailment (MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7; no Kazakh in its training
    languages — Kazakh pairs are reported separately).
  meaning_kept(A7) = judge_meaning AND LaBSE >= 0.80.
  python -m src.eval.check_samples prep      # A1–A4 auto columns, human_check.csv (20 A7 pairs, seed 42), judge prompt
  LLM_URL=... python -m src.eval.check_samples judge
  python -m src.eval.check_samples kappa                  # blind human annotation in results/E4/human_check.csv
"""
import concurrent.futures as cf
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import requests

from src.charmaps import INVISIBLE_ATTACK, INVISIBLE_EXTRA, KK2RU, KK_LATIN_2021
from src.normalize import normalize

URL = os.environ.get("LLM_URL", "http://127.0.0.1:18080")
PROMPT_FILE = "configs/judge_prompt.txt"
PROMPT = """Ты — эксперт по казахскому и русскому языкам. Сравни исходное сообщение и его переписанный вариант.
Сен — қазақ және орыс тілдерінің сарапшысысың.

Вопрос 1 (meaning): сохранён ли смысл исходного сообщения — те же утверждения, факты, числа, призывы к действию? Смена стиля или тона не считается потерей смысла. 1 — да, 0 — нет.
Вопрос 2 (natural): читается ли переписанный вариант естественно, как текст, написанный человеком на этом языке? 1 — да, 0 — нет.

Исходное сообщение:
\"\"\"{orig}\"\"\"

Переписанный вариант:
\"\"\"{adv}\"\"\"

Ответь строго в формате JSON без пояснений: {{"meaning": 0 или 1, "natural": 0 или 1}}"""
INV = set(INVISIBLE_ATTACK) | INVISIBLE_EXTRA
LAT_FOLD = {"ä": "a", "ğ": "g", "q": "k", "ñ": "n", "ö": "o", "ū": "u", "ü": "u", "ı": "i", "ş": "sh"}
CYR_FOLD = {k: v for k, v in KK2RU.items()}


def strict_clean(t):
    return normalize(t)[0]


def fold(t):
    t = normalize(t)[0].lower()  # invisibles out, homoglyphs back to each token's dominant alphabet (D2)
    t = "".join(CYR_FOLD.get(c, c) for c in t)  # ә->а ...
    t = "".join(KK_LATIN_2021.get(c, c) for c in t)  # Cyrillic -> Latin 2021
    t = "".join(LAT_FOLD.get(c, c) for c in t)
    return re.sub(r"\s+", " ", t).strip()


def prep():
    s = pd.read_csv("results/E4/samples.csv")
    char = s.attack.isin(["A1", "A2", "A3", "A4"])
    s["meaning_kept"] = np.where(char, 1, np.nan)
    s["readable_strict"] = np.where(char, [int(strict_clean(a) == strict_clean(o)) for o, a in zip(s.original, s.adversarial)], np.nan)
    s["readable_folded"] = np.where(char, [int(fold(a) == fold(o)) for o, a in zip(s.original, s.adversarial)], np.nan)
    s["readable"] = s["readable_strict"]
    s.to_csv("results/E4/samples.csv", index=False)
    os.makedirs("configs", exist_ok=True)
    open(PROMPT_FILE, "w", encoding="utf-8").write(PROMPT + "\n")
    rows = []
    for f in glob.glob("results/E4/M6/42/predictions_A7_*.parquet"):
        p = pd.read_parquet(f)
        rows.append(p[["id", "attack", "text_adv"]])
    p = pd.concat(rows)
    te = pd.read_parquet("data/splits/test.parquet").set_index("id")
    hc = p.sample(20, random_state=42)
    hc = pd.DataFrame({"id": hc.id.values, "attack": hc.attack.values, "lang": te.loc[hc.id, "lang"].values,
                       "original": te.loc[hc.id, "text"].values, "attacked": hc.text_adv.values,
                       "my_meaning_kept": "", "my_readable": ""})
    if not os.path.exists("results/E4/human_check.csv"):  # never overwrite the author's answers
        hc.to_csv("results/E4/human_check.csv", index=False)
    print("A1–A4:", s[char].groupby("attack")[["readable_strict", "readable_folded"]].mean().to_dict())


def ask(orig, adv):
    r = requests.post(f"{URL}/v1/chat/completions", timeout=1800, json={
        "messages": [{"role": "user", "content": PROMPT.format(orig=orig[:1500], adv=adv[:1500])}], "max_tokens": 30,
        "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status()
    ans = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*?\}", ans, re.S)
    try:
        d = json.loads(m.group())
        return int(d["meaning"]), int(d["natural"]), ans
    except Exception:  # noqa: BLE001
        return np.nan, np.nan, ans


def nli_bidir(a, b, path="models/nli_mdeberta_xnli"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(int(os.environ.get("NTHREADS", 28)))
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).eval()
    ent = [i for i, l in m.config.id2label.items() if l.lower().startswith("entail")][0]

    def p(x, y):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 16):
                enc = tok(list(x[i:i + 16]), list(y[i:i + 16]), truncation=True, max_length=256, padding=True, return_tensors="pt")
                out.append(torch.softmax(m(**enc).logits, -1)[:, ent].numpy())
        return np.concatenate(out)
    return p(a, b), p(b, a)


def judge():
    par = pd.read_parquet("results/E4/A7_paraphrases.parquet")
    par = par[par.keep].reset_index(drop=True)
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda r: ask(r[0], r[1]), zip(par.text_orig, par.text_adv)))
    par["judge_meaning"] = [r[0] for r in res]
    par["judge_natural"] = [r[1] for r in res]
    par["judge_raw"] = [r[2] for r in res]
    f, b = nli_bidir(par.text_orig.tolist(), par.text_adv.tolist())
    par["nli_entail_fwd"], par["nli_entail_bwd"] = f, b
    par["nli_equivalent"] = ((f > 0.5) & (b > 0.5)).astype(int)
    par["meaning_kept"] = ((par.judge_meaning == 1) & (par.labse >= 0.80)).astype(int)
    par.drop(columns=["text_orig", "text_adv"]).to_csv("results/E4/A7_judged.csv", index=False)  # no corpus texts
    finish()


def judged():
    """A7_judged.csv with the paraphrase texts re-attached (same row order as the kept paraphrases)."""
    par = pd.read_parquet("results/E4/A7_paraphrases.parquet")
    j = pd.read_csv("results/E4/A7_judged.csv")
    j["text_adv"] = par[par.keep].reset_index(drop=True).text_adv
    return j


def finish():
    """Write the A7 rows of samples.csv and the summary from the saved judge output (no LLM calls)."""
    par = judged()
    s = pd.read_csv("results/E4/samples.csv")
    key = par.drop_duplicates("text_adv").set_index("text_adv")  # identical paraphrases of different styles
    a7 = s.attack.str.startswith("A7")
    for col, src in [("judge_meaning", "judge_meaning"), ("judge_natural", "judge_natural"), ("labse", "labse"),
                     ("nli_equivalent", "nli_equivalent"), ("meaning_kept", "meaning_kept")]:
        s.loc[a7, col] = s.loc[a7, "adversarial"].map(key[src])
    s.loc[a7, "readable"] = s.loc[a7, "judge_natural"]
    s.to_csv("results/E4/samples.csv", index=False)
    summ = summary(s, par)
    print(json.dumps(summ, indent=1, ensure_ascii=False))


def summary(s, par):
    char = s.attack.isin(["A1", "A2", "A3", "A4"])
    out = {"A1_A4": {"n": int(char.sum()), "meaning_kept_share": 1.0,
                     "readable_strict_share": float(s.loc[char, "readable_strict"].mean()),
                     "readable_folded_share": float(s.loc[char, "readable_folded"].mean()),
                     "readable_strict_by_attack": s[char].groupby("attack").readable_strict.mean().to_dict(),
                     "meaningful_share_strict": float(s.loc[char, "readable_strict"].mean()),
                     "meaningful_share_folded": float(s.loc[char, "readable_folded"].mean())}}
    a7 = s.attack.str.startswith("A7")
    if "meaning_kept" in s and a7.any():
        out["A7_samples"] = {"n": int(a7.sum()), "meaning_kept_share": float(s.loc[a7, "meaning_kept"].mean()),
                             "readable_share": float(s.loc[a7, "judge_natural"].mean()),
                             "meaningful_share": float(((s.loc[a7, "meaning_kept"] == 1) & (s.loc[a7, "judge_natural"] == 1)).mean())}
    if par is not None:
        g = par.groupby("style")
        out["A7_all_kept_paraphrases"] = {
            "n": len(par), "judge_invalid": int(par.judge_meaning.isna().sum()),
            "judge_meaning_share": float(par.judge_meaning.mean()), "judge_natural_share": float(par.judge_natural.mean()),
            "labse_ge_080_share": float((par.labse >= 0.80).mean()),
            "nli_equivalent_share": float(par.nli_equivalent.mean()),
            "nli_equivalent_share_ru": float(par[par.lang == "ru"].nli_equivalent.mean()),
            "nli_equivalent_share_kk": float(par[par.lang == "kk"].nli_equivalent.mean()),
            "meaning_kept_share": float(par.meaning_kept.mean()),
            "meaningful_share": float(((par.meaning_kept == 1) & (par.judge_natural == 1)).mean()),
            "meaning_kept_by_style": g.meaning_kept.mean().to_dict(),
            "judge_vs_nli_agreement": float((par.judge_meaning == par.nli_equivalent).mean())}
        out["A7_caveat"] = bool(out["A7_all_kept_paraphrases"]["meaningful_share"] < 0.70)
    json.dump(out, open("results/E4/samples_auto_check.json", "w"), indent=1, ensure_ascii=False)
    return out


def kappa(annotator="human"):
    """Agreement of the blind human annotation (results/E4/human_check.csv) with the LLM judge."""
    from sklearn.metrics import cohen_kappa_score
    h = pd.read_csv("results/E4/human_check.csv", encoding="utf-8-sig")
    h = h[h.my_meaning_kept.notna() & h.my_readable.notna()]
    j = judged().drop_duplicates("text_adv").set_index("text_adv")
    h["judge_meaning"] = h.attacked.map(j.judge_meaning)
    h["judge_natural"] = h.attacked.map(j.judge_natural)
    h = h.dropna(subset=["judge_meaning", "judge_natural"])
    out = {"annotator": annotator, "n": len(h),
           "kappa_meaning": float(cohen_kappa_score(h.my_meaning_kept.astype(int), h.judge_meaning.astype(int))),
           "kappa_readable": float(cohen_kappa_score(h.my_readable.astype(int), h.judge_natural.astype(int))),
           "agreement_meaning": float((h.my_meaning_kept.astype(int) == h.judge_meaning.astype(int)).mean()),
           "agreement_readable": float((h.my_readable.astype(int) == h.judge_natural.astype(int)).mean()),
           "human_meaning_share": float(h.my_meaning_kept.astype(int).mean()),
           "human_readable_share": float(h.my_readable.astype(int).mean()),
           "judge_meaning_share": float(h.judge_meaning.astype(int).mean()),
           "judge_natural_share": float(h.judge_natural.astype(int).mean())}
    # percentile bootstrap 95% CI over pairs (1000 resamples); resamples with a constant column give nan and are skipped
    rng = np.random.default_rng(2026)
    for col_h, col_j, name in (("my_meaning_kept", "judge_meaning", "kappa_meaning"), ("my_readable", "judge_natural", "kappa_readable")):
        x, y, ks = h[col_h].astype(int).values, h[col_j].astype(int).values, []
        for _ in range(1000):
            i = rng.integers(0, len(x), len(x))
            with np.errstate(all="ignore"):
                k = cohen_kappa_score(x[i], y[i])
            if k == k:
                ks.append(k)
        out[f"{name}_ci_lo"], out[f"{name}_ci_hi"] = (float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5)))
        out[f"{name}_n_boot_valid"] = len(ks)
    json.dump(out, open(f"results/E4/{annotator}_vs_judge_kappa.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    {"prep": prep, "judge": judge, "finish": finish, "kappa": kappa}[sys.argv[1]](*sys.argv[2:])
