"""A7: paraphrase / style-change attack with a local LLM; model-agnostic, generated once for all test misinfo/forged.

Styles: "rewrite as an official МЧС message" and "rewrite neutrally as a news item". Same language as the source.
Filter: LaBSE cosine similarity to the original >= 0.80. chrF to the original is stored.
  LLM_URL=http://127.0.0.1:18080 python -m src.attacks.a7_paraphrase --llm qwen3.5-35b-a3b
Output: results/E4/A7_paraphrases.parquet
"""
import argparse
import concurrent.futures as cf
import os

import numpy as np
import pandas as pd

from src.data.langid import detect_lang
from src.data.synth_llm import chat, clean

STYLES = {
    "official": "Перепиши это сообщение на {lang} языке так, чтобы оно выглядело как официальное сообщение МЧС Казахстана: "
                "официальный тон, структура пресс-релиза. Сохрани все утверждения, числа, ссылки и призывы без изменений по смыслу. "
                "Только текст.\n\nСообщение:\n\"\"\"{text}\"\"\"",
    "neutral": "Перепиши это сообщение на {lang} языке нейтрально, как короткую новость информационного агентства. "
               "Сохрани все утверждения, числа, ссылки и призывы без изменений по смыслу. Только текст.\n\nСообщение:\n\"\"\"{text}\"\"\"",
}
LANG_NAME = {"kk": "казахском", "ru": "русском", "mixed": "том же"}


def labse_sim(a, b, path="models/LaBSE"):
    from transformers import AutoModel, AutoTokenizer
    import torch
    tok, m = AutoTokenizer.from_pretrained(path), AutoModel.from_pretrained(path).eval()

    def emb(texts):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=256, return_tensors="pt")
                e = m(**enc).pooler_output
                out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
        return np.concatenate(out)
    return (emb(list(a)) * emb(list(b))).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.80)
    ap.add_argument("--max-n", type=int, default=0, help="stratified subsample of test misinfo/forged (time budget)")
    a = ap.parse_args()
    te = pd.read_parquet("data/splits/test.parquet")
    te = te[te.label != "credible"]
    if a.max_n and len(te) > a.max_n:
        te = pd.concat([g.sample(min(len(g), a.max_n // 2), random_state=0) for _, g in te.groupby("label")])
    jobs = [(r.id, r.text, r.lang, s) for r in te.itertuples() for s in STYLES]

    def run(j):
        rid, text, lang, style = j
        try:
            return rid, style, clean(chat(STYLES[style].format(lang=LANG_NAME.get(lang, "том же"), text=text[:1500]),
                                          max_tokens=600, temperature=0.7))
        except Exception as e:
            return rid, style, None
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, jobs))
    df = pd.DataFrame(res, columns=["id", "style", "text_adv"]).dropna()
    src = te.set_index("id")
    df["text_orig"] = df.id.map(src.text)
    df["label"] = df.id.map(src.label)
    df["lang"] = df.id.map(src.lang)
    df["is_synthetic"] = df.id.map(src.is_synthetic)
    df["lang_adv"] = df.text_adv.map(detect_lang)
    df["labse"] = labse_sim(df.text_orig, df.text_adv)
    from sacrebleu.metrics import CHRF
    chrf = CHRF()
    df["chrf"] = [chrf.sentence_score(h, [r]).score for h, r in zip(df.text_adv, df.text_orig)]
    df["keep"] = (df.labse >= a.thr) & (df.lang_adv == df.lang)
    os.makedirs("results/E4", exist_ok=True)
    df.to_parquet("results/E4/A7_paraphrases.parquet", index=False)
    print(df.groupby("style")[["labse", "chrf", "keep"]].mean())


if __name__ == "__main__":
    main()
