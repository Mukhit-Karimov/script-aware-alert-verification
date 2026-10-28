"""E12: independent check on REAL forged official messages (impersonation of state bodies), never used in training.

Steps (resumable):
  screen  LLM pre-screen of all verbatim rumours from debunk articles (data/processed/misinfo_judged.jsonl)
          + manually collected items (data/raw/e12_web.jsonl) against the inclusion criteria below
          -> data/processed/e12_screen.jsonl
  build   keep screened positives, attach corpus split, drop items seen in train/val or with a near-duplicate
          (word 3-shingle Jaccard >= 0.5) in train/val -> data/splits/test_real_forged.parquet
          and results/E12/candidates_for_author_check.csv (for the authors' confirmation)
  eval    run trained models on the set -> results/E12/<model>/<seed>/predictions.parquet, results/E12/summary.json

Inclusion criteria (all must hold):
  C1 the text presents itself as issued by, or speaks in the voice of, a state body or official service
     (Ministry of Emergency Situations / its departments, akimat, police / Ministry of Internal Affairs, a ministry,
     Kazhydromet, a mass-alert service), or reproduces the format of an official notice;
  C2 it is a warning, instruction or notice about safety, an emergency or a public-order measure;
  C3 an official body or a fact-checker publicly refuted it (the source article is a debunk).

  python -m src.eval.real_forged screen --workers 8
  python -m src.eval.real_forged build
  python -m src.eval.real_forged eval
"""
import argparse
import concurrent.futures as cf
import json
import os
import re

import numpy as np
import pandas as pd

SCREEN = "data/processed/e12_screen.jsonl"
WEB = "data/raw/e12_web.jsonl"
OUT_SPLIT = "data/splits/test_real_forged.parquet"
RES = "results/E12"
LABELS = ["credible", "misinfo", "forged_alert"]
OFFICIAL_RE = re.compile(r"МЧС|ДЧС|ТЖМ|ТЖД|төтенше|чрезвычайн|акимат|әкімдік|әкімі|аким|полиц|МВД|ІІМ|министерств|министрлі|"
                         r"ведомств|гидромет|оповещ|хабарланд|внимани|назар аудар|официальн|ресми|правительств|үкімет|"
                         r"комитет|департамент|штаб|санэпид|Минздрав|ДСМ|Mass ?Alert|112|101|102", re.I)

PROMPT = """Ниже текст ложного сообщения, которое распространялось в мессенджерах или соцсетях и было опровергнуто.
Сообщение: \"\"\"{text}\"\"\"

Ответь на два вопроса только по самому тексту сообщения:
C1: Выдаёт ли сообщение себя за сообщение государственного органа или официальной службы (МЧС/ДЧС/ТЖМ, акимат/әкімдік, полиция/МВД/ІІМ, министерство, Казгидромет, система оповещения) — то есть написано от их имени, подписано ими, ссылается на них как на автора («МЧС сообщает», «по поручению акимата») или копирует формат официального уведомления? Простой пересказ слухов («говорят, что…», «знакомый из МЧС сказал») — это НЕ официальное сообщение.
C2: Является ли оно предупреждением, инструкцией или уведомлением о безопасности, чрезвычайной ситуации или мерах для населения (эвакуация, землетрясение, погода, отключения, карантин, комендантский час, угрозы)?
Ответь строго JSON: {{"c1": true|false, "c2": true|false, "official_body": "<орган или null>", "reason": "<одно предложение>"}}"""


def ws(t):
    return re.sub(r"\s+", " ", t or "").strip()


def load_rumours():
    items = []
    for l in open("data/processed/misinfo_judged.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d.get("rumour"):
            items.append({"text": ws(d["rumour"]), "source": d["source"], "url": d["url"], "date": d.get("date"),
                          "origin": "debunk_corpus"})
    if os.path.exists(WEB):
        for l in open(WEB, encoding="utf-8"):
            d = json.loads(l)
            items.append({"text": ws(d["text"]), "source": d["source"], "url": d["url"], "date": d.get("date"),
                          "origin": "web_manual"})
    seen, out = set(), []
    for it in items:
        # C1 needs a named official body or an official-notice marker; the LLM screens only such texts
        if it["origin"] == "debunk_corpus" and not OFFICIAL_RE.search(it["text"]):
            continue
        if it["text"] not in seen:
            seen.add(it["text"]); out.append(it)
    return out


def screen(workers):
    from src.data.synth_llm import chat
    done = set()
    if os.path.exists(SCREEN):
        done = {json.loads(l)["text"] for l in open(SCREEN, encoding="utf-8")}
    todo = [it for it in load_rumours() if it["text"] not in done]
    print("to screen", len(todo), flush=True)

    def one(it):
        try:
            out = chat(PROMPT.format(text=it["text"][:2500]), max_tokens=200, temperature=0.0)
            j = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
        except Exception as e:  # unparsable -> negative, kept for the record
            j = {"c1": False, "c2": False, "official_body": None, "reason": f"error: {e}"}
        return {**it, **j}

    with open(SCREEN, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, todo)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if i % 100 == 0:
                print(i, flush=True)


def shingles(t, n=3):
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def build():
    from src.data.langid import detect_lang
    s = pd.DataFrame([json.loads(l) for l in open(SCREEN, encoding="utf-8")])
    pos = s[(s.c1 == True) & (s.c2 == True)].copy()  # noqa: E712
    corpus = pd.read_parquet("data/processed/corpus.parquet")
    split_of = {}
    for sp in ["train", "val", "test", "test_ood_event"]:
        for t in pd.read_parquet(f"data/splits/{sp}.parquet").text:
            split_of.setdefault(ws(t), sp)
    pos["corpus_split"] = pos.text.map(lambda t: split_of.get(t, "not_in_corpus"))
    pos["corpus_label"] = pos.text.map(dict(zip(corpus.text.map(ws), corpus.label))).fillna("none")
    trval = pd.concat([pd.read_parquet("data/splits/train.parquet"), pd.read_parquet("data/splits/val.parquet")])
    tv_sh = [shingles(t) for t in trval.text]
    pos["max_jaccard_trainval"] = [max((jaccard(shingles(t), x) for x in tv_sh), default=0.0) for t in pos.text]
    keep = pos[pos.corpus_split.isin(["test", "test_ood_event", "not_in_corpus"]) & (pos.max_jaccard_trainval < 0.5)].copy()
    keep = keep[keep.text.str.len() >= 40]
    keep["id"] = [f"rf{i:04d}" for i in range(len(keep))]
    keep["label"] = "forged_alert"
    keep["lang"] = keep.text.map(detect_lang)
    keep["is_synthetic"] = False
    os.makedirs(RES, exist_ok=True)
    cols = ["id", "text", "label", "lang", "is_synthetic", "official_body", "source", "url", "date", "origin",
            "corpus_split", "corpus_label", "max_jaccard_trainval", "reason"]
    keep[cols].to_parquet(OUT_SPLIT, index=False)
    chk = keep[["id", "text", "official_body", "url", "reason"]].copy()
    chk["author_c1_official_voice"] = ""
    chk["author_c2_alert_or_notice"] = ""
    chk["author_c3_debunk_confirmed"] = ""
    chk.to_csv(f"{RES}/candidates_for_author_check.csv", index=False, encoding="utf-8-sig")
    stats = {"screened": int(len(s)), "screen_positive": int(len(pos)),
             "positive_by_split": pos.corpus_split.value_counts().to_dict(),
             "dropped_seen_in_train_val": int((pos.corpus_split.isin(["train", "val"])).sum()),
             "dropped_near_dup": int((pos.corpus_split.isin(["test", "test_ood_event", "not_in_corpus"]) & (pos.max_jaccard_trainval >= 0.5)).sum()),
             "n": int(len(keep)), "by_lang": keep.lang.value_counts().to_dict(),
             "by_origin": keep.origin.value_counts().to_dict(),
             "corpus_label_of_kept": keep.corpus_label.value_counts().to_dict()}
    json.dump(stats, open(f"{RES}/set_stats.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def boot_ci(x, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def evaluate(models):
    from src.attacks.run_attacks import Counted, load_model
    df = pd.read_parquet(OUT_SPLIT)
    if os.path.exists(f"{RES}/excluded_ids.txt"):  # items the authors rejected at confirmation
        bad = set(open(f"{RES}/excluded_ids.txt").read().split())
        df = df[~df.id.isin(bad)]
    summary = {"n": int(len(df)), "by_lang": df.lang.value_counts().to_dict()}
    for name, exp, base, defense, seeds in models:
        per = []
        for seed in seeds:
            if base.startswith("M9"):
                if seed != seeds[0]:
                    continue
            elif not os.path.exists(f"results/{exp}/{base}/{seed}"):
                continue
            P = Counted(load_model(exp, base, seed), defense)(df.text.tolist())
            pred = np.array(LABELS)[P.argmax(1)]
            out = df[["id", "lang"]].copy()
            out["y_pred"] = pred
            out["p_credible"], out["p_misinfo"], out["p_forged"] = P[:, 0], P[:, 1], P[:, 2]
            os.makedirs(f"{RES}/{name}/{seed}", exist_ok=True)
            out.to_parquet(f"{RES}/{name}/{seed}/predictions.parquet", index=False)
            det = (pred != "credible").astype(float)
            r = {"seed": seed, "detection_rate": float(det.mean()), "detection_ci": boot_ci(det),
                 "forged_rate": float((pred == "forged_alert").mean()), "misinfo_rate": float((pred == "misinfo").mean()),
                 "credible_rate": float((pred == "credible").mean())}
            for lg in sorted(df.lang.unique()):
                mk = (df.lang == lg).values
                r[f"detection_rate.{lg}"] = float(det[mk].mean())
            per.append(r)
            print(name, r, flush=True)
        if per:
            agg = {k: float(np.mean([p[k] for p in per])) for k in per[0] if k not in ("seed", "detection_ci")}
            agg.update({f"{k}_std": float(np.std([p[k] for p in per])) for k in ("detection_rate", "forged_rate")})
            summary[name] = {"mean": agg, "per_seed": per, "n_seeds": len(per)}
    json.dump(summary, open(f"{RES}/summary.json", "w"), indent=1, ensure_ascii=False)


MODELS = [("M1", "E1", "M1", "none", [13, 42, 2026]),
          ("M6", "E1", "M6", "none", [13, 42, 2026]),
          ("M8", "E1", "M8", "none", [13, 42, 2026]),
          ("M6_D4", "E5", "M6_D3", "aware", [13, 42, 2026]),
          ("M9", "E1", "M9_qwen3.5-35b-a3b", "none", [13])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["screen", "build", "eval"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.step == "screen":
        screen(a.workers)
    elif a.step == "build":
        build()
    else:
        evaluate(MODELS)
