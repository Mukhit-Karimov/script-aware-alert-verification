"""E4: black-box evasion attacks on misinfo/forged_alert test examples that the model got right.

Threat model (paper, Section 3): attacker wants the message classified as `credible`; only label/probability access.
  python -m src.attacks.run_attacks --exp E1 --model M6 --seed 13 --attacks A1,A2,A3,A4,A6 [--defense none|naive|aware]
Writes results/E4/<model>[_<defense>]/<seed>/predictions_<attack>_<budget>.parquet + metrics json, and
results/E4/<...>/samples_<attack>.csv (before/after pairs for manual review).
"""
import argparse
import json
import os
import pickle
import random
import zlib

import numpy as np
import pandas as pd

from src.attacks.a5_wordbug import attack as a5_attack
from src.attacks.char_attacks import ATTACKS, BUDGETS, apply
from src.charmaps import CYR2LAT, INVISIBLE_ATTACK, KK2RU, LAT2CYR
from src.eval.metrics import LABELS, attack_metrics
from src.normalize import normalize, normalize_naive

SPLITS = os.environ.get("SPLITS", "data/splits")  # E11: data/splits_strip, data/splits_src

CRED = LABELS.index("credible")


class Counted:
    """Wraps predict_proba(texts) with an optional defense and counts queries."""

    def __init__(self, fn, defense="none"):
        self.fn, self.defense, self.queries = fn, defense, 0

    def __call__(self, texts):
        self.queries += len(texts)
        if self.defense == "aware":
            texts = [normalize(t)[0] for t in texts]
        elif self.defense == "naive":
            texts = [normalize_naive(t) for t in texts]
        return self.fn(list(texts))


def load_model(exp, model, seed):
    if model.startswith("M9"):  # LLM zero-shot via llama-server (LLM_URL)
        import concurrent.futures as cf
        from src.models.llm_classifier import classify

        def f(texts):
            with cf.ThreadPoolExecutor(8) as ex:
                return np.array([r[2] for r in ex.map(classify, texts)])
        return f
    d = f"results/{exp}/{model}/{seed}"
    if os.path.exists(f"{d}/model.pkl"):
        m = pickle.load(open(f"{d}/model.pkl", "rb"))
        return m.predict_proba
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.models.transformer import predict_proba
    torch.set_num_threads(int(os.environ.get("NTHREADS", 28)))
    tok = AutoTokenizer.from_pretrained(f"{d}/model")
    mdl = AutoModelForSequenceClassification.from_pretrained(f"{d}/model")
    return lambda texts: predict_proba(mdl, tok, texts)


def greedy_a6(text, model, max_edits=10, rng=None, n_cand=24):
    """A6: greedy search over A1+A2+A4 single edits; each step keeps the edit that lowers p(true class) most."""
    rng = rng or random.Random(0)
    cur = text
    for step in range(max_edits):
        cands = []
        pos = [i for i, c in enumerate(cur) if c in CYR2LAT or c in LAT2CYR]
        kk = [i for i, c in enumerate(cur) if c in KK2RU]
        inner = [i for i in range(1, len(cur)) if cur[i - 1].isalpha() and cur[i].isalpha()]
        for _ in range(n_cand):
            kind = rng.choice([k for k, v in (("A1", pos), ("A2", inner), ("A4", kk)) if v] or ["A2"])
            if kind == "A1":
                i = rng.choice(pos); c = cur[i]
                cands.append(cur[:i] + (CYR2LAT.get(c) or LAT2CYR[c]) + cur[i + 1:])
            elif kind == "A4":
                i = rng.choice(kk); cands.append(cur[:i] + KK2RU[cur[i]] + cur[i + 1:])
            elif inner:
                i = rng.choice(inner); cands.append(cur[:i] + rng.choice(INVISIBLE_ATTACK) + cur[i:])
        if not cands:
            break
        P = model(cands)
        j = int(P[:, CRED].argmax())
        cur = cands[j]
        if P[j].argmax() == CRED:
            break
    return cur, {"n_edits": step + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--attacks", default="A1,A2,A3,A4,A6")
    ap.add_argument("--defense", default="none", choices=["none", "naive", "aware"])
    ap.add_argument("--max-n", type=int, default=0, help="subsample attacked set (e.g. 300 for slow models)")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--budgets", default="", help="override budgets for char attacks, e.g. A1=0.1;A2=5")
    a = ap.parse_args()
    te = pd.read_parquet(f"{SPLITS}/test.parquet")
    model = Counted(load_model(a.exp, a.model, a.seed), a.defense)
    if a.model.startswith("M9"):  # reuse the E1 clean predictions (one LLM pass over the test set is ~2 h)
        pr = pd.read_parquet(f"results/E1/{a.model}/13/predictions_test.parquet").set_index("id")
        te = te[te.id.isin(pr.index)].reset_index(drop=True)  # M9 was evaluated on a stratified subsample
        clean_p = pr.loc[te.id, ["p_credible", "p_misinfo", "p_forged"]].values
    else:
        clean_p = model(te.text.tolist())
    clean_pred = np.asarray(LABELS)[clean_p.argmax(1)]
    tgt = (te.label != "credible").values & (clean_pred == te.label.values)
    att = te[tgt].copy()
    if a.max_n and len(att) > a.max_n:
        att = att.sample(a.max_n, random_state=a.seed)
    name = a.model + ("" if a.defense == "none" else f"+{a.defense}")
    out_dir = f"results/{os.environ.get('E4_DIR', 'E4')}/{name}/{a.seed}"
    os.makedirs(out_dir, exist_ok=True)
    nonc = (te.label != "credible").values
    summary = {"n_test_noncredible": int(nonc.sum()), "clean_acc_noncredible": float((clean_pred[nonc] == te.label.values[nonc]).mean()),
               "n_attacked": len(att), "results": []}
    para = None
    for atk in a.attacks.split(","):
        budgets = BUDGETS.get(atk, [10])
        over = dict(kv.split("=") for kv in a.budgets.split(";") if kv)
        if atk in over:
            budgets = [float(v) if "." in v else int(v) for v in over[atk].split(",")]  # e.g. A1=0.1,0.2
        if atk == "A5":
            budgets = [0.05, 0.10, 0.20]
        if atk.startswith("A7"):
            budgets = [1]
            if para is None:
                para = pd.read_parquet("results/E4/A7_paraphrases.parquet")
                para = para[para.keep]
        samples = []
        for b in budgets:
            q0 = model.queries
            advs, infos = [], []
            cur = att
            if atk.startswith("A7"):  # A7_official / A7_neutral: only examples with an accepted paraphrase
                p = para[para["style"] == atk.split("_")[1]].set_index("id")
                cur = att[att.id.isin(p.index)]
            for k, (rid, t, lab) in enumerate(zip(cur.id, cur.text, cur.label)):
                seed = zlib.crc32(f"{rid}|{atk}|{b}|{a.seed}".encode())
                if atk == "A6":
                    adv, info = greedy_a6(t, model, max_edits=b, rng=random.Random(seed))
                elif atk == "A5":
                    adv, info = a5_attack(t, LABELS.index(lab), model, b, random.Random(seed))
                elif atk.startswith("A7"):
                    adv, info = p.loc[rid, "text_adv"], {"labse": float(p.loc[rid, "labse"]), "chrf": float(p.loc[rid, "chrf"])}
                else:
                    adv, info = apply(atk, t, b, seed)
                advs.append(adv); infos.append(info)
            if not advs:
                continue
            att_b = att
            att = cur  # metrics below refer to the evaluated subset
            P = model(advs)
            pred = np.asarray(LABELS)[P.argmax(1)]
            # robust accuracy over the full non-credible test set: unattacked misclassified stay wrong
            m = attack_metrics(att.label.values, att.label.values, pred)
            m.update({"attack": atk, "budget": b, "queries_per_example": (model.queries - q0) / max(1, len(att)),
                      # = clean accuracy x share of attacked examples that stay correct (exact on the full set,
                      # unbiased estimate when the attacked set is subsampled with --max-n)
                      "robust_acc_noncredible": summary["clean_acc_noncredible"] * float((pred == att.label.values).mean()),
                      "mean_frac_chars": float(np.mean([i.get("frac_chars", np.nan) for i in infos])) if infos and "frac_chars" in infos[0] else None,
                      "mean_frac_words": float(np.mean([i.get("frac_words", np.nan) for i in infos])) if infos and "frac_words" in infos[0] else None,
                      "mean_edits": float(np.mean([i.get("n_edits", np.nan) for i in infos])) if atk == "A6" else None,
                      "n_evaluated": len(att),
                      "mean_labse": float(np.mean([i["labse"] for i in infos])) if atk.startswith("A7") else None,
                      "mean_chrf": float(np.mean([i["chrf"] for i in infos])) if atk.startswith("A7") else None})
            flip = pred == "credible"
            for key, mask in [("real", ~att.is_synthetic.values.astype(bool)), ("synthetic", att.is_synthetic.values.astype(bool))] + \
                             [(f"lang_{g}", (att.lang == g).values) for g in ("kk", "ru", "mixed")]:
                m[f"asr_{key}"] = float(flip[mask].mean()) if mask.any() else None
                m[f"n_{key}"] = int(mask.sum())
            summary["results"].append(m)
            pd.DataFrame({"id": att.id.values, "y_true": att.label.values, "y_pred": pred, "p_credible": P[:, 0],
                          "p_misinfo": P[:, 1], "p_forged": P[:, 2], "lang": att.lang.values,
                          "is_synthetic": att.is_synthetic.values, "attack": atk, "budget": b,
                          "text_adv": advs}).to_parquet(f"{out_dir}/predictions_{atk}_{b}.parquet", index=False)
            ok = [(o, x, l) for o, x, l, p in zip(att.text, advs, att.label, pred) if p == "credible"]
            samples += [{"attack": atk, "budget": b, "label": l, "original": o, "adversarial": x} for o, x, l in ok]
            print(json.dumps({k: m[k] for k in ("attack", "budget", "asr", "robust_acc_noncredible", "queries_per_example")}), flush=True)
            att = att_b
        if samples:
            pd.DataFrame(samples).sample(min(a.n_samples, len(samples)), random_state=a.seed).to_csv(
                f"{out_dir}/samples_{atk}.csv", index=False)
        per = {k: v for k, v in summary.items() if k != "results"}
        per["results"] = [r for r in summary["results"] if r["attack"] == atk]
        json.dump(per, open(f"{out_dir}/metrics_{atk}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
