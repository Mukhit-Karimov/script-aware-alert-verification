"""paper_numbers.json: flat {name: number} for the paper text, built ONLY from results/ and tables/ files."""
import glob
import json
import os

import pandas as pd

N = {}


def put(k, v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return
    N[k] = round(float(v), 10)  # full precision: rounding happens once, when a number is printed


def main():
    st = json.load(open("results/data_stats.json"))
    for s in ("train", "val", "test", "test_real", "test_ood_event"):
        put(f"data.{s}.n", st[s]["n"])
        for l, n in st[s]["label"].items():
            put(f"data.{s}.{l}", n)
    a = json.load(open("results/misinfo_label_audit.json"))
    put("data.audit.real_misinfo_test", a["n_real_misinfo_test"]); put("data.audit.valid", a["n_valid"]); put("data.audit.precision", a["precision"])
    put("data.langid.accuracy", json.load(open("results/langid_check.json"))["accuracy"])
    if os.path.exists("tables/T2_models.csv"):
        for _, r in pd.read_csv("tables/T2_models.csv").iterrows():
            p = f"E1.{r.model}.{r.slice}"
            put(f"{p}.macro_f1", r.macro_f1_mean); put(f"{p}.macro_f1_std", r.macro_f1_std)
            put(f"{p}.ci_lo", r.ci_lo); put(f"{p}.ci_hi", r.ci_hi); put(f"{p}.recall_forged", r.recall_forged)
            put(f"{p}.pr_auc", r.pr_auc); put(f"{p}.mcc", r.mcc)
    for f in glob.glob("results/E1/M*_*/13/metrics_test.json"):
        j = json.load(open(f)); put(f"E1.{f.split('/')[2].split('_')[0]}.invalid_share", j.get("invalid_share"))
    if os.path.exists("tables/T4_attacks.csv"):
        for _, r in pd.read_csv("tables/T4_attacks.csv").iterrows():
            p = f"E4.{r.model}.{r.attack}.budget{r.budget}"
            for c in ("asr", "asr_std", "robust_acc", "robust_acc_std", "clean_acc", "asr_real_verified", "asr_kk", "asr_ru", "queries"):
                put(f"{p}.{c}", r[c])
    if os.path.exists("tables/T5_defenses.csv"):
        for _, r in pd.read_csv("tables/T5_defenses.csv").iterrows():
            put(f"E5.{r.defense}.{r.attack}.budget{r.budget}.robust_acc", r.robust_acc); put(f"E5.{r.defense}.{r.attack}.budget{r.budget}.asr", r.asr)
        for _, r in pd.read_csv("tables/T5_defenses_clean.csv").iterrows():
            put(f"E5.{r.defense}.clean_f1", r.clean_f1); put(f"E5.{r.defense}.clean_f1_std", r.clean_f1_std); put(f"E5.{r.defense}.clean_f1_real_only", r.clean_f1_real)
    if os.path.exists("tables/T6_fusion.csv"):
        for _, r in pd.read_csv("tables/T6_fusion.csv").iterrows():
            for c in ("false_block_rate", "review_load", "missed_forgery_rate"):
                put(f"E6.{r.model}.{r.slice}.{c}", r[f"{c}_mean"])
    for f in glob.glob("results/E7/*/13/calibration.json"):
        j = json.load(open(f)); n = f.split("/")[2]
        put(f"E7.{n}.T", j["T"])
        for t in ("before", "after"):
            put(f"E7.{n}.{t}.ece", j[t]["ece"]); put(f"E7.{n}.{t}.brier", j[t]["brier"])
            for c, v in j[t]["risk_at_cov"].items():
                put(f"E7.{n}.{t}.risk_at_cov{c}", v)
    if os.path.exists("results/E8/efficiency.json"):
        for r in json.load(open("results/E8/efficiency.json")):
            p = f"E8.{r['model']}.t{r['threads']}.b{r['batch']}"
            put(f"{p}.p50_ms", r["p50_ms_per_msg"]); put(f"{p}.p95_ms", r["p95_ms_per_msg"]); put(f"{p}.msg_s", r["throughput_msg_s"]); put(f"{p}.size_mb", r["size_mb"])
    if os.path.exists("results/E9/fragmentation.json"):
        for tk, d in json.load(open("results/E9/fragmentation.json")).items():
            for atk, r in d.items():
                put(f"E9.frag.{tk}.{atk}.changed_before", r["subtok_per_changed_word_clean"]); put(f"E9.frag.{tk}.{atk}.changed_after", r["subtok_per_changed_word_adv"])
    if os.path.exists("results/E10_friedman.json"):
        j = json.load(open("results/E10_friedman.json")); put("E10.friedman.p", j["p"]); put("E10.nemenyi.cd", j["cd"])
    # ---- v2 (23.10.2026): E11 source ablation, label audit, Layer-3 CIs, automatic sample check, leakage audit
    if os.path.exists("tables/T9_source_ablation.csv"):
        for _, r in pd.read_csv("tables/T9_source_ablation.csv").iterrows():
            p = f"E11.{r.config.split(' ')[0].replace('+', '')}.{'D4' if 'D4' in r.config else 'D0'}.{r.metric.replace(' ', '_')}"
            put(f"{p}.orig", r.orig_mean); put(f"{p}.strip", r.strip_mean)
    if os.path.exists("results/E11/source_split_summary.csv"):
        for _, r in pd.read_csv("results/E11/source_split_summary.csv").iterrows():
            for c in ("macro_f1", "mcc", "recall_credible", "recall_misinfo", "recall_forged"):
                put(f"E11.source_split.{r.model}.{c}", r[f"{c}_mean"])
    if os.path.exists("results/E11/source_stats.json"):
        j = json.load(open("results/E11/source_stats.json"))
        put("E11.credible_total", j["credible_total"]); put("E11.credible_not_main_source", j["credible_not_main_source"])
        put("E11.share_main_source", j["share_main_source"])
    if os.path.exists("results/label_audit_summary.json"):
        j = json.load(open("results/label_audit_summary.json"))
        put("audit.train_val.n_suspicious", j["train_val"]["n_suspicious"]); put("audit.test.n_suspicious", j["test"]["n_suspicious"])
    if os.path.exists("results/E6/fusion_ci.csv"):
        for _, r in pd.read_csv("results/E6/fusion_ci.csv").iterrows():
            p = f"E6.{r.model}.{r.slice}.{r.metric}"
            put(f"{p}.ci_lo", r.ci_lo); put(f"{p}.ci_hi", r.ci_hi); put(f"{p}.val", r.val_value)
    if os.path.exists("results/E4/samples_auto_check.json"):
        j = json.load(open("results/E4/samples_auto_check.json"))
        put("E4.check.A1_A4.readable_strict", j["A1_A4"]["readable_strict_share"]); put("E4.check.A1_A4.readable_folded", j["A1_A4"]["readable_folded_share"])
        a = j.get("A7_all_kept_paraphrases")
        if a:
            for c in ("judge_meaning_share", "judge_natural_share", "nli_equivalent_share", "meaning_kept_share", "meaningful_share"):
                put(f"E4.check.A7.{c}", a[c])
    if os.path.exists("results/E4/human_vs_judge_kappa.json"):
        j = json.load(open("results/E4/human_vs_judge_kappa.json"))
        put("E4.check.kappa_meaning", j["kappa_meaning"]); put("E4.check.kappa_readable", j["kappa_readable"])
    if os.path.exists("results/leakage_audit.json"):
        j = json.load(open("results/leakage_audit.json"))
        for t in ("0.5", "0.7", "0.85"):
            put(f"leakage.near_dup_{t}.share_test", j[f"near_dup_jaccard_ge_{t}"]["share_test"])
        put("leakage.length_only_macro_f1", j["length_only_depth2_tree_macro_f1_test"])
    json.dump(dict(sorted(N.items())), open("paper_numbers.json", "w"), indent=1)
    print(len(N), "numbers")


if __name__ == "__main__":
    main()
