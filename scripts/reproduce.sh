#!/bin/bash
# Full pipeline on one CPU server (no GPU needed). Run inside tmux/nohup; the whole run takes several days of CPU time.
#   bash scripts/reproduce.sh
# Requirements: .venv with requirements.txt, models from scripts/download_models.sh, a llama.cpp build
# (LLAMA_SERVER=/path/to/llama-server). Optional NUMA pinning: RUN="numactl --cpunodebind=0 --membind=0".
set -eu
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs results
export NTHREADS=${NTHREADS:-28}
RUN=${RUN:-}
SEEDS="13 42 2026"
LLM=qwen3.5-35b-a3b
export LLM_URL=http://127.0.0.1:18080
python -m pytest -q tests

${LLAMA_SERVER:-llama-server} -m models/llm/Qwen3.5-35B-A3B-Q4_K_M.gguf --host 127.0.0.1 --port 18080 \
   -t $NTHREADS -c 32768 -np 8 --jinja > logs/llama_server.log 2>&1 &
SRV=$!
until curl -s 127.0.0.1:18080/health | grep -q ok; do sleep 10; done

# 1) corpus: public posts and debunk articles -> verbatim rumours -> synthetic messages -> dedup -> splits
for ch in kazhydromet qr_tjm; do python -m src.data.scrape_telegram $ch --min-date 2020-01-01 > logs/scrape_tg_$ch.log 2>&1; done
python -m src.data.scrape_news > logs/scrape_news.log 2>&1
python -m src.data.scrape_factcheck > logs/scrape_factcheck.log 2>&1
python -m src.data.judge_misinfo --workers 8 > logs/judge_misinfo.log 2>&1
python -m src.data.build_real && python -m src.data.make_corpus dedup 12000
python -m src.data.synth_llm forged --n 3000 --model $LLM --workers 8 > logs/synth_forged.log 2>&1
python -m src.data.synth_llm misinfo --n 2000 --model $LLM --workers 8 > logs/synth_misinfo.log 2>&1
python -m src.data.make_corpus build
python - <<'EOF'
import pandas as pd
for s in ["train", "val"]:
    d = pd.read_parquet(f"data/splits/{s}.parquet")
    for g in ["ru", "kk"]:
        d[d.lang == g].to_parquet(f"data/splits/{s}_{g}.parquet", index=False)
EOF
# real audited subset: rumours kept by the audit in results/review/test_real_misinfo_audit.csv
python - <<'EOF'
import pandas as pd
a = pd.read_csv("results/review/test_real_misinfo_audit.csv"); t = pd.read_parquet("data/splits/test_real.parquet")
t[(t.label == "credible") | t.id.isin(set(a[a.valid_rumour].id))].to_parquet("data/splits/test_real_verified.parquet", index=False)
EOF
python -m src.data.strip_source_markers          # E11 splits: data/splits_strip, data/splits_src
python -m src.eval.leakage_audit

# 2) E1 clean models (M1-M5, M6 = XLM-R, M8 = Kaz-RoBERTa; M7 = mDeBERTa-v3 was not run), LLM classifiers M9/M10
for s in $SEEDS; do
  for m in M1 M2 M3; do python -m src.models.baselines --model $m --seed $s >> logs/E1_$m.log 2>&1; done
  $RUN python -m src.models.hybrid --seed $s >> logs/E1_M4.log 2>&1
  for m in M5 M6 M8; do $RUN python -m src.models.transformer --model $m --seed $s >> logs/E1_$m.log 2>&1; done
done
python -m src.models.llm_classifier --model M9 --llm $LLM --split test --max-n 900 >> logs/E1_llm.log 2>&1
python -m src.models.llm_classifier --model M9 --llm $LLM --split test_ood_event >> logs/E1_llm.log 2>&1
python -m src.models.llm_classifier --model M10 --llm $LLM --split test --max-n 900 >> logs/E1_llm.log 2>&1
for g in ru kk; do                                   # E2 cross-lingual transfer (one seed)
  $RUN python -m src.models.transformer --model M6 --seed 13 --exp E2 --name M6_train_$g --train-file train_$g --val-file val_$g >> logs/E2.log 2>&1
done

# 3) D3 adversarial training (continued fine-tuning on train + attacked copies of 30% of messages)
for s in $SEEDS; do
  python -m src.attacks.augment --seed $s >> logs/E5_D3.log 2>&1
  $RUN python -m src.models.transformer --model M6 --seed $s --exp E5 --name M6_D3 --train-file train_adv_$s \
     --init results/E1/M6/$s/model --epochs 1 >> logs/E5_D3.log 2>&1
done

# 4) E4/E5 attacks against models and defenses (D1 = naive, D2 = aware normalization, D4 = D3 + D2)
python -m src.attacks.a7_paraphrase --llm $LLM --max-n 300 >> logs/E4_A7.log 2>&1
atk(){ $RUN python -m src.attacks.run_attacks --exp $1 --model $2 --seed $3 --defense $4 --attacks $5 >> logs/E4_attacks.log 2>&1; }
for s in $SEEDS; do
  for m in M1 M4 M5 M8; do atk E1 $m $s none A1,A2,A3,A4; done
  for d in none naive aware; do atk E1 M6 $s $d A1,A2,A3,A4; done
  for d in none aware; do atk E5 M6_D3 $s $d A1,A2,A3,A4; done
  for m in M1 M4; do atk E1 $m $s none A7_official,A7_neutral; done
  for d in none aware; do atk E1 M6 $s $d A7_official,A7_neutral; atk E5 M6_D3 $s $d A7_official,A7_neutral; done
done
atk E1 M1 13 none A5,A6; atk E1 M6 13 none A5,A6; atk E5 M6_D3 13 aware A5,A6
atk E1 M8 13 none A7_official,A7_neutral
python -m src.attacks.run_attacks --exp E1 --model M9_$LLM --seed 13 --attacks A1,A2,A7_official,A7_neutral --max-n 150 \
   --budgets "A1=0.1;A2=5" >> logs/E4_attacks.log 2>&1

# 5) A7 meaning check (LLM judge + LaBSE + NLI); after the blind human annotation: kappa
python -m src.eval.check_samples prep
python -m src.eval.check_samples judge >> logs/judge.log 2>&1
[ -f results/E4/human_check.csv ] && python -m src.eval.check_samples kappa >> logs/judge.log 2>&1 || true

# 6) E12 independent check on real forged official messages
python -m src.eval.real_forged screen --workers 8 >> logs/E12.log 2>&1
python -m src.eval.real_forged build >> logs/E12.log 2>&1
kill $SRV; wait $SRV 2>/dev/null || true
python -m src.eval.real_forged eval >> logs/E12.log 2>&1

# 7) E11 source-marker ablation and source split
BUD="A1=0.1,0.2;A2=5,10,20;A3=0.1,0.2,0.5,1.0;A4=0.1,0.2,0.5,1.0"
for s in $SEEDS; do
  SPLITS=data/splits_strip python -m src.models.baselines --model M1 --seed $s --exp E11_strip >> logs/E11.log 2>&1
  SPLITS=data/splits_strip $RUN python -m src.models.transformer --model M6 --seed $s --exp E11_strip >> logs/E11.log 2>&1
  SPLITS=data/splits_strip python -m src.attacks.augment --seed $s >> logs/E11.log 2>&1
  SPLITS=data/splits_strip $RUN python -m src.models.transformer --model M6 --seed $s --exp E11_strip --name M6_D3 \
     --train-file train_adv_$s --init results/E11_strip/M6/$s/model --epochs 1 >> logs/E11.log 2>&1
  SPLITS=data/splits_strip python -m src.eval.eval_defense --exp E11_strip --model M6_D3 --seed $s --defense aware \
     --name M6_D3+aware --out-exp E11_strip >> logs/E11.log 2>&1
  for cfg in "M1 none" "M6 none" "M6_D3 aware"; do set -- $cfg
    SPLITS=data/splits_strip E4_DIR=E11_strip/E4 $RUN python -m src.attacks.run_attacks --exp E11_strip --model $1 --seed $s \
       --defense $2 --attacks A1,A2,A3,A4 --budgets "$BUD" >> logs/E11.log 2>&1
  done
  E4_DIR=E11_orig/E4 python -m src.attacks.run_attacks --exp E1 --model M1 --seed $s --attacks A1,A2,A3,A4 --budgets "$BUD" >> logs/E11.log 2>&1
  E4_DIR=E11_orig/E4 $RUN python -m src.attacks.run_attacks --exp E1 --model M6 --seed $s --attacks A1,A2,A3,A4 --budgets "$BUD" >> logs/E11.log 2>&1
  E4_DIR=E11_orig/E4 $RUN python -m src.attacks.run_attacks --exp E5 --model M6_D3 --seed $s --defense aware \
     --attacks A1,A2,A3,A4 --budgets "$BUD" >> logs/E11.log 2>&1
  SPLITS=data/splits_src python -m src.models.baselines --model M1 --seed $s --exp E11_src >> logs/E11.log 2>&1
done
SPLITS=data/splits_src $RUN python -m src.models.transformer --model M6 --seed 13 --exp E11_src >> logs/E11.log 2>&1

# 8) clean evaluation of defenses, Layer 3 (E6, E6b, E6c), calibration (E7), efficiency (E8), explanations (E9)
for s in $SEEDS; do
  python -m src.eval.eval_defense --exp E1 --model M6 --seed $s --defense naive --name M6+naive >> logs/E5_eval.log 2>&1
  python -m src.eval.eval_defense --exp E1 --model M6 --seed $s --defense aware --name M6+aware >> logs/E5_eval.log 2>&1
  python -m src.eval.eval_defense --exp E5 --model M6_D3 --seed $s --defense aware --name M6_D3+aware >> logs/E5_eval.log 2>&1
  python -m src.fusion --seed $s >> logs/E6.log 2>&1
done
python -m src.eval.fusion_attacked >> logs/E6_attacked.log 2>&1
python -m src.fusion_v2 >> logs/E6c.log 2>&1
python -m src.eval.fusion_ci >> logs/E6.log 2>&1
python -m src.eval.calibration --pred-dir results/E1/M1 --name M1 >> logs/E7.log 2>&1
python -m src.eval.calibration --pred-dir results/E1/M6 --name M6 >> logs/E7.log 2>&1
python -m src.eval.calibration --pred-dir results/E5/M6_D3+aware --name M6_D4 >> logs/E7.log 2>&1
python -m src.eval.efficiency >> logs/E8.log 2>&1            # run on an otherwise idle machine
python -m src.eval.explain m1 >> logs/E9.log 2>&1
python -m src.eval.explain shap >> logs/E9.log 2>&1
python -m src.eval.fragmentation >> logs/E9.log 2>&1

# 9) statistics (E10), label audit, tables, figures, number registry and its check
python scripts/extra_stats.py mcnemar samples t8 budget >> logs/report.log 2>&1
python -m src.eval.refresh_metrics results/E1 results/E2 results/E5 results/E11_strip results/E11_src >> logs/report.log 2>&1
python -m src.eval.label_audit >> logs/report.log 2>&1
for m in data_figs e1_figs e4e5_figs other_figs e11_figs; do python -m src.viz.$m >> logs/report.log 2>&1; done
python scripts/make_paper_numbers.py >> logs/report.log 2>&1
python scripts/verify_numbers.py > logs/verify.log 2>&1
