# Script-Aware Alert Verification

Code for the paper **"Script-Aware Defenses Against Adversarial Evasion in Kazakh–Russian Content Verification for
Public Emergency Notification Platforms"** (ICIIT 2027).

Emergency notification platforms authenticate *who* sent a message, but not *whether its text can be trusted*.
A machine-learning content filter closes this gap and becomes a target itself. This repository contains the full
experimental pipeline: corpus construction, script-aware evasion attacks for bilingual Kazakh–Russian text,
a three-layer defense and the statistical evaluation.

## Method

| Layer | Module | What it does |
|---|---|---|
| 1. Script-aware normalization (D2) | `src/normalize.py` | removes format/invisible characters, applies NFKC without decomposing Kazakh letters (ә ғ қ ң ө ұ ү һ і), maps homoglyphs to each token's dominant script, keeps 2021 Kazakh Latin tokens, URLs and handles intact, reports anomalies |
| 2. Content classifier | `src/models/` | TF-IDF baselines (M1–M3), CNN-BiLSTM-Attention (M4), mBERT (M5), XLM-R (M6), Kaz-RoBERTa (M8), LLM zero/few-shot (M9/M10); adversarial training D3 via `src/attacks/augment.py` |
| 3. Risk fusion | `src/fusion.py`, `src/fusion_v2.py` | combines class probabilities, the Layer-1 report and (simulated) source features; monotone, adversarially trained anomaly weights; thresholds for publish / human review / block |

The naive baseline D1 (`normalize_naive`) maps every homoglyph to Cyrillic and therefore rewrites legitimate Kazakh
Latin text.

## Attacks (`src/attacks/`)

| ID | Attack | Budget |
|---|---|---|
| A1 | Cyrillic↔Latin homoglyphs | 1–20% of characters |
| A2 | zero-width characters and soft hyphens inside words | 1–20 insertions |
| A3 | partial transliteration into the 2021 Kazakh Latin alphabet | 10–100% of words |
| A4 | Kazakh-specific letters → Russian look-alikes | 10–100% of words |
| A5 | DeepWordBug-style typos (query-guided) | 5–20% of words |
| A6 | greedy search over A1+A2+A4 edits | ≤ 10 edits |
| A7 | LLM paraphrase into official or neutral style, LaBSE ≥ 0.80 | 1 paraphrase |

Character maps used by the attacks and the normalizer are in `src/charmaps.py`.

## Repository layout

```
src/data/     scraping of public sources, rumour extraction, synthesis, deduplication (MinHash), splits
src/attacks/  A1–A7 and the adversarial-training augmentation
src/models/   baselines, hybrid model, transformer fine-tuning, LLM classifier (llama.cpp server)
src/eval/     metrics, bootstrap / McNemar / Friedman, calibration, efficiency, explanations, audits, E12
src/viz/      figures (Okabe–Ito palette, PDF + PNG + alt text)
configs/      simulated source-metadata distributions (E6), LLM-judge prompt
scripts/      model download, full reproduction, number registry and its verification
tests/        unit tests for the normalizer, metrics and statistics
```

Every experiment writes `results/<exp>/<model>/<seed>/predictions*.parquet` and `metrics*.json`; tables, figures and
the number registry (`paper_numbers.json`) are built only from these files, and `scripts/verify_numbers.py` checks
that every reported number is backed by a result file.

## Setup

All experiments ran on CPUs (2× Intel Xeon Gold 6330N, Python 3.12); no GPU is required.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install --no-deps textattack==0.3.11
bash scripts/download_models.sh            # encoders, LaBSE, NLI model, Qwen3.5-35B-A3B (GGUF)
python -m pytest -q tests
```

The LLM components (rumour extraction, synthetic messages, M9/M10, A7, the A7 judge) use
[llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` with Qwen3.5-35B-A3B (Q4_K_M); set
`LLAMA_SERVER=/path/to/llama-server`.

## Reproduction

```bash
bash scripts/reproduce.sh     # several CPU-days; run inside tmux or nohup
```

Seeds are 13, 42 and 2026 for every trained model; results are reported as mean ± std over seeds.

## Data

The corpus is **not redistributed**. It is rebuilt by the scripts in `src/data/` from public sources: posts of the
official Telegram channels of the Ministry of Emergency Situations of Kazakhstan and of Kazhydromet (credible),
verbatim rumour quotes from debunking articles of factcheck.kz and Tengrinews (real misinformation), and synthetic
misinformation and forged alerts generated with an open LLM. Every row carries `is_synthetic`; results are always
reported separately for the real subset. Source metadata used by Layer 3 are simulated (`configs/E6.yaml`), so
Layer-3 results are a scenario analysis.

## Responsible use

The attack code implements well-known perturbation classes (homoglyphs, invisible characters, transliteration,
typos, paraphrasing) and is released together with the defenses to support robust deployment of content filters.
Synthetic forged alerts are generated only for training and evaluating detectors and are not distributed.

## License

Code: MIT (see `LICENSE`). Pretrained models keep their own licenses.
