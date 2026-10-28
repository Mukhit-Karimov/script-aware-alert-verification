#!/bin/bash
# Pretrained encoders -> models/<name>, LLM (GGUF, for llama.cpp) -> models/llm.
set -u
cd "$(dirname "$0")/.."
for pair in FacebookAI/xlm-roberta-base:xlm-roberta-base google-bert/bert-base-multilingual-cased:bert-base-multilingual-cased \
  kz-transformers/kaz-roberta-conversational:kaz-roberta-conversational sentence-transformers/LaBSE:LaBSE \
  MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7:nli_mdeberta_xnli; do
  repo=${pair%%:*}; name=${pair##*:}
  hf download "$repo" --local-dir "models/$name" --exclude "*.h5" --exclude "*.msgpack" --exclude "*.ot" --exclude "onnx/*" \
     --exclude "openvino/*" --exclude "tf_model*" --exclude "flax*" --exclude "rust_model*" --exclude "*.onnx" || echo "FAIL $repo"
done
mkdir -p models/llm
hf download unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q4_K_M.gguf --local-dir models/llm || echo "FAIL Qwen3.5-35B-A3B"
