#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server for the agents.
#   A100 40/80GB : defaults below (Qwen2.5-7B-Instruct, bf16)
#   T4 (Colab/Kaggle free tier): MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ DTYPE=half MAX_LEN=8192 ./scripts/serve_vllm.sh
set -euo pipefail
MODEL="${MODEL:-${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
DTYPE="${DTYPE:-bfloat16}"
MAX_LEN="${MAX_LEN:-16384}"
PORT="${PORT:-8000}"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
