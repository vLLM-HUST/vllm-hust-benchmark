#!/usr/bin/env bash
# Native comparison arm: same model/cache settings as MuxPD; native chunked prefill
# and async scheduling stay at their defaults. Extra args override (e.g. budget, graphs).
set -eo pipefail
source /root/workspace/runtime/env.sh >/dev/null 2>&1
exec /root/workspace/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /root/workspace/models/Qwen3.5-35B-A3B --served-model-name qwen3.5-35b \
  --host 127.0.0.1 --port "${VLLM_SERVER_PORT:-8300}" \
  --tensor-parallel-size "${MUXPD_TP_SIZE:-2}" --enable-expert-parallel \
  --dtype bfloat16 --max-model-len 16384 --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 --language-model-only \
  --no-enable-prefix-caching --seed 0 "$@"
