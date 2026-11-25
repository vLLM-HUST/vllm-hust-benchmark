#!/usr/bin/env bash
# Full prompt budget is intentional: MuxPD never splits a prefill into chunks.
set -eo pipefail
workspace="${MUXPD_WORKSPACE:-/root/workspace}"
source "$workspace/runtime/env.sh"
set -u
exec "$workspace/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$workspace/models/Qwen3.5-35B-A3B" \
  --served-model-name Qwen3.5-35B-A3B \
  --host 127.0.0.1 --port "${VLLM_SERVER_PORT:-8002}" \
  --tensor-parallel-size "${MUXPD_TP_SIZE:-2}" --enable-expert-parallel \
  --dtype bfloat16 --max-model-len 16384 --max-num-seqs 8 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 \
  --language-model-only --enforce-eager \
  --no-enable-chunked-prefill --no-async-scheduling \
  --no-enable-prefix-caching --mamba-cache-mode none \
  --scheduler-cls vllm_muxpd.scheduler.MuxPDScheduler \
  --batch-admission-policy vllm_muxpd.policy.MuxPDPolicy \
  "$@"
