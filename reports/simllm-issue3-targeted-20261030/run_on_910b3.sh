#!/usr/bin/env bash
set -eo pipefail

# Qualified train8-docker diagnostic cell. Use a free card and port; the
# harness creates only a disposable package overlay and owned server processes.
source /data/jxd/Ascend-9.1.0/cann-9.1.0/set_env.sh
source /data/jxd/Ascend-9.1.0/nnal/atb/set_env.sh
set -u
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/data/jxd/cache/torch_ext
export TORCHINDUCTOR_CACHE_DIR=/data/jxd/cache/torchinductor
export TRITON_CACHE_DIR=/data/jxd/cache/triton
export VLLM_CACHE_ROOT=/tmp/simllm-issue3-quality-oct30/cache

root=/tmp/simllm-issue3-quality-oct30
/data/jxd/envs/vllm-hust-v1-cann91/bin/python \
  "$root/run_quality_apc.py" \
  --final-wheel "$root/wheels/final/vllm_ascend_simllm-0.1.0.dev0-py3-none-any.whl" \
  --rollback-wheel "$root/wheels/rollback/vllm_ascend_simllm-0.1.0.dev0-py3-none-any.whl" \
  --model "$root/model-local" \
  --input "$root/workload.jsonl" \
  --device 7 --port 8147 --output "$root/run1" "$@"
