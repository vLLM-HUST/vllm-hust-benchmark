#!/usr/bin/env bash
set -eo pipefail
source /data/jxd/Ascend-9.1.0/cann-9.1.0/set_env.sh
source /data/jxd/Ascend-9.1.0/nnal/atb/set_env.sh
cd /tmp/simllm-issue3-similarity
PY=/data/jxd/envs/vllm-hust-v1-cann91/bin/python
WHEEL=/tmp/simllm-issue3-speed-nov2/wheels/new/vllm_ascend_simllm-0.1.0.dev0-py3-none-any.whl
MODEL=/data/jxd/Qwen3.5-35B-A3B-W8A8
PORT=8163
export ASCEND_RT_VISIBLE_DEVICES=4 COMPILE_CUSTOM_KERNELS=1 ASCEND_ATB_CXX_ABI=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:False TORCHINDUCTOR_AUTOGRAD_CACHE=0 PYTHONUNBUFFERED=1
$PY -m venv --system-site-packages venv
venv/bin/python -m pip install --no-deps --ignore-installed "$WHEEL" > install.log 2>&1
venv/bin/python -c 'import vllm_ascend_simllm,sys; print(vllm_ascend_simllm.__file__,sys.prefix)' > identity.txt
sha256sum "$WHEEL" "$MODEL/config.json" input-100.jsonl > SHA256SUMS
server_pid=''
cleanup() { if [ -n "$server_pid" ]; then kill -- -"$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; server_pid=''; fi; }
trap cleanup EXIT
for arm in baseline plugin_default plugin_floor8192; do
  cleanup
  case "$arm" in baseline) export VLLM_ASCEND_SIMLLM_ENABLED=0 VLLM_ASCEND_SIMLLM_MIN_PROMPT_TOKENS=0;; plugin_default) export VLLM_ASCEND_SIMLLM_ENABLED=1 VLLM_ASCEND_SIMLLM_MIN_PROMPT_TOKENS=0;; plugin_floor8192) export VLLM_ASCEND_SIMLLM_ENABLED=1 VLLM_ASCEND_SIMLLM_MIN_PROMPT_TOKENS=8192;; esac
  echo "START $arm $(date -u +%FT%TZ)" | tee -a progress.log
  setsid venv/bin/python -m vllm.entrypoints.cli.main serve "$MODEL" --served-model-name qwen35-similarity --trust-remote-code --quantization ascend --safetensors-load-strategy eager --tensor-parallel-size 1 --max-model-len 8192 --gpu-memory-utilization 0.85 --block-size 128 --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-seqs 4 --enforce-eager --seed 0 --host 127.0.0.1 --port "$PORT" > "$arm.server.log" 2>&1 &
  server_pid=$!
  ready=0
  for i in $(seq 1 600); do if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi; if ! kill -0 "$server_pid" 2>/dev/null; then break; fi; sleep 2; done
  if [ "$ready" -ne 1 ]; then echo "FAILED_READY $arm" | tee -a progress.log; exit 1; fi
  echo "READY $arm $(date -u +%FT%TZ)" | tee -a progress.log
  venv/bin/python bench_similarity_workload.py --input input-100.jsonl --url "http://127.0.0.1:$PORT" --model qwen35-similarity --max-tokens 1 --capture-text --chat --concurrency 4 > "$arm.outputs.jsonl"
  echo "DONE $arm $(date -u +%FT%TZ)" | tee -a progress.log
  cleanup
  sleep 5
done
