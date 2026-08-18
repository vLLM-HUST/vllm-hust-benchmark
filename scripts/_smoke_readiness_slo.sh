#!/usr/bin/env bash
# Minimal smoke test: start vllm server, wait for readiness, run bench serve once.
# Captures server.log, client_result.json, /metrics output for format inspection.
set -euo pipefail

MODEL_PATH="${VLLM_HUST_MODEL:-/data/shared_datasets/strict-models/Qwen2.5-14B-Instruct}"
PORT="${VLLM_HUST_PORT:-8011}"
OUTPUT_DIR="${VLLM_HUST_OUTPUT:-/tmp/readiness_smoke}"
PYTHON_BIN="${VLLM_HUST_PYTHON:-/root/miniconda3/envs/vllm-hust-dev/bin/python}"
VLLM_HUST_REPO="${VLLM_HUST_REPO:-/root/vllm/vllm-hust}"
VLLM_ASCEND_HUST_REPO="${VLLM_HUST_ASCEND_REPO:-/root/vllm/vllm-ascend-hust}"
CLI_COMPAT="${VLLM_HUST_CLI_COMPAT:-/root/vllm/vllm-hust-benchmark/scripts/run_vllm_cli_compat.py}"

mkdir -p "${OUTPUT_DIR}"
SERVER_LOG="${OUTPUT_DIR}/server.log"
CLIENT_RESULT="${OUTPUT_DIR}/client_result.json"
METRICS_OUT="${OUTPUT_DIR}/metrics.txt"
STARTUP_TS="${OUTPUT_DIR}/startup_ts.txt"

# Source Ascend env (these scripts use unbound vars; run under set +u).
export ZSH_VERSION=""
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u
export LD_LIBRARY_PATH="/root/miniconda3/envs/vllm-hust-dev/lib:${LD_LIBRARY_PATH:-}"
# Prepend vllm repos but preserve PYTHONPATH from Ascend set_env.sh (pyACL).
export PYTHONPATH="${VLLM_ASCEND_HUST_REPO}:${VLLM_HUST_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/data/shared_datasets/vllm-hust-benchmark/huggingface
export VLLM_CACHE_ROOT=/tmp/readiness_smoke_cache
rm -rf "${VLLM_CACHE_ROOT}"
mkdir -p "${VLLM_CACHE_ROOT}"
# Bypass proxy for local server health checks and bench serve requests.
# The shared login shell exports http_proxy/https_proxy/all_proxy; --noproxy on
# curl does not reliably bypass all_proxy (SOCKS), so force the env var too.
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

# The conda lib dir prepended to LD_LIBRARY_PATH below breaks system curl
# (libldap.so.2 -> OpenSSL symbol mismatch). Health/metrics checks must run with
# the system library path, so wrap curl to drop the conda override.
sys_curl() {
    env LD_LIBRARY_PATH= curl --noproxy "*" "$@"
}

echo "INFO: starting server (model=${MODEL_PATH}, port=${PORT})"
date -u +%Y-%m-%dT%H:%M:%SZ > "${STARTUP_TS}"

setsid "${PYTHON_BIN}" -u -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name Qwen2.5-14B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.6 \
  --enforce-eager \
  --trust-remote-code \
  --no-enable-log-requests \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-model-len 32768 \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PGID=$!

cleanup() {
  if [[ -n "${SERVER_PGID}" ]]; then
    kill -- -"${SERVER_PGID}" 2>/dev/null || true
    wait "${SERVER_PGID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT TERM INT

echo "INFO: waiting for readiness (pgid=${SERVER_PGID})"
READY=0
for i in $(seq 1 600); do
  if sys_curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY=1
    echo "INFO: server ready after ${i}s"
    break
  fi
  sleep 1
done

if [[ ${READY} -ne 1 ]]; then
  echo "ERROR: server failed readiness after 600s" >&2
  tail -30 "${SERVER_LOG}" >&2
  exit 1
fi

date -u +%Y-%m-%dT%H:%M:%SZ >> "${STARTUP_TS}"

echo "INFO: running bench serve (random-online, 50 prompts, 1 RPS)"
NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost" \
"${PYTHON_BIN}" "${CLI_COMPAT}" bench serve \
  --save-result \
  --result-dir "$(dirname "${CLIENT_RESULT}")" \
  --result-filename "$(basename "${CLIENT_RESULT}")" \
  --backend vllm \
  --endpoint /v1/completions \
  --base-url "http://127.0.0.1:${PORT}" \
  --dataset-name random \
  --model Qwen2.5-14B-Instruct \
  --tokenizer "${MODEL_PATH}" \
  --num-prompts 50 \
  --input-len 1024 \
  --output-len 256 \
  --request-rate 1

echo "INFO: fetching /metrics"
sys_curl -sf "http://127.0.0.1:${PORT}/metrics" > "${METRICS_OUT}" || echo "WARN: /metrics unavailable" >&2

echo "INFO: smoke test complete"
echo "SERVER_LOG=${SERVER_LOG}"
echo "CLIENT_RESULT=${CLIENT_RESULT}"
echo "METRICS_OUT=${METRICS_OUT}"
echo "---CLIENT RESULT (head)---"
head -c 2000 "${CLIENT_RESULT}"
echo ""
echo "---METRICS (kv-related, head)---"
grep -iE "kv_cache|preempt|evict|vllm:num" "${METRICS_OUT}" | head -15 || echo "no kv metrics found"
