#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# e2e TPOT A/B for the fi_sampling plugin (W2b package).
#
# Usage:  bash run_e2e_tpot.sh <card> <case> <mode>
#   card  6|7                      NPU card (must be idle; ASCEND_RT_VISIBLE_DEVICES)
#   case  api1|joint               workload (see below)
#   mode  off|on                   VLLM_HUST_FI_SAMPLING
#
# Workloads (both use the SAME random dataset and generation lengths; only the
# truncation parameters and concurrency differ):
#   api1  : no top-k/top-p (vLLM default random sampling) -- the main win region
#   joint : top_k=50 / top_p=0.95, concurrency 256          -- the joint win region
#
# CWD is forced to /tmp: running any vllm command from /vllm-workspace is
# forbidden (the source checkout shadows the installed vllm package).

set -euo pipefail

CARD="${1:?card}"
CASE="${2:?case}"
MODE="${3:?mode}"

MODEL=/data/shared_models/Qwen--Qwen2.5-Coder-14B-Instruct
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${CASE}_${MODE}"
PORT=8321
LOG="/tmp/w2b_serve_${TAG}.log"
CLIENT_LOG="/tmp/w2b_bench_${TAG}.log"

case "${CASE}" in
  api1)
    CLIENT_ARGS=(--dataset-name random --random-input-len 1024 --random-output-len 256
                 --num-prompts 256 --max-concurrency 64)
    ;;
  joint)
    CLIENT_ARGS=(--dataset-name random --random-input-len 1024 --random-output-len 256
                 --num-prompts 512 --max-concurrency 256 --top-k 50 --top-p 0.95)
    ;;
  *)
    echo "unknown case ${CASE}" >&2; exit 2 ;;
esac

if [ "${MODE}" = "on" ]; then
  export VLLM_HUST_FI_SAMPLING=1
  export VLLM_HUST_FI_SAMPLING_TRACE=1
else
  unset VLLM_HUST_FI_SAMPLING || true
  unset VLLM_HUST_FI_SAMPLING_TRACE || true
fi

export ASCEND_RT_VISIBLE_DEVICES="${CARD}"
export HF_HUB_OFFLINE=1

cd /tmp
nohup vllm serve "${MODEL}" \
  --max-model-len 4096 --enforce-eager --max-num-seqs 512 \
  --served-model-name qwen14b --port "${PORT}" \
  > "${LOG}" 2>&1 &
SERVE_PID=$!
echo "serve pid=${SERVE_PID} log=${LOG}"

cleanup() {
  kill "${SERVE_PID}" 2>/dev/null || true
  wait "${SERVE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "server died during startup" >&2; tail -40 "${LOG}" >&2; exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "health timeout" >&2; exit 1; }

# Prove the FI route actually engaged (or not) for this leg.
# NOTE: `grep -c` exits 1 on zero matches; under `set -e`/pipefail that would
# abort the run, so the count is captured with `|| true`.
fi_active=$(grep -c "fi_sampling sampling path is ACTIVE" "${LOG}" || true)
echo "fi-active-lines=${fi_active}"
if [ "${MODE}" = "on" ] && [ "${fi_active}" -lt 1 ]; then
  echo "ERROR: FI plugin did not activate in the on leg" >&2; exit 1
fi
if [ "${MODE}" = "off" ] && [ "${fi_active}" -ne 0 ]; then
  echo "ERROR: FI plugin activated in the off leg" >&2; exit 1
fi

vllm bench serve \
  --backend openai-chat --model qwen14b --tokenizer "${MODEL}" \
  --host 127.0.0.1 --port "${PORT}" \
  --endpoint /v1/chat/completions \
  --ignore-eos --num-warmups 8 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99 \
  --save-result --result-dir "${OUT_DIR}/results" \
  --result-filename "serve_${TAG}.json" \
  "${CLIENT_ARGS[@]}" 2>&1 | tee "${CLIENT_LOG}"

echo "==== ${TAG} done ===="
if [ "${MODE}" = "on" ]; then
  echo "--- route trace (first decisions) ---"
  grep "fi-sampling trace" "${LOG}" | tail -5 || true
  echo "--- route histogram ---"
  grep "fi-sampling histogram" "${LOG}" | tail -2 || true
fi
