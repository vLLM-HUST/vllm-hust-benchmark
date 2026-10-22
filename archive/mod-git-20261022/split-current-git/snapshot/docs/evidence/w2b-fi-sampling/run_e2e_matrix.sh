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
# e2e TPOT matrix for the fi_sampling plugin (W2b package).
#
# Usage: bash run_e2e_matrix.sh <card> <case> <repeat>
#   card   6|7       NPU card (idle; ASCEND_RT_VISIBLE_DEVICES)
#   case   api1|joint
#   repeat  number of independent server lifecycles (>=2 for a median)
#
# Cases (all use the random dataset, ignore_eos, fixed lengths):
#   api1  : no top-k/top-p (vLLM default random sampling)  -> the main win region
#           max-model-len 4096, 1024 in / 256 out, concurrency 64
#   joint : top_k=50 / top_p=0.95, concurrency 512          -> the joint win region
#           max-model-len 512, 128 in / 64 out
#           (the context is deliberately short: a single 910B2 cannot hold
#            B>=256 with a 4k context, and the joint route is gated on B>=256)
#
# CWD is forced to /tmp: running any vllm command from /vllm-workspace is
# forbidden (the source checkout shadows the installed vllm package).

set -euo pipefail

CARD="${1:?card}"
CASE="${2:?case}"
REPEAT="${3:?repeat}"

MODEL=/data/shared_models/Qwen--Qwen2.5-Coder-14B-Instruct
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8331

case "${CASE}" in
  api1)
    SERVER_ARGS=(--max-model-len 4096 --max-num-seqs 512)
    CLIENT_ARGS=(--dataset-name random --random-input-len 1024 --random-output-len 256
                 --num-prompts 256 --max-concurrency 64)
    ;;
  joint)
    SERVER_ARGS=(--max-model-len 512 --max-num-seqs 512)
    CLIENT_ARGS=(--dataset-name random --random-input-len 128 --random-output-len 64
                 --num-prompts 512 --max-concurrency 512 --top-k 50 --top-p 0.95)
    ;;
  *)
    echo "unknown case ${CASE}" >&2; exit 2 ;;
esac

export ASCEND_RT_VISIBLE_DEVICES="${CARD}"
export HF_HUB_OFFLINE=1
cd /tmp

for rep in $(seq 1 "${REPEAT}"); do
  for mode in off on; do
    TAG="${CASE}_${mode}_r${rep}"
    LOG="/tmp/w2b_serve_${TAG}.log"
    CLIENT_LOG="${OUT_DIR}/logs_bench_${TAG}.txt"

    if [ "${mode}" = "on" ]; then
      export VLLM_HUST_FI_SAMPLING=1
      export VLLM_HUST_FI_SAMPLING_TRACE=1
    else
      unset VLLM_HUST_FI_SAMPLING || true
      unset VLLM_HUST_FI_SAMPLING_TRACE || true
    fi

    echo "==== ${TAG} : starting server ===="
    # --generation-config vllm: the model's generation_config.json otherwise
    # overrides the served defaults with top_k=20/top_p=0.8, which would put
    # even the "no truncation" case on the truncated (fork) route.
    nohup vllm serve "${MODEL}" --enforce-eager --served-model-name qwen14b \
      --generation-config vllm \
      --port "${PORT}" "${SERVER_ARGS[@]}" > "${LOG}" 2>&1 &
    SERVE_PID=$!
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

    # `grep -c` exits 1 on zero matches; under pipefail that aborts the run.
    fi_active=$(grep -c "fi_sampling sampling path is ACTIVE" "${LOG}" || true)
    echo "fi-active-lines=${fi_active}"
    if [ "${mode}" = "on" ] && [ "${fi_active}" -lt 1 ]; then
      echo "ERROR: FI plugin did not activate in the on leg" >&2; exit 1
    fi
    if [ "${mode}" = "off" ] && [ "${fi_active}" -ne 0 ]; then
      echo "ERROR: FI plugin activated in the off leg" >&2; exit 1
    fi

    vllm bench serve \
      --backend openai-chat --model qwen14b --tokenizer "${MODEL}" \
      --host 127.0.0.1 --port "${PORT}" --endpoint /v1/chat/completions \
      --ignore-eos --num-warmups 8 \
      --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99 \
      --save-result --result-dir "${OUT_DIR}/results" \
      --result-filename "serve_${TAG}.json" \
      "${CLIENT_ARGS[@]}" > "${CLIENT_LOG}" 2>&1

    echo "--- ${TAG} metrics ---"
    grep -E "Successful requests|Failed requests|Mean TPOT|Median TPOT|P95 TPOT|P99 TPOT|Mean TTFT|Output token throughput" "${CLIENT_LOG}"
    if [ "${mode}" = "on" ]; then
      echo "--- ${TAG} route histogram (final) ---"
      grep -o "histogram calls=[0-9]* max_B=[0-9]*\] {[^}]*}" "${LOG}" | tail -2 || true
      grep -oE "route=[a-z_0-9]+ B=[0-9]+" "${LOG}" | sort | uniq -c | sort -rn | head -3 || true
    fi

    cleanup
    trap - EXIT
    sleep 5
  done
done

echo "==== matrix ${CASE} done ===="
