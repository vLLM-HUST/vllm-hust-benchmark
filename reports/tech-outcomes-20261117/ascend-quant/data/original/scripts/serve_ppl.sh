#!/usr/bin/env bash
# Launch one server for the wikitext-2 PPL A/B, reusing run_swe_server.sh.
#
# Same CANN 9.1 per-process environment as the running TP1 curve server
# (read off /proc/<pid>/environ of the live process), so bf16 and w8a8 are
# served by an identical stack and differ only by --quantization ascend.
#
# Usage:  ./serve_ppl.sh <bf16|w8a8> <cards e.g. 1,2> <port>
#   LABEL=w8a8 CARD=3,4 PORT=8102 ./serve_ppl.sh w8a8 3,4 8102
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CANN_SET_ENV=/data/jxd/Ascend-9.1.0/ascend-toolkit/set_env.sh
export ATB_SET_ENV=/data/jxd/Ascend-9.1.0/nnal/atb/set_env.sh
export VENDOR_OPP=1
export PERF_FIX_KERNELS_SO=0          # CANN 9.1 -> keep the original kernels so
export TORCHINDUCTOR_AUTOGRAD_CACHE=0 # and do not swap in the CANN8.5 build
export PERF_CACHE_ROOT=/data/jxd/cache/cann91-w8a8

LABEL="${1:?usage: serve_ppl.sh <bf16|w8a8> <cards> <port>}"
CARD="${2:?cards}"
PORT="${3:?port}"

# Identical on both models so the A/B differs only by --quantization ascend.
#   * capture_size 64 - the default 35 ACL graph sizes hit EE1023
#     (rtStreamBeginCapture resource alloc fail) at TP2 on this box; 35 -> 11.
#   * prefetch       - DPC is not recognised as a network FS so vLLM disables
#     auto-prefetch; forcing it cuts the 67 GiB bf16 load from ~15 min.
# NB: do not write this as ${EXTRA_SERVER_ARGS:-...} - bash ends the expansion at
# the first '}', which swallows the JSON closing brace and fuses the two args.
if [[ -z "${EXTRA_SERVER_ARGS:-}" ]]; then
    EXTRA_SERVER_ARGS='--compilation-config {"max_cudagraph_capture_size":64} --safetensors-load-strategy=prefetch'
fi
export EXTRA_SERVER_ARGS

echo "[ppl] label=${LABEL} cards=${CARD} port=${PORT} ctx=${MAX_MODEL_LEN:-4096}"
exec env LABEL="${LABEL}" TP="${TP:-2}" PORT="${PORT}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" \
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}" \
    ASCEND_RT_VISIBLE_DEVICES="${CARD}" \
    "${SCRIPT_DIR}/run_swe_server.sh"
