#!/bin/bash
# W2b fi_sampling e2e A/B refresh -- thin driver around
# bench_fisampling_refresh_e2e.py.
#
# Pre-registered knobs live in the Python harness; this wrapper only fixes the
# host contract: run from /tmp (never /vllm-workspace), NPU card 7 (checked
# idle), offline HF, shared torch.compile AOT cache disabled (its HIT hard-
# crashes engine init on this baseline -- symmetric across arms), and the
# SPARE-card lock /tmp/w3-npu.lock so a concurrent job cannot steal card 7.
#
# Usage:
#   nohup flock /tmp/w3-npu.lock -c \
#     'cd /tmp && ASCEND_RT_VISIBLE_DEVICES=7 bash \
#      /vllm-workspace/vllm-ascend-split-batch-hust/bench/bench_fisampling_refresh_e2e.sh' \
#     > /tmp/fisampling_refresh.log 2>&1 &
set -u
cd /tmp || exit 1
# The login shell exports ASCEND_RT_VISIBLE_DEVICES=0 (a busy card), so the card
# is pinned here -- never inherited -- and only overridable via FISAMPLING_CARD.
export ASCEND_RT_VISIBLE_DEVICES="${FISAMPLING_CARD:-7}"
export HF_HUB_OFFLINE=1
export VLLM_DISABLE_COMPILE_CACHE=1
mkdir -p /tmp/fisampling_refresh
exec python3 "$(dirname "$(readlink -f "$0")")/bench_fisampling_refresh_e2e.py" "$@"
