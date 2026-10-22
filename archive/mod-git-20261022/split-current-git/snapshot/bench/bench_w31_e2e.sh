#!/bin/bash
# W3.1 fi_gelu e2e A/B -- thin driver around bench_w31_e2e.py.
#
# Pre-registered knobs live in the Python harness (and in
# bench/results/EXPECTATIONS-w31-e2e.md); this wrapper only fixes the host
# contract: run from /tmp (never /vllm-workspace), NPU card 7, offline HF,
# and the shared torch.compile AOT cache disabled (the shared cache HIT hard-
# crashes engine init on this v1 baseline -- see cascade section3-performance.md;
# disabling is symmetric across arms and happens outside the timed region).
#
# Usage:
#   nohup bash bench/bench_w31_e2e.sh > /tmp/w31_e2e/run.log 2>&1 &
# (equivalent to the plain `python3 bench_w31_e2e.py` invocation from /tmp)
set -u
cd /tmp || exit 1
export ASCEND_RT_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1
export VLLM_DISABLE_COMPILE_CACHE=1
mkdir -p /tmp/w31_e2e
exec python3 "$(dirname "$(readlink -f "$0")")/bench_w31_e2e.py" "$@"
