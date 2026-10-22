#!/bin/bash
# DIAGNOSTIC: graph twin with the signature shim + VLLM_ASCEND_CASCADE_TRACE=1,
# to prove whether the twin captures/replays (the plugin's INFO capture markers
# go to a non-vLLM logger and are invisible in the serve log).
set -x
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2
RUN=/vllm-workspace/flashinfer-migration/cascade-evidence/ev2_run.py
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1

rm -rf /tmp/ev2_cache_traceB64_ie_on
env VLLM_CACHE_ROOT=/tmp/ev2_cache_traceB64_ie_on \
    VLLM_ASCEND_CASCADE_TRACE=1 EV2_SHIM_SIG=1 \
    EV2_BATCH=64 EV2_SENT=420 EV2_GEN=128 EV2_MAXLEN=8192 \
    EV2_EAGER=0 EV2_GRAPH=1 EV2_CAPTURE=32,64 EV2_IGNORE_EOS=1 \
    python -u "$RUN" on traceB64_ie_on > "$OUT/traceB64_ie_on.log" 2>&1
echo "rc=$? tag=traceB64_ie_on"
echo "=== capture-body success ==="; grep -c "capture body SUCCESS" "$OUT/traceB64_ie_on.log"
echo "=== capture-body entry ==="; grep -c "capture body: num_tokens" "$OUT/traceB64_ie_on.log"
echo "=== replay cascade key hit ==="; grep -c "replay update: cascade key hit" "$OUT/traceB64_ie_on.log"
echo "=== wrapper replay_swap ==="; grep -c "replay_swap=True" "$OUT/traceB64_ie_on.log"
echo "=== twin miss ==="; grep -c "twin miss" "$OUT/traceB64_ie_on.log"
echo "=== stage1 skip ==="; grep -c "stage1 re-bind skipped" "$OUT/traceB64_ie_on.log"
tail -3 "$OUT/traceB64_ie_on.log"
