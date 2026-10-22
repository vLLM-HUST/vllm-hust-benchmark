#!/bin/bash
# ev3 rerun follow-up: (a) W2 adaptive-gate leg on the fixed build, unshimmed;
# (b) kernel anchors re-derived on the current build.
#
# Same as logs/v1-ev3/run_gate_and_anchors.sh except the ON leg carries NO shim.
set -u
cd /tmp
H=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev3-rerun
export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_DISABLE_COMPILE_CACHE=1
CANN_PP="${PYTHONPATH:-}"

for tag in gate32_r1 gate32_r2; do
  echo "=== BEGIN gate leg tag=$tag $(date -Is)" | tee -a "$H/run_gate_rerun.progress"
  export PYTHONPATH="$CANN_PP"
  export VLLM_ASCEND_CASCADE_TRACE=1
  export VLLM_ASCEND_CASCADE_ADAPTIVE_GATE=1
  BATCH=32 python3 "$H/cascade_sweep_plug.py.frozen" on "$tag" \
    > "$H/sweep_on_gate_b32_${tag##*_}.log" 2>&1
  echo "=== END gate leg tag=$tag rc=$? $(date -Is)" | tee -a "$H/run_gate_rerun.progress"
done

echo "=== BEGIN kernel anchors $(date -Is)" | tee -a "$H/run_gate_rerun.progress"
unset VLLM_ASCEND_CASCADE_TRACE VLLM_ASCEND_CASCADE_ADAPTIVE_GATE
unset VLLM_ASCEND_ENABLE_CASCADE_DECODE VLLM_ASCEND_ENABLE_CASCADE_GRAPH
unset VLLM_ASCEND_CASCADE_MIN_PREFIX VLLM_ASCEND_CASCADE_MIN_REQS
python3 /vllm-workspace/cascade-c3-results/probes/w0_b2_timing_probe.py 200 custom \
  > "$H/kernel_anchor_w0_custom.log" 2>&1
echo "=== END kernel anchors rc=$? $(date -Is)" | tee -a "$H/run_gate_rerun.progress"
echo "GATE_AND_ANCHORS RERUN DONE $(date -Is)" | tee -a "$H/run_gate_rerun.progress"
