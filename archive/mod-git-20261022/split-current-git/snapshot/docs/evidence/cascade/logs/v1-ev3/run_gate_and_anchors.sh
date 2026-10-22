#!/bin/bash
# ev3 follow-up 2: (a) W2 adaptive-gate leg on the new baseline, (b) kernel anchors.
#
# (a) ON graph + VLLM_ASCEND_CASCADE_ADAPTIVE_GATE=1.  The gate micro-bench runs
#     in cascade_gate_self.py (subprocess) and prints one [cas-gate] line per
#     (N, prefix) cell; the replay paths then consume the verdict.  BATCH=32
#     because p420_b32 is the measured loss cell.
# (b) w0_b2_timing_probe.py "custom" mode: fa_fp32_stage1 @4356/@8192,
#     stage-2 FIA @260 B64, lse_merge @B64H40 — the kernel-level anchors that
#     EVIDENCE.md §3 lists.  Read-only script from cascade-c3-results/probes.
set -u
cd /tmp
H=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev3
export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_DISABLE_COMPILE_CACHE=1

echo "=== BEGIN gate leg $(date -Is)" | tee -a "$H/run_gate.progress"
export PYTHONPATH="$H/shim:${PYTHONPATH:-}"
export VLLM_ASCEND_CASCADE_TRACE=1
export VLLM_ASCEND_CASCADE_ADAPTIVE_GATE=1
BATCH=32 python3 "$H/cascade_sweep_plug.py.frozen" on gate32_v1 \
  > "$H/sweep_on_gate_b32.log" 2>&1
echo "=== END gate leg rc=$? $(date -Is)" | tee -a "$H/run_gate.progress"

echo "=== BEGIN kernel anchors $(date -Is)" | tee -a "$H/run_gate.progress"
# Drop the shim from PYTHONPATH but KEEP the CANN entries (acl/tbe live there;
# clearing PYTHONPATH makes torch_npu fail with ModuleNotFoundError: acl).
CANN_PP="$PYTHONPATH"
CANN_PP="${CANN_PP#"$H/shim":}"
export PYTHONPATH="$CANN_PP"
unset VLLM_ASCEND_CASCADE_TRACE VLLM_ASCEND_CASCADE_ADAPTIVE_GATE
unset VLLM_ASCEND_ENABLE_CASCADE_DECODE VLLM_ASCEND_ENABLE_CASCADE_GRAPH
unset VLLM_ASCEND_CASCADE_MIN_PREFIX VLLM_ASCEND_CASCADE_MIN_REQS
python3 /vllm-workspace/cascade-c3-results/probes/w0_b2_timing_probe.py 200 custom \
  > "$H/kernel_anchor_w0_custom.log" 2>&1
echo "=== END kernel anchors rc=$? $(date -Is)" | tee -a "$H/run_gate.progress"
echo "GATE_AND_ANCHORS DONE $(date -Is)" | tee -a "$H/run_gate.progress"
