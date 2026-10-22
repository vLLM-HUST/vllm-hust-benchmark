#!/bin/bash
# Extra: adaptive-gate leg at B=64, unshimmed.  The unshimmed matrix shows
# p420_b64 is now a decisive loss (+6.7%), and the gate micro-bench table says
# N=64 P=4096 -> "on" (cascade faster at the op level).  This leg measures
# whether the W2 gate actually protects the B=64 4k cell e2e.
set -u
cd /tmp
H=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev3-rerun
export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ASCEND_CASCADE_TRACE=1
export VLLM_ASCEND_CASCADE_ADAPTIVE_GATE=1
for tag in gate64_r1 gate64_r2; do
  echo "=== BEGIN gate64 tag=$tag $(date -Is)" | tee -a "$H/run_gate64_rerun.progress"
  BATCH=64 python3 "$H/cascade_sweep_plug.py.frozen" on "$tag" \
    > "$H/sweep_on_gate_b64_${tag##*_}.log" 2>&1
  echo "=== END gate64 tag=$tag rc=$? $(date -Is)" | tee -a "$H/run_gate64_rerun.progress"
done
echo "GATE64 RERUN DONE $(date -Is)" | tee -a "$H/run_gate64_rerun.progress"
