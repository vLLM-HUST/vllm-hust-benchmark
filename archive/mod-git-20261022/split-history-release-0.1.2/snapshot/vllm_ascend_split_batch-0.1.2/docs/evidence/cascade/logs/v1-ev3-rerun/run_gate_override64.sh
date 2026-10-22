#!/bin/bash
# Diagnostic: isolate WHY the ADAPTIVE_GATE b64 leg is ~0.4 s/cell faster than
# the gate-unset b64 leg even though the gate decides "on" for N=64 P=4096
# (no fallback).  GATE_OVERRIDE=on forces every bucket on AND skips the bench
# subprocess (cascade_gate.py:163), so:
#   gate unset            -> no bench, no gate consumption code   (~11.2 s)
#   GATE_OVERRIDE=on      -> no bench, gate consumption forced on (?)
#   ADAPTIVE_GATE=1       -> bench subprocess + consumption        (~10.78 s)
# If the override leg lands at ~11.2 s the bench subprocess is the cause;
# if it lands at ~10.78 s the gate consumption code path is the cause.
set -u
cd /tmp
H=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev3-rerun
export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ASCEND_CASCADE_TRACE=1
export VLLM_ASCEND_CASCADE_ADAPTIVE_GATE=1
export VLLM_ASCEND_CASCADE_GATE_OVERRIDE=on
for tag in ovr64_r1 ovr64_r2; do
  echo "=== BEGIN ovr64 tag=$tag $(date -Is)" | tee -a "$H/run_gate_override64.progress"
  BATCH=64 python3 "$H/cascade_sweep_plug.py.frozen" on "$tag" \
    > "$H/sweep_on_override_b64_${tag##*_}.log" 2>&1
  echo "=== END ovr64 tag=$tag rc=$? $(date -Is)" | tee -a "$H/run_gate_override64.progress"
done
echo "OVERRIDE64 DONE $(date -Is)" | tee -a "$H/run_gate_override64.progress"
