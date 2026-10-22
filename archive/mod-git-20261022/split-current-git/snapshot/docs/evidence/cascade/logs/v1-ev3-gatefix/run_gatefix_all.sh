#!/bin/bash
# G6 gatefix verification: adaptive-gate legs at B=32 x2 and B=64 x2 on the
# fixed build (plugin c4121e2, small-prefix bench margin guard).
# Same protocol as logs/v1-ev3-rerun/run_gate{,64}_rerun.sh; kernel anchors
# omitted (unaffected by the gate verdict change).  Single sequential wrapper
# to avoid card contention.
set -u
cd /tmp
G=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev3-gatefix
export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_DISABLE_COMPILE_CACHE=1
CANN_PP="${PYTHONPATH:-}"
export PYTHONPATH="$CANN_PP"
export VLLM_ASCEND_CASCADE_TRACE=1
export VLLM_ASCEND_CASCADE_ADAPTIVE_GATE=1

for tag in r1 r2; do
  echo "=== BEGIN gate32 tag=$tag $(date -Is)" | tee -a "$G/gatefix.progress"
  BATCH=32 python3 "$G/cascade_sweep_plug.py.frozen" on "gate32_$tag" \
    > "$G/sweep_on_gate32_$tag.log" 2>&1
  echo "=== END gate32 tag=$tag rc=$? $(date -Is)" | tee -a "$G/gatefix.progress"
done

for tag in r1 r2; do
  echo "=== BEGIN gate64 tag=$tag $(date -Is)" | tee -a "$G/gatefix.progress"
  BATCH=64 python3 "$G/cascade_sweep_plug.py.frozen" on "gate64_$tag" \
    > "$G/sweep_on_gate64_$tag.log" 2>&1
  echo "=== END gate64 tag=$tag rc=$? $(date -Is)" | tee -a "$G/gatefix.progress"
done

echo "GATEFIX ALL DONE $(date -Is)" | tee -a "$G/gatefix.progress"
