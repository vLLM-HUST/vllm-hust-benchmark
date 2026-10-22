#!/bin/bash
# Temporal control for the B=64 cells.  The ungated ON b64 legs (18:37-18:49)
# measured 11.20/11.20/11.28 s while the ADAPTIVE_GATE b64 legs (19:20-19:26)
# measured 10.75/10.78 s on EVERY cell, even though the gate decided "on" for
# N=64 P=4096 (i.e. no fallback).  Re-run an interleaved OFF/ON b64 pair now so
# the earlier numbers can be checked against the current machine state.
set -u
cd /tmp
H=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev3-rerun
export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_DISABLE_COMPILE_CACHE=1
leg() {  # leg <mode> <tag> <batch> <log>
  local mode="$1" tag="$2" batch="$3" log="$4"
  echo "=== BEGIN mode=$mode tag=$tag batch=$batch $(date -Is)" \
    | tee -a "$H/run_b64_control.progress"
  if [ "$mode" = "on" ]; then export VLLM_ASCEND_CASCADE_TRACE=1; else unset VLLM_ASCEND_CASCADE_TRACE; fi
  BATCH="$batch" python3 "$H/cascade_sweep_plug.py.frozen" "$mode" "$tag" > "$H/$log" 2>&1
  echo "=== END mode=$mode tag=$tag rc=$? $(date -Is)" \
    | tee -a "$H/run_b64_control.progress"
}
leg off off64_d 64 sweep_off_b64_d.log
leg on  on64_d  64 sweep_on_graph_b64_d.log
leg off off64_e 64 sweep_off_b64_e.log
leg on  on64_e  64 sweep_on_graph_b64_e.log
echo "B64 CONTROL DONE $(date -Is)" | tee -a "$H/run_b64_control.progress"
