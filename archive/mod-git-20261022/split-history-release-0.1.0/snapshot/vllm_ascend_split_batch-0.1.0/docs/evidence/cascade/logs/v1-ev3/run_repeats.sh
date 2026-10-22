#!/bin/bash
# ev3 follow-up 3: extra repeats for the cells closest to the noise band
# (p800_b32 at -2.4% and p420_b64 at +3.5%) plus one more B=64 pair.
set -u
cd /tmp
H=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev3
FROZEN=$H/cascade_sweep_plug.py.frozen
export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_DISABLE_COMPILE_CACHE=1
CANN_PP="${PYTHONPATH:-}"

leg() {  # leg <mode> <tag> <batch> <pp> <log>
  local mode="$1" tag="$2" batch="$3" pp="$4" log="$5"
  echo "=== BEGIN mode=$mode tag=$tag batch=$batch $(date -Is)" \
    | tee -a "$H/run_repeats.progress"
  if [ -n "$pp" ]; then export PYTHONPATH="$pp:$CANN_PP"; else export PYTHONPATH="$CANN_PP"; fi
  if [ "$mode" = "on" ]; then export VLLM_ASCEND_CASCADE_TRACE=1; else unset VLLM_ASCEND_CASCADE_TRACE; fi
  BATCH="$batch" python3 "$FROZEN" "$mode" "$tag" > "$H/$log" 2>&1
  echo "=== END mode=$mode tag=$tag rc=$? $(date -Is)" | tee -a "$H/run_repeats.progress"
}

leg off off32_c 32 ""      sweep_off_b32_c.log
leg on  ong32_c 32 "$H/shim" sweep_on_graph_shim_b32_c.log
leg off off64_c 64 ""      sweep_off_b64_c.log
leg on  ong64_e 64 "$H/shim" sweep_on_graph_shim_b64_e.log
echo "REPEATS DONE $(date -Is)" | tee -a "$H/run_repeats.progress"
