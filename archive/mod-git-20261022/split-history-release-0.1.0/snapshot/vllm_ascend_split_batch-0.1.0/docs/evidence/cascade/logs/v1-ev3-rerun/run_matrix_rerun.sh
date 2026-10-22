#!/bin/bash
# ev3 RERUN (unshimmed) — card 7, plugin HEAD ddc0120 (D1-D4 fixed).
#
# Reuses logs/v1-ev3/cascade_sweep_plug.py.frozen VERBATIM (sha256
# 3dbcd45630ac3dabece70f36f47cfeefa74fd2eca72696bb55d3912c1a852557) and the
# same cell matrix / env as the shimmed ev3 run.
#
# CHANGE vs run_matrix.sh / run_repeats.sh (stated deliberately):
#   1. NO harness-side shim on the ON legs. PYTHONPATH is left as inherited
#      (CANN acl/tbe entries only). This is the whole point of the rerun.
#   2. OFF/ON repeats are INTERLEAVED (off_a, on_a, off_b, on_b, off_c, on_c)
#      instead of grouped, so a thermal/neighbour drift affects both legs of a
#      pair rather than one group. Per-cell medians are unchanged by ordering.
set -u
cd /tmp
H=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev3-rerun
FROZEN=$H/cascade_sweep_plug.py.frozen
export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_DISABLE_COMPILE_CACHE=1
# CANN's set_env.sh puts acl/tbe on PYTHONPATH; never clobber it, and never add
# the old shim. Record it for the report.
echo "PYTHONPATH=${PYTHONPATH:-}" | tee -a "$H/run_matrix_rerun.progress"

leg() {  # leg <mode> <tag> <batch> <log>
  local mode="$1" tag="$2" batch="$3" log="$4"
  echo "=== BEGIN mode=$mode tag=$tag batch=$batch $(date -Is)" \
    | tee -a "$H/run_matrix_rerun.progress"
  if [ "$mode" = "on" ]; then export VLLM_ASCEND_CASCADE_TRACE=1; else unset VLLM_ASCEND_CASCADE_TRACE; fi
  BATCH="$batch" python3 "$FROZEN" "$mode" "$tag" > "$H/$log" 2>&1
  echo "=== END mode=$mode tag=$tag rc=$? $(date -Is)" \
    | tee -a "$H/run_matrix_rerun.progress"
}

# --- B=64 (interleaved OFF/ON x3) -----------------------------------------
leg off off64_a 64 sweep_off_b64_a.log
leg on  on64_a  64 sweep_on_graph_b64_a.log
leg off off64_b 64 sweep_off_b64_b.log
leg on  on64_b  64 sweep_on_graph_b64_b.log
leg off off64_c 64 sweep_off_b64_c.log
leg on  on64_c  64 sweep_on_graph_b64_c.log
# --- B=32 (the documented loss cell; interleaved OFF/ON x3) ----------------
leg off off32_a 32 sweep_off_b32_a.log
leg on  on32_a  32 sweep_on_graph_b32_a.log
leg off off32_b 32 sweep_off_b32_b.log
leg on  on32_b  32 sweep_on_graph_b32_b.log
leg off off32_c 32 sweep_off_b32_c.log
leg on  on32_c  32 sweep_on_graph_b32_c.log
echo "MATRIX RERUN DONE $(date -Is)" | tee -a "$H/run_matrix_rerun.progress"
