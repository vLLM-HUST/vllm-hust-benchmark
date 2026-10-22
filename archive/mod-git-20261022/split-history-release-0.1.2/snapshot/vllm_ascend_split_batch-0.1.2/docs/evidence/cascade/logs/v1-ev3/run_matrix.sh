#!/bin/bash
# ev3 definitive matrix on card 7.
#
# Legs:
#   off        : frozen upstream harness, cascade envs unset        (graph mode)
#   on         : frozen upstream harness, decode+graph envs set     (graph mode)
#   on_shim    : same as `on` but with the harness-side capture-signature shim
#   off_eager  : ev3 harness copy, cascade envs unset, enforce_eager
#   on_eager   : ev3 harness copy, decode env set, graph env unset, enforce_eager
#
# VLLM_DISABLE_COMPILE_CACHE=1 on every leg: the shared torch.compile AOT cache
# in /root/.cache/vllm produced a hard engine-init failure on a cache HIT
# (`TypeError: 'NoneType' object is not callable`, sweep_off_b64_r2.log), which
# is unrelated to cascade.  Disabling it makes every leg compile fresh and
# removes the cache-hit/miss asymmetry between OFF and ON.
set -u
cd /tmp
H=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev3
FROZEN=$H/cascade_sweep_plug.py.frozen
EV3=$H/cascade_sweep_ev3.py
SHIM=$H/shim

export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_DISABLE_COMPILE_CACHE=1
# CANN's set_env.sh puts acl/tbe on PYTHONPATH; never clobber it.
CANN_PP="${PYTHONPATH:-}"

leg() {  # leg <harness> <mode> <tag> <batch> <extra_pythonpath> <log>
  local harness="$1" mode="$2" tag="$3" batch="$4" pp="$5" log="$6"
  echo "=== BEGIN mode=$mode tag=$tag batch=$batch pp=${pp:-none} $(date -Is)" \
    | tee -a "$H/run_matrix.progress"
  if [ -n "$pp" ]; then export PYTHONPATH="$pp:$CANN_PP"; else export PYTHONPATH="$CANN_PP"; fi
  if [ "$mode" = "on" ] || [ "$mode" = "on_eager" ]; then
    export VLLM_ASCEND_CASCADE_TRACE=1
  else
    unset VLLM_ASCEND_CASCADE_TRACE
  fi
  BATCH="$batch" python3 "$harness" "$mode" "$tag" > "$H/$log" 2>&1
  echo "=== END mode=$mode tag=$tag rc=$? $(date -Is)" | tee -a "$H/run_matrix.progress"
}

# --- B=64, three prefixes in one process: OFF x2, ON(shim) x2 -------------
leg "$FROZEN" off       off64_a   64 ""      sweep_off_b64_a.log
leg "$FROZEN" off       off64_b   64 ""      sweep_off_b64_b.log
leg "$FROZEN" on        ong64_a   64 "$SHIM" sweep_on_graph_shim_b64_a.log
leg "$FROZEN" on        ong64_b   64 "$SHIM" sweep_on_graph_shim_b64_b.log
# --- B=32 (the documented loss cell): OFF x2, ON(shim) x1 ------------------
leg "$FROZEN" off       off32_a   32 ""      sweep_off_b32_a.log
leg "$FROZEN" off       off32_b   32 ""      sweep_off_b32_b.log
leg "$FROZEN" on        ong32_a   32 "$SHIM" sweep_on_graph_shim_b32_a.log
# --- eager lane (no capture patch involved): OFF x1, ON x1 -----------------
leg "$EV3"    off_eager offe64_a  64 ""      sweep_off_eager_b64_a.log
leg "$EV3"    on_eager  one64_a   64 ""      sweep_on_eager_b64_a.log
echo "MATRIX DONE $(date -Is)" | tee -a "$H/run_matrix.progress"
