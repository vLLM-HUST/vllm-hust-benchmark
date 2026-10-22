#!/bin/bash
# ev3 follow-up: B=64 graph ON leg with the FULL 3-fix harness shim.
# The earlier ong64_a/b attempts ran with a 1-fix shim and died on the
# _update_full_graph_params_if_needed / update_graph_params drift.
set -u
cd /tmp
H=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev3
export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ASCEND_CASCADE_TRACE=1
export PYTHONPATH="$H/shim:${PYTHONPATH:-}"
for tag in ong64_c ong64_d; do
  echo "=== BEGIN tag=$tag $(date -Is)" | tee -a "$H/run_on_graph64.progress"
  BATCH=64 python3 "$H/cascade_sweep_plug.py.frozen" on "$tag" \
    > "$H/sweep_on_graph_shim_b64_${tag##*_}.log" 2>&1
  echo "=== END tag=$tag rc=$? $(date -Is)" | tee -a "$H/run_on_graph64.progress"
done
# B=32 second repeat for spread
echo "=== BEGIN tag=ong32_b $(date -Is)" | tee -a "$H/run_on_graph64.progress"
BATCH=32 python3 "$H/cascade_sweep_plug.py.frozen" on ong32_b \
  > "$H/sweep_on_graph_shim_b32_b.log" 2>&1
echo "=== END tag=ong32_b rc=$? $(date -Is)" | tee -a "$H/run_on_graph64.progress"
echo "ON_GRAPH64 DONE $(date -Is)" | tee -a "$H/run_on_graph64.progress"
