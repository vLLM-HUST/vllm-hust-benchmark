#!/bin/bash
# ev3 primary matrix: cascade OFF vs ON (graph) on the new baseline, card 7.
# Uses the FROZEN upstream harness copy (sha256 recorded in run_meta.txt).
set -u
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=7
H=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev3
HARNESS=$H/cascade_sweep_plug.py.frozen

run() {  # run <mode> <tag> <batch> <only> <logname>
  local mode="$1" tag="$2" batch="$3" only="$4" log="$5"
  echo "=== BEGIN mode=$mode tag=$tag batch=$batch only=$only $(date -Is)" | tee -a "$H/run_primary.progress"
  BATCH="$batch" ONLY="$only" python3 "$HARNESS" "$mode" "$tag" > "$H/$log" 2>&1
  echo "=== END mode=$mode tag=$tag rc=$? $(date -Is)" | tee -a "$H/run_primary.progress"
}

# --- B=64, prefixes 420/800/1600 : 2 OFF repeats (all three prefixes per process)
run off off_r1 64 "" sweep_off_b64_r1.log
run off off_r2 64 "" sweep_off_b64_r2.log
# --- B=64 ON graph, 2 attempts (expected to fail: host signature drift)
run on on_graph_r1 64 "" sweep_on_graph_b64_r1.log
run on on_graph_r2 64 "" sweep_on_graph_b64_r2.log
# --- B=32 loss cell: 2 OFF repeats
run off off_b32_r1 32 "" sweep_off_b32_r1.log
run off off_b32_r2 32 "" sweep_off_b32_r2.log
# --- B=32 ON graph, 1 attempt
run on on_graph_b32_r1 32 "" sweep_on_graph_b32_r1.log
echo "PRIMARY DONE $(date -Is)" | tee -a "$H/run_primary.progress"
