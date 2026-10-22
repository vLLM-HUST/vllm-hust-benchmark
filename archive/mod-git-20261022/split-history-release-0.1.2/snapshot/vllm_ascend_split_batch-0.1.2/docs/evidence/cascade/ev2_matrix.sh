#!/bin/bash
# Evidence #2 (correctness alignment) matrix on the v1 baseline, card 6.
#
# Lane A (eager, the achievable faithful reproduction of the C3 Tier-1 lane):
#   C3 config = B=64, shared ~4356 tok (sent=420), gen=128, det (temp=0),
#   enable_prefix_caching, max_model_len=8192, enforce_eager.
#   legs: off / on(bf16) / on_fp32 / on_run2 (determinism) / on_torchmerge
# Lane B (graph twin, the C3 Tier-0 lane): B=64, cudagraph_capture_sizes=[32,64]
#   legs: off / on   -> expected to expose the host-contract drift.
# Lane C (graph twin, labeled harness shim): same as B/on but with a runtime
#   signature shim for NPUModelRunner._update_full_graph_params_if_needed.
set -x
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2
RUN=/vllm-workspace/flashinfer-migration/cascade-evidence/ev2_run.py
mkdir -p "$OUT"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp

export ASCEND_RT_VISIBLE_DEVICES=6
export HF_HUB_OFFLINE=1

npu-smi info -t usages -i 6 | grep -i 'HBM Usage Rate' | tee "$OUT/card6_before_matrix.txt"

C3="EV2_SENT=420 EV2_BATCH=64 EV2_GEN=128 EV2_MAXLEN=8192"

echo "########## LANE A: EAGER ##########"
for leg in off on on_fp32 on_run2 on_torchmerge; do
  echo "===== A/$leg ====="
  case $leg in
    on_torchmerge) EXTRA="EV2_TORCHMERGE=1"; MODE=on ;;
    *) EXTRA=""; MODE=$leg ;;
  esac
  env $C3 EV2_EAGER=1 EV2_GRAPH=0 $EXTRA \
    python -u "$RUN" "$MODE" "eager_$leg" > "$OUT/eager_$leg.log" 2>&1
  echo "rc=$? leg=$leg"
  tail -2 "$OUT/eager_$leg.log"
done

echo "########## LANE B: GRAPH (twin) ##########"
for leg in off on; do
  echo "===== B/$leg ====="
  env $C3 EV2_EAGER=0 EV2_GRAPH=1 EV2_CAPTURE=32,64 \
    python -u "$RUN" "$leg" "graph_$leg" > "$OUT/graph_$leg.log" 2>&1
  echo "rc=$? leg=$leg"
  tail -2 "$OUT/graph_$leg.log"
done

echo "########## LANE C: GRAPH + labeled signature shim ##########"
for leg in on; do
  echo "===== C/$leg ====="
  env $C3 EV2_EAGER=0 EV2_GRAPH=1 EV2_CAPTURE=32,64 EV2_SHIM_SIG=1 \
    python -u "$RUN" "$leg" "graph_shim_$leg" > "$OUT/graph_shim_$leg.log" 2>&1
  echo "rc=$? leg=$leg"
  tail -2 "$OUT/graph_shim_$leg.log"
done

echo "########## MATRIX DONE ##########"
ls -la /tmp/ev2_*.json
