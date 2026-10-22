#!/bin/bash
# Evidence #2 re-run (correctness alignment) on the FIXED plugin build (ddc0120),
# UNSHIMMED.  Harness = ev2_run.py / ev2_compare.py (reused verbatim from the
# prior evidence-#2 run; no harness edit).
#
# Symmetric baseline-quirk env on EVERY leg (documented, see report):
#   VLLM_DISABLE_COMPILE_CACHE=1   (shared AOT cache hard-crashes this baseline)
#   VLLM_CACHE_ROOT=/tmp/ev2rerun_cache_<tag>   (fresh per leg)
# Card 6 only.  CWD=/tmp (never /vllm-workspace).
set -x
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2-rerun
RUN=/vllm-workspace/flashinfer-migration/cascade-evidence/ev2_run.py
mkdir -p "$OUT/raw_dumps"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 VLLM_DISABLE_COMPILE_CACHE=1

npu-smi info -t usages -i 6 | grep -i 'HBM Usage Rate' | tee "$OUT/card6_before.txt"

run() {  # mode tag batch sent gen eager graph ignore_eos trace extra...
  local mode=$1 tag=$2 batch=$3 sent=$4 gen=$5 eager=$6 graph=$7 ie=$8 trace=$9
  shift 9
  rm -rf "/tmp/ev2rerun_cache_$tag"
  echo "== $tag ==" >> "$OUT/card6_perleg.txt"
  npu-smi info -t usages -i 6 | grep -i 'HBM Usage Rate' >> "$OUT/card6_perleg.txt"
  timeout 2400 env VLLM_CACHE_ROOT="/tmp/ev2rerun_cache_$tag" \
      VLLM_ASCEND_CASCADE_TRACE="$trace" \
      EV2_BATCH=$batch EV2_SENT=$sent EV2_GEN=$gen EV2_MAXLEN=8192 \
      EV2_EAGER=$eager EV2_GRAPH=$graph EV2_IGNORE_EOS=$ie "$@" \
      python -u "$RUN" "$mode" "$tag" > "$OUT/$tag.log" 2>&1
  local rc=$?
  cp -f "/tmp/ev2_${mode}_${tag}.json" "$OUT/raw_dumps/" 2>/dev/null
  echo "rc=$rc tag=$tag"
  tail -1 "$OUT/$tag.log"
  grep -m1 'cascade plugin loaded' "$OUT/$tag.log"
  echo "cascade-active=$(grep -c 'cascade-active' "$OUT/$tag.log")"
  echo "capture-SUCCESS=$(grep -c 'capture body SUCCESS' "$OUT/$tag.log")"
  echo "replay-key-hit=$(grep -c 'replay update: cascade key hit' "$OUT/$tag.log")"
  echo "twin-miss=$(grep -c 'cascade twin missing for descriptor' "$OUT/$tag.log")"
  echo "drift-warn=$(grep -c 'delegating to the original implementation' "$OUT/$tag.log")"
  echo "typeerror=$(grep -c 'TypeError' "$OUT/$tag.log")"
}

echo "##### LANE G (graph, UNSHIMMED, main acceptance): B=64 shared~4356 gen128 ignore_eos #####"
run off        gB64_ie_off        64 420 128 0 1 1 0
run on         gB64_ie_on         64 420 128 0 1 1 1
run on_run2    gB64_ie_on_run2    64 420 128 0 1 1 0
run off        gB64_ie_off2       64 420 128 0 1 1 0
run on_fp32    gB64_ie_on_fp32    64 420 128 0 1 1 0

echo "##### LANE G: B=32 #####"
run off        gB32_ie_off        32 420 128 0 1 1 0
run on         gB32_ie_on         32 420 128 0 1 1 1

echo "##### LANE E (eager): B=64 shared~4356 gen128 ignore_eos #####"
run off        eB64_ie_off        64 420 128 1 0 1 0
run on         eB64_ie_on         64 420 128 1 0 1 1
run on_run2    eB64_ie_on_run2    64 420 128 1 0 1 0
run off        eB64_ie_off2       64 420 128 1 0 1 0
run on_fp32    eB64_ie_on_fp32    64 420 128 1 0 1 0
run on         eB64_ie_on_torchmerge 64 420 128 1 0 1 0 EV2_TORCHMERGE=1

echo "##### LANE E: literal C3 config (no ignore_eos) #####"
run off        eB64_lit_off       64 420 128 1 0 0 0
run on         eB64_lit_on        64 420 128 1 0 0 1

echo "##### LANE E: B=32 #####"
run off        eB32_ie_off        32 420 128 1 0 1 0
run on         eB32_ie_on         32 420 128 1 0 1 1

echo "##### MATRIX DONE #####"
npu-smi info -t usages -i 6 | grep -i 'HBM Usage Rate' | tee "$OUT/card6_after.txt"
ls -la "$OUT/raw_dumps"
