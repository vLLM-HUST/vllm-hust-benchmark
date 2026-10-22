#!/bin/bash
# Evidence #2 re-run, BATCH 2: GRAPH lane in the LITERAL historical C3 config
# (no ignore_eos -> the stand-in model emits EOS after 3-5 tokens; this is the
# closest reproduction of the historical measurement, whose off leg produced
# out_tokens=6469 for 64 arms on the real 14B).
# Harness ev2_run.py reused verbatim.  Same symmetric baseline env as batch 1.
set -x
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2-rerun
RUN=/vllm-workspace/flashinfer-migration/cascade-evidence/ev2_run.py
mkdir -p "$OUT/raw_dumps"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 VLLM_DISABLE_COMPILE_CACHE=1

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
  echo "twin-miss-warn=$(grep -c 'cascade twin missing for descriptor' "$OUT/$tag.log")"
  echo "drift-warn=$(grep -c 'delegating to the original implementation' "$OUT/$tag.log")"
  echo "typeerror=$(grep -c 'TypeError' "$OUT/$tag.log")"
}

echo "##### LANE G literal C3 config (graph, UNSHIMMED, no ignore_eos) B=64 #####"
run off     gB64_lit_off     64 420 128 0 1 0 0
run on      gB64_lit_on      64 420 128 0 1 0 1
run on_run2 gB64_lit_on_run2 64 420 128 0 1 0 1
echo "##### LANE G literal C3 config B=32 #####"
run off     gB32_lit_off     32 420 128 0 1 0 0
run on      gB32_lit_on      32 420 128 0 1 0 1
echo "##### EXTRA: graph B=64 ignore_eos ON run2 WITH TRACE (per-step proof for the determinism leg) #####"
run on_run2 gB64_ie_on_run2_trace 64 420 128 0 1 1 1
echo "##### BATCH2 DONE #####"
npu-smi info -t usages -i 6 | grep -i 'HBM Usage Rate' | tee "$OUT/card6_after.txt"
ls -la "$OUT/raw_dumps"
