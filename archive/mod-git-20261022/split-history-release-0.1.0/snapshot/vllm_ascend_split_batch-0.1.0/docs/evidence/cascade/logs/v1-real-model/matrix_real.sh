#!/bin/bash
# Evidence #2 "like-for-like" re-run on the REAL Qwen2.5-14B-Instruct.
# Harness = ev2_run_real.py (verbatim cascade/ev2_run.py; sole deviation = MODEL path).
# ALL legs use the LITERAL historical config: NATURAL EOS, EV2_IGNORE_EOS=0 is NEVER set
# (ev2_run_real.py defaults IGNORE_EOS=0).  No tuning, no ignore_eos.
#
# Card discipline: card 7 only, every NPU-touching command serialized on
# /tmp/w3-npu.lock.  Symmetric baseline-quirk env on EVERY leg:
#   VLLM_DISABLE_COMPILE_CACHE=1 + fresh VLLM_CACHE_ROOT=/tmp/ev2real_cache_<tag>
set -x
OUT=/vllm-workspace/knowledge/evidence/cascade/logs/v1-real-model
RUN=$OUT/ev2_run_real.py
LOCK=/tmp/w3-npu.lock
mkdir -p "$OUT/raw_dumps"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 VLLM_DISABLE_COMPILE_CACHE=1

flock $LOCK npu-smi info -t usages -i 7 | grep -i 'HBM Usage Rate' | tee "$OUT/card7_before.txt"

run() {  # mode tag batch sent gen eager graph ignore_eos trace extra...
  local mode=$1 tag=$2 batch=$3 sent=$4 gen=$5 eager=$6 graph=$7 ie=$8 trace=$9
  shift 9
  rm -rf "/tmp/ev2real_cache_$tag"
  echo "== $tag ==" >> "$OUT/card7_perleg.txt"
  flock $LOCK npu-smi info -t usages -i 7 | grep -i 'HBM Usage Rate' >> "$OUT/card7_perleg.txt"
  flock $LOCK timeout 3600 env VLLM_CACHE_ROOT="/tmp/ev2real_cache_$tag" \
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
  flock $LOCK npu-smi info -t usages -i 7 | grep -i 'HBM Usage Rate' >> "$OUT/card7_perleg.txt"
}

echo "##### LANE G (graph, main acceptance) B=64 shared~4224 gen128 NATURAL EOS #####"
run off     gB64_off     64 420 128 0 1 0 0
run on      gB64_on      64 420 128 0 1 0 1
run on_run2 gB64_on_run2 64 420 128 0 1 0 1
echo "##### LANE G B=32 NATURAL EOS #####"
run off     gB32_off     32 420 128 0 1 0 0
run on      gB32_on      32 420 128 0 1 0 1
echo "##### LANE E (eager, bf16 reference frame) B=64 NATURAL EOS #####"
run off     eB64_off     64 420 128 1 0 0 0
run off     eB64_off2    64 420 128 1 0 0 0
run on      eB64_on      64 420 128 1 0 0 1
echo "##### MATRIX DONE #####"
flock $LOCK npu-smi info -t usages -i 7 | grep -i 'HBM Usage Rate' | tee "$OUT/card7_after.txt"
ls -la "$OUT/raw_dumps"

echo "##### leftover python/vllm processes (yours + others) #####"
ps -ef | grep -E 'python|vllm' | grep -v grep | tee "$OUT/card7_leftover_procs.txt"
