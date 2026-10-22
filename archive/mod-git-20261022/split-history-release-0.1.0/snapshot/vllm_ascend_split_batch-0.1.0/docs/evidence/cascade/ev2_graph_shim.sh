#!/bin/bash
# DIAGNOSTIC lane: graph twin with the three known signature drifts patched by
# the harness shim (ev2_shim.py).  Answers "would the twin work if fixed?".
# Results are reported in a clearly labeled diagnostics section, never as the
# acceptance verdict.
set -x
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2
RUN=/vllm-workspace/flashinfer-migration/cascade-evidence/ev2_run.py
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1

leg() {  # mode tag batch sent gen
  local mode=$1 tag=$2 batch=$3 sent=$4 gen=$5
  rm -rf "/tmp/ev2_cache_$tag"
  env VLLM_CACHE_ROOT="/tmp/ev2_cache_$tag" EV2_SHIM_SIG=1 \
      EV2_BATCH=$batch EV2_SENT=$sent EV2_GEN=$gen EV2_MAXLEN=8192 \
      EV2_EAGER=0 EV2_GRAPH=1 EV2_CAPTURE=32,64 EV2_IGNORE_EOS=1 \
      python -u "$RUN" "$mode" "$tag" > "$OUT/$tag.log" 2>&1
  echo "rc=$? tag=$tag"
  grep -m1 "cascade plugin loaded" "$OUT/$tag.log"
  grep -m1 "cascade aclgraph captured\|cascade-active" "$OUT/$tag.log"
  grep -m1 "TypeError\|507011\|Error" "$OUT/$tag.log"
  tail -2 "$OUT/$tag.log"
}

echo "##### shim smoke (B=4 gen=8) #####"
leg on shim_smoke_on 4 420 8
echo "##### shim main (B=64 shared~4356 gen128) #####"
leg off shimB64_ie_off 64 420 128
leg on  shimB64_ie_on  64 420 128
echo "##### DONE #####"
ls -la /tmp/ev2_*_shim*.json 2>/dev/null
