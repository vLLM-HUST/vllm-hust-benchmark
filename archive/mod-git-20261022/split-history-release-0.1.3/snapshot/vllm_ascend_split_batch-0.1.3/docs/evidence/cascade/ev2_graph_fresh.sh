#!/bin/bash
# Graph lane with a FRESH VLLM_CACHE_ROOT per leg (the shared /root/.cache/vllm
# AOT entry proved unstable across runs: graph_off succeeded at 16:07, then all
# graph runs at 16:32-16:35 died with "'NoneType' object is not callable" while
# loading the same cache hash).
set -x
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2
RUN=/vllm-workspace/flashinfer-migration/cascade-evidence/ev2_run.py
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1

leg() {  # mode tag
  local mode=$1 tag=$2
  rm -rf "/tmp/ev2_cache_$tag"
  env VLLM_CACHE_ROOT="/tmp/ev2_cache_$tag" \
      EV2_BATCH=64 EV2_SENT=420 EV2_GEN=128 EV2_MAXLEN=8192 \
      EV2_EAGER=0 EV2_GRAPH=1 EV2_CAPTURE=32,64 EV2_IGNORE_EOS=1 \
      python -u "$RUN" "$mode" "$tag" > "$OUT/$tag.log" 2>&1
  echo "rc=$? tag=$tag"
  tail -2 "$OUT/$tag.log"
}

echo "##### graph lane, fresh cache per leg #####"
leg off graphB64_ie_off
leg on  graphB64_ie_on
echo "##### DONE #####"
ls -la /tmp/ev2_*_graphB64_ie_*.json 2>/dev/null
