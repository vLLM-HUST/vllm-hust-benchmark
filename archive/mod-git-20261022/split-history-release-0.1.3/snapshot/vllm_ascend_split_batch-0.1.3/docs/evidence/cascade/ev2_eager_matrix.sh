#!/bin/bash
# Lane A (eager) correctness matrix with ignore_eos=1 (W5 harness discipline:
# "ignore_eos is MANDATORY: without it generation ends after ~8 tokens").
# Shapes: C3 main shape B=64 and the B=32 regression shape, both shared~4356.
set -x
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2
RUN=/vllm-workspace/flashinfer-migration/cascade-evidence/ev2_run.py
mkdir -p "$OUT"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1

run() {  # mode tag batch sent gen eager graph extra...
  local mode=$1 tag=$2 batch=$3 sent=$4 gen=$5 eager=$6 graph=$7
  shift 7
  env EV2_BATCH=$batch EV2_SENT=$sent EV2_GEN=$gen EV2_MAXLEN=8192 \
      EV2_EAGER=$eager EV2_GRAPH=$graph EV2_IGNORE_EOS=1 "$@" \
      python -u "$RUN" "$mode" "$tag" > "$OUT/$tag.log" 2>&1
  echo "rc=$? tag=$tag"
  tail -2 "$OUT/$tag.log"
}

echo "##### S1: B=64 shared~4356 gen128 ignore_eos eager #####"
run off        eagerB64_ie_off       64 420 128 1 0
run on         eagerB64_ie_on        64 420 128 1 0
run on_run2    eagerB64_ie_on_run2   64 420 128 1 0
run on_fp32    eagerB64_ie_on_fp32   64 420 128 1 0

echo "##### S2: B=32 shared~4356 gen128 ignore_eos eager #####"
run off        eagerB32_ie_off       32 420 128 1 0
run on         eagerB32_ie_on        32 420 128 1 0

echo "##### DONE #####"
ls -la /tmp/ev2_*_eagerB*_ie_*.json
