#!/bin/bash
# Controls + ablation at the full 128-step decode depth (ignore_eos):
#   C1 eager off rerun            -> off-lane determinism (noise floor)
#   C2 graph off rerun (shim)     -> graph off-lane determinism
#   C3 eager on with torch-merge  -> does the lse_merge kernel contribute?
set -x
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2
RUN=/vllm-workspace/flashinfer-migration/cascade-evidence/ev2_run.py
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1
C3="EV2_BATCH=64 EV2_SENT=420 EV2_GEN=128 EV2_MAXLEN=8192 EV2_IGNORE_EOS=1"

echo "##### C1: eager off rerun #####"
env $C3 EV2_EAGER=1 EV2_GRAPH=0 python -u "$RUN" off eagerB64_ie_off2 \
  > "$OUT/eagerB64_ie_off2.log" 2>&1
echo "rc=$? C1"; tail -1 "$OUT/eagerB64_ie_off2.log"

echo "##### C2: graph off rerun (shim) #####"
rm -rf /tmp/ev2_cache_shimB64_ie_off2
env VLLM_CACHE_ROOT=/tmp/ev2_cache_shimB64_ie_off2 EV2_SHIM_SIG=1 $C3 \
    EV2_EAGER=0 EV2_GRAPH=1 EV2_CAPTURE=32,64 python -u "$RUN" off shimB64_ie_off2 \
  > "$OUT/shimB64_ie_off2.log" 2>&1
echo "rc=$? C2"; tail -1 "$OUT/shimB64_ie_off2.log"

echo "##### C3: eager on + torch-merge #####"
env $C3 EV2_EAGER=1 EV2_GRAPH=0 EV2_TORCHMERGE=1 python -u "$RUN" on eagerB64_ie_on_torchmerge \
  > "$OUT/eagerB64_ie_on_torchmerge.log" 2>&1
echo "rc=$? C3"; tail -1 "$OUT/eagerB64_ie_on_torchmerge.log"
echo "##### DONE #####"
