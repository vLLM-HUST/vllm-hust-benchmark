#!/bin/bash
# Evidence #2 smoke: does cascade actually activate on the v1 baseline?
set -x
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2
mkdir -p "$OUT"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp

npu-smi info -t usages -i 6 | grep -i 'HBM Usage Rate' | tee "$OUT/smoke_card6_before.txt"

EV2_SENT=420 EV2_BATCH=4 EV2_GEN=8 EV2_EAGER=1 EV2_MAXLEN=8192 \
ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
  python -u /vllm-workspace/flashinfer-migration/cascade-evidence/ev2_run.py on smoke_eager \
  > "$OUT/smoke_eager_on.log" 2>&1
echo "smoke_eager_rc=$?"
grep -c 'cascade plugin loaded' "$OUT/smoke_eager_on.log"
grep -m3 'cascade plugin loaded' "$OUT/smoke_eager_on.log"
grep -c 'cascade-active' "$OUT/smoke_eager_on.log"
grep -m3 'cascade-active' "$OUT/smoke_eager_on.log"
tail -5 "$OUT/smoke_eager_on.log"
echo "=== DONE ==="
