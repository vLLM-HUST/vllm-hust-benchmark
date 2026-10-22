#!/bin/bash
# Environment + baseline provenance capture for the real-model evidence-#2 re-run.
OUT=/vllm-workspace/knowledge/evidence/cascade/logs/v1-real-model
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp
{
echo "=== date ==="
date -u
echo "=== pip versions ==="
pip list 2>/dev/null | grep -iE 'ascend-kernel|^torch |^torch_npu|transformers|^vllm |vllm-ascend-split-batch|vllm-hust-ext|triton'
echo "=== plugin git (vllm-ascend-split-batch-hust) ==="
git -C /vllm-workspace/vllm-ascend-split-batch-hust rev-parse HEAD
git -C /vllm-workspace/vllm-ascend-split-batch-hust log -1 --format='%H %ci %s'
echo "dirty files: $(git -C /vllm-workspace/vllm-ascend-split-batch-hust status --porcelain | wc -l)"
git -C /vllm-workspace/vllm-ascend-split-batch-hust status --porcelain
echo "=== vllm-hust git (baseline host) ==="
git -C /vllm-workspace/vllm-hust log -1 --format='%H %ci %s'
echo "=== CANN ==="
echo "ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME"
ls -d /usr/local/ascend91 2>/dev/null
echo "=== model under test ==="
echo "/data/shared_models/Qwen--Qwen2.5-14B-Instruct"
python - <<'PY'
import json
d=json.load(open('/data/shared_models/Qwen--Qwen2.5-14B-Instruct/config.json'))
print("layers=%s hidden=%s heads=%s kv_heads=%s dtype=%s" % (
  d['num_hidden_layers'], d['hidden_size'], d['num_attention_heads'],
  d['num_key_value_heads'], d['torch_dtype']))
PY
echo "=== card7 HBM before run ==="
cat "$OUT/card7_before.txt" 2>/dev/null || echo "(not sampled yet)"
} > "$OUT/env_and_versions.txt" 2>&1
cat "$OUT/env_and_versions.txt"
