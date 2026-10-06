#!/bin/bash
set -euo pipefail

root=/root/issue-214-136-formal-20261007
core="$root/worktrees/vllm-hust-current"
plugin="$root/worktrees/vllm-ascend-engram-fix"
model=/models/Qwen--Qwen2.5-14B-Instruct
fla_wheel="$root/wheels/flash_linear_attention_npu_a2-26.9.1+deva4a7958-py3-none-manylinux_2_34_aarch64.whl"
device=${1:-1}
port=${2:-8111}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out="$root/qualification/issue136-current-main-tp1-static-capability-fallback-$stamp"
server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "$out"
test "$(git -C "$core" rev-parse HEAD)" = c696cc916ef30c7eb57c3e90a9f278e82e6e0bc7
test "$(git -C "$plugin" rev-parse HEAD)" = b400fab66eb2d1cf2e69bca32e807f187d382965
test -z "$(git -C "$core" status --porcelain)"
test -z "$(git -C "$plugin" status --porcelain)"
echo "7c3d3d07419a9cb1a17ca0d683d4d5de82d69c6523ada96396c9d6d0317e583a  $fla_wheel" | sha256sum --check

source /usr/local/Ascend/cann/set_env.sh
export ASCEND_VISIBLE_DEVICES="$device"
export ASCEND_RT_VISIBLE_DEVICES="$device"
export PYTHONPATH="$plugin:$core${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1

npu-smi info > "$out/npu-before.txt"
git -C "$core" rev-parse HEAD > "$out/core-commit.txt"
git -C "$plugin" rev-parse HEAD > "$out/plugin-commit.txt"
printf '%s\n' "$device" > "$out/runtime-device.txt"
printf '%s\n' "$port" > "$out/port.txt"
sha256sum /vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C*.so > "$out/extension.sha256"
sha256sum "$fla_wheel" > "$out/fla-wheel.sha256"

python - "$out/import-provenance.json" <<'PY'
import importlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import torch
import torch_npu
import vllm
import vllm_ascend
import fla_npu.ops.ascendc  # noqa: F401

extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
payload = {
    "schema": "issue136-current-main-import-provenance/v1",
    "vllm": {
        "module": vllm.__file__,
        "module_version": getattr(vllm, "__version__", None),
        "distribution_version": metadata.version("vllm"),
    },
    "vllm_ascend": {
        "module": vllm_ascend.__file__,
        "module_version": getattr(vllm_ascend, "__version__", None),
        "distribution_version": metadata.version("vllm-ascend"),
        "extension": extension.__file__,
    },
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "fla_npu_distribution": metadata.version("flash-linear-attention-npu-a2"),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

setsid python -u -m vllm.entrypoints.openai.api_server \
  --model "$model" \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.6 \
  --max-model-len 32768 \
  --host 127.0.0.1 \
  --port "$port" \
  > "$out/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" > "$out/server-wrapper.pid"

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" > "$out/health.txt" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if (( ready == 0 )); then
  printf 'startup-failed\n' > "$out/STATUS"
  exit 1
fi

curl -fsS --max-time 180 \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"prompt\":\"Qualification: reply OK\",\"temperature\":0,\"max_tokens\":8}" \
  "http://127.0.0.1:$port/v1/completions" > "$out/completion.json"
python - "$out/completion.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if not payload.get("choices"):
    raise SystemExit("qualification response has no choices")
PY

cleanup
server_pid=""
npu-smi info > "$out/npu-after.txt"
printf 'OK\n' > "$out/STATUS"
find "$out" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$out/SHA256SUMS"
printf '%s\n' "$out"
