set -eo pipefail
task_dir=/root/latchmoe-five-datasets-20261010
mkdir -p "$task_dir/input" "$task_dir/logs" "$task_dir/evidence"
exec > >(tee "$task_dir/logs/00-server-preflight.log") 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ'
cat /proc/sys/kernel/hostname
uname -m
npu-smi info
df -h /root /models
free -h
printf '%s\n' MODEL_CONFIGS
find /models -maxdepth 2 -name config.json -print
printf '%s\n' CAPABILITIES
for name in python python3 git curl docker podman harbor jq rg rsync; do
  if command -v "$name"; then :; else printf 'MISSING_COMMAND=%s\n' "$name"; fi
done
for path in /var/run/docker.sock /run/podman/podman.sock /root/vllm-hust-eval-data/pujiang-five /root/latchmoe-env /root/latchmoe-validation-20261008 /root/latchmoe-stack /root/.cache/huggingface/hub; do
  if test -e "$path"; then ls -ld "$path"; else printf 'MISSING_PATH=%s\n' "$path"; fi
done
printf '%s\n' ROOT_TASK_DIRECTORIES
find /root -maxdepth 2 -type d \( -iname '*benchmark*' -o -iname '*eval*' -o -iname '*dataset*' -o -iname '*qwen*' -o -iname '*harbor*' \) -print
printf '%s\n' SAFE_HF_CACHE_INVENTORY
if test -d /root/.cache/huggingface/hub; then find /root/.cache/huggingface/hub -maxdepth 1 -type d -print; fi
printf '%s\n' PROTECTED_PACKAGE_VERSIONS
/usr/local/python3.12.13/bin/python3 - <<'PY'
from importlib import metadata
import json
print(json.dumps({name: metadata.version(name) for name in ('torch', 'torch-npu', 'vllm', 'vllm-ascend')}, indent=2))
PY
printf '%s\n' LATCHMOE_PACKAGE_VERSIONS
/root/latchmoe-env/bin/python - <<'PY'
from importlib import metadata
import json
print(json.dumps({name: metadata.version(name) for name in ('torch', 'torch-npu', 'vllm', 'vllm-ascend', 'vllm-moe-offload-ascend')}, indent=2))
PY
printf '%s\n' LOCKED_HOSTS
git -C /root/latchmoe-stack/vllm-hust rev-parse HEAD
git -C /root/latchmoe-stack/vllm-hust status --porcelain
git -C /root/latchmoe-stack/vllm-ascend-hust rev-parse HEAD
git -C /root/latchmoe-stack/vllm-ascend-hust status --porcelain
printf '%s\n' PREFLIGHT_COMPLETE
