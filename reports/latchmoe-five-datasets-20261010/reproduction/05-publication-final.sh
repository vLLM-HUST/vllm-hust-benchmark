#!/usr/bin/env bash
set -euo pipefail
task_root=/root/latchmoe-five-datasets-20261010
cp "$task_root/input/latchmoe_dataset_pair.py" "$task_root/benchmark-source/src/vllm_hust_benchmark/latchmoe_dataset_pair.py"
source /root/latchmoe-validation-20261008/input/activate-latchmoe.sh
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$task_root/benchmark-source"
PYTHONPATH="$task_root/benchmark-source/src" python -m pytest tests/test_latchmoe_dataset_pair.py tests/test_mmlu_pro_pyramidkv.py tests/test_frontierscience_pyramidkv.py tests/test_pyramidkv_hle_audit.py tests/test_pyramidkv_swe_audit.py tests/test_pyramidkv_terminal_audit.py tests/test_pyramidkv_terminal_plan.py -q 2>&1 | tee "$task_root/logs/05-benchmark-final.log"
{
 date -u +%FT%TZ
 cat /proc/sys/kernel/hostname
 npu-smi info
 sha256sum "$task_root/mod-source/vllm_moe_offload_ascend/moe_offload/autoconfig.py" "$task_root/mod-source/tests/test_autoconfig.py" "$task_root/benchmark-source/src/vllm_hust_benchmark/latchmoe_dataset_pair.py" "$task_root/benchmark-source/tests/test_latchmoe_dataset_pair.py"
 for task_repo in /root/latchmoe-stack/vllm-hust /root/latchmoe-stack/vllm-ascend-hust; do
  git -C "$task_repo" rev-parse HEAD
  git -C "$task_repo" status --porcelain
 done
 /root/latchmoe-env/bin/python -c 'from importlib.metadata import version; print({x:version(x) for x in ("torch","torch-npu","vllm","vllm-ascend","vllm-moe-offload-ascend")})'
 /usr/local/python3.12.13/bin/python -c 'from importlib.metadata import version; print({x:version(x) for x in ("torch","torch-npu","vllm","vllm-ascend")})'
} 2>&1 | tee "$task_root/logs/05-postflight.log"
printf 'PUBLICATION_VERIFICATION_COMPLETE\n'
