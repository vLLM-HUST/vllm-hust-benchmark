#!/usr/bin/env bash
set -euo pipefail
task_root=/root/latchmoe-five-datasets-20261010
mkdir -p "$task_root/mod-source" "$task_root/benchmark-source"
tar -xf "$task_root/input/mod-base.tar.gz" -C "$task_root/mod-source"
tar -xf "$task_root/input/benchmark-base.tar.gz" -C "$task_root/benchmark-source"
cp "$task_root/input/autoconfig.py" "$task_root/mod-source/vllm_moe_offload_ascend/moe_offload/autoconfig.py"
cp "$task_root/input/test_autoconfig.py" "$task_root/mod-source/tests/test_autoconfig.py"
cp "$task_root/input/latchmoe_dataset_pair.py" "$task_root/benchmark-source/src/vllm_hust_benchmark/"
cp "$task_root/input/test_latchmoe_dataset_pair.py" "$task_root/benchmark-source/tests/"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export VLLM_PLUGINS=ascend
source /root/latchmoe-validation-20261008/input/activate-latchmoe.sh
cd "$task_root/mod-source"
PYTHONPATH="$task_root/mod-source" python -m pytest tests/test_autoconfig.py tests/test_configuration_integration.py tests/test_capabilities.py tests/test_model_registry_v2.py -q 2>&1 | tee "$task_root/logs/03-mod-green.log"
PYTHONPATH="$task_root/mod-source" python "$task_root/input/02-config-repro.py" 2>&1 | tee "$task_root/logs/03-config-candidate-repro.log"
cd "$task_root/benchmark-source"
PYTHONPATH="$task_root/benchmark-source/src" python -m pytest tests/test_latchmoe_dataset_pair.py tests/test_mmlu_pro_pyramidkv.py tests/test_frontierscience_pyramidkv.py tests/test_pyramidkv_hle_audit.py tests/test_pyramidkv_swe_audit.py tests/test_pyramidkv_terminal_audit.py tests/test_pyramidkv_terminal_plan.py -q 2>&1 | tee "$task_root/logs/03-benchmark-green.log"
printf 'CANDIDATE_TESTS_COMPLETE\n'
