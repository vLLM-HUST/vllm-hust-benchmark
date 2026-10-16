#!/usr/bin/env bash
set -euo pipefail
task_root=/root/latchmoe-five-datasets-20261010
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export VLLM_PLUGINS=ascend
/root/latchmoe-env/bin/python "$task_root/input/02-config-repro.py" 2>&1 | tee "$task_root/logs/02-config-original-repro.log"
set +e
/root/latchmoe-env/bin/python -m pytest "$task_root/input/test_autoconfig.py" -q -k 'nested or malformed_text or flat_config' > "$task_root/logs/02-config-red.log" 2>&1
task_red_exit=$?
set -e
cat "$task_root/logs/02-config-red.log"
if [[ "$task_red_exit" != 1 ]]; then
  printf 'Unexpected RED exit: %s\n' "$task_red_exit"
  exit 1
fi
printf 'EXPECTED_RED_CONFIRMED\n'
