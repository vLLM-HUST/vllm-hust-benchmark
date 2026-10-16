#!/usr/bin/env bash
set -euo pipefail
task_root=/root/latchmoe-five-datasets-20261010
date -u +%FT%TZ | tee "$task_root/logs/01-model-audit-after-restart.log"
cat /proc/sys/kernel/hostname | tee -a "$task_root/logs/01-model-audit-after-restart.log"
/root/latchmoe-env/bin/python "$task_root/input/01-model-audit.py" 2>&1 | tee -a "$task_root/logs/01-model-audit-after-restart.log"
