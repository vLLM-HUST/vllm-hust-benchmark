#!/bin/bash
# ev3 smoke: ON leg, B=64, p420 only. Verifies the harness works on the new
# baseline AND that cascade actually engages (startup marker + [cascade-active]).
set -u
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_ASCEND_CASCADE_TRACE=1
export ONLY=p420_b64
export BATCH=64
LOG=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev3/smoke_on_b64.log
python3 /vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev3/cascade_sweep_plug.py.frozen on smoke_b64 > "$LOG" 2>&1
echo "rc=$?" >> "$LOG"
