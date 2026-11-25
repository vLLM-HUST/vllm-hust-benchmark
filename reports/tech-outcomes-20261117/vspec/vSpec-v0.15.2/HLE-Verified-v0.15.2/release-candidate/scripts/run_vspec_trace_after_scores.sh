#!/usr/bin/env bash
# Two independent diagnostic runs with per-decision logging enabled.
# Never use their token/s values as a replacement for the trace-off paired rows.
set -euo pipefail

source /usr/local/Ascend/cann-9.1.0/set_env.sh
export VSPEC_IPC_BASE=/root/.vspec-ipc-1311-trace
export HI_PYTHON=/root/recovery/vspec-install-r8-ZoYz6V/venv/bin/python

client_python=/root/recovery/vspec-client-clean-20261107/bin/python
runner=/root/recovery/vspec-download-reinstall-20261108/examples/qwen35_three_datasets/paired.py
work=/root/recovery/issue18-delivery-20261113

for receipt in \
    /root/f1311n2/runs/frontierscience/native/round2/score-receipt.json \
    /root/f1311n4/runs/frontierscience/native/round2/score-receipt.json \
    /root/f1311n6/runs/frontierscience/native/round2/score-receipt.json \
    /root/f1311n8/runs/frontierscience/native/round2/score-receipt.json \
    /root/f1311bridge-retest-20261116/runs/frontierscience/dynamic/round2/score-receipt.json \
    /root/h1311n2/runs/hle-verified/native/round2/score-receipt.json \
    /root/h1311n4/runs/hle-verified/native/round2/score-receipt.json \
    /root/h1311n6/runs/hle-verified/native/round2/score-receipt.json \
    /root/h1311n8/runs/hle-verified/native/round2/score-receipt.json \
    /root/h1311hist1-retest-20261116/runs/hle-verified/dynamic/round2/score-receipt.json; do
    [[ -f "$receipt" ]] || { echo "Second-round score missing: $receipt" >&2; exit 1; }
done

"$client_python" "$runner" run \
    --setup /root/f1311bridge-trace-20261116/setup.json \
    --arm dynamic --dataset frontierscience \
    --plan "$work/fs-plan160-alt" \
    --selection "$work/fs-selection160-alt.json" \
    --tag trace-r1 --startup-timeout 2400 --execute

"$client_python" "$runner" run \
    --setup /root/h1311hist1-trace-20261116/setup.json \
    --arm dynamic --dataset hle-verified \
    --plan "$work/hle-plan1811-alt" \
    --selection "$work/hle-selection200-alt.json" \
    --tag trace-r1 --startup-timeout 2400 --execute
