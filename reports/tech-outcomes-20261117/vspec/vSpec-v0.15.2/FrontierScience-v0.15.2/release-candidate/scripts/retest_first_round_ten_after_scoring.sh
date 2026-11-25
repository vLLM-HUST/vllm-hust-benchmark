#!/usr/bin/env bash
# One new solver round for the ten previously completed FS/HLE arms.
# Score the original responses first. This script never invokes a judge.
set -euo pipefail

source /usr/local/Ascend/cann-9.1.0/set_env.sh
export VSPEC_IPC_BASE=/root/.vspec-ipc-1311-retest
export HI_PYTHON=/root/recovery/vspec-install-r8-ZoYz6V/venv/bin/python

client_python=/root/recovery/vspec-client-clean-20261107/bin/python
runner=/root/recovery/vspec-download-reinstall-20261108/examples/qwen35_three_datasets/paired.py
work=/root/recovery/issue18-delivery-20261113

for receipt in \
    /root/f1311n2/runs/frontierscience/native/r4/score-receipt.json \
    /root/f1311n4/runs/frontierscience/native/r3/score-receipt.json \
    /root/f1311n6/runs/frontierscience/native/r1/score-receipt.json \
    /root/f1311n8/runs/frontierscience/native/r1/score-receipt.json \
    /root/f1311bridge/runs/frontierscience/dynamic/r1/score-receipt.json \
    /root/h1311n2/runs/hle-verified/native/r1/score-receipt.json \
    /root/h1311n4/runs/hle-verified/native/r1/score-receipt.json \
    /root/h1311n6/runs/hle-verified/native/r1/score-receipt.json \
    /root/h1311n8/runs/hle-verified/native/r2/score-receipt.json \
    /root/h1311hist1/runs/hle-verified/dynamic/r1/score-receipt.json; do
    [[ -f "$receipt" ]] || { echo "First-round score missing: $receipt" >&2; exit 1; }
done

for setup in \
    /root/f1311bridge-retest-20261116/setup.json \
    /root/h1311hist1-retest-20261116/setup.json; do
    [[ -f "$setup" ]] || { echo "Current-card vSpec setup missing: $setup" >&2; exit 1; }
done

run_arm() {
    local dataset="$1" arm="$2" root="$3" plan="$4" selection="$5"
    echo "Retesting $dataset $arm in $root" >&2
    "$client_python" "$runner" run \
        --setup "$root/setup.json" --arm "$arm" --dataset "$dataset" \
        --plan "$work/$plan" --selection "$work/$selection" \
        --tag round2 --startup-timeout 2400 --execute
}

for depth in 2 4 6 8; do
    run_arm frontierscience native "/root/f1311n${depth}" fs-plan160-alt fs-selection160-alt.json
done
run_arm frontierscience dynamic /root/f1311bridge-retest-20261116 fs-plan160-alt fs-selection160-alt.json

for depth in 2 4 6 8; do
    run_arm hle-verified native "/root/h1311n${depth}" hle-plan1811-alt hle-selection200-alt.json
done
run_arm hle-verified dynamic /root/h1311hist1-retest-20261116 hle-plan1811-alt hle-selection200-alt.json
