#!/usr/bin/env bash
# Sequential supplementary runs on card 6:
#   1) 19.22 GiB same-KV W4A8/GDN arm (pairs with existing W8A8 util99 runs)
#   2) near-max long-context: capacity smoke 262,144 + NIAH/longgen at 261K/260.8K
# Run as root inside user8-vllm-ascend; detached with nohup.
set -uo pipefail

ROOT=/data/user8/tech-outcomes-20261117/ascend-quant
LOG=$ROOT/supplementary.log

{
    echo "=== SUPPLEMENTARY START $(date -u +%FT%TZ) ==="
    CARD=6 bash "$ROOT/run_util99_w4a8_card6.sh"
    rc1=$?
    echo "=== 19.22 DONE rc=$rc1 $(date -u +%FT%TZ) ==="

    CARD=6 bash "$ROOT/run_longctx_262144_card6.sh"
    rc2=$?
    echo "=== 262K DONE rc=$rc2 $(date -u +%FT%TZ) ==="

    echo "=== SUPPLEMENTARY END rc1=$rc1 rc2=$rc2 $(date -u +%FT%TZ) ==="
} >> "$LOG" 2>&1
