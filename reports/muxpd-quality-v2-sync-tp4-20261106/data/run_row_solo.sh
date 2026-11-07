#!/usr/bin/env bash
# QUALITY-ROW-SOLO.md part A then part B (frozen).
set -o pipefail
cd /root/workspace/muxpd-eval/sync-v2
[ "$(git -C /root/workspace/vllm-ascend-MuxPD-hust rev-parse HEAD)" = 43fcaf4279640bcc086119b0671cb7fd4228836a ] || { echo WRONG_HEAD; exit 1; }
[ -z "$(git -C /root/workspace/vllm-ascend-MuxPD-hust status --porcelain)" ] || { echo DIRTY; exit 1; }
sha256sum -c ../quality/QUALITY-ROW-SOLO.sha256 || { echo PLAN_CHANGED; exit 1; }
source /root/workspace/runtime/env.sh >/dev/null 2>&1
export HCCL_DETERMINISTIC=true
unset VLLM_BATCH_INVARIANT
SRC=/root/workspace/vllm-ascend-MuxPD-hust
one() { l=$1; shift; python root/run_quality_controlled_solo.py --label $l --source $SRC --frozen quality/frozen-v2 \
    --program quality/run_quality.py --timeout 9300 --max-steps 1700000 --max-wall-seconds 9000 "$@" \
    > logs/$l.controller.log 2>&1; rc=$?; echo "$(date -u +%FT%TZ) $l rc=$rc"; return $rc; }
nout() { python3 -c "import json;print(len(json.load(open('runs/$1/results/run.json'))['outputs']))" 2>/dev/null || echo 0; }
try() { l=$1; shift; one $l "$@" || { [ "$(nout $l)" = 0 ] && one $l-retry "$@"; }; }
N4="--role native --config root/native-quality-fusion-engine-args.json --devices 0,1,2,3"
try ns4-1 $N4
try ns4-2 $N4
echo PART_A_DONE
try ns2-1 --role native --config root/native-quality-fusion-tp2-engine-args.json --devices 0,1 --tp-override 2
try rw2-1 --role raw --config root/raw-row-fusion-tp2-engine-args.json --devices 0,1 --tp-override 2 --admission batched
echo ROW_SOLO_ALL_DONE
