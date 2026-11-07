#!/usr/bin/env bash
# QUALITY-V2-SYNC-TP4-ROW.md (frozen): sv2-w1, sv2-w2 sequential, devices 0-3, HCCL_DETERMINISTIC=true.
set -o pipefail
cd /root/workspace/muxpd-eval/sync-v2
[ "$(git -C /root/workspace/vllm-ascend-MuxPD-hust rev-parse HEAD)" = 43fcaf4279640bcc086119b0671cb7fd4228836a ] || { echo WRONG_HEAD; exit 1; }
sha256sum -c <(head -2 /root/workspace/muxpd-eval/quality/QUALITY-V2-SYNC-TP4-ROW.sha256 | sed 's#  quality/#  /root/workspace/muxpd-eval/quality/#; s#  sync-v2/#  /root/workspace/muxpd-eval/sync-v2/#') || { echo PLAN_CHANGED; exit 1; }
source /root/workspace/runtime/env.sh >/dev/null 2>&1
export HCCL_DETERMINISTIC=true
unset VLLM_BATCH_INVARIANT
one() {
  python root/run_quality_controlled_v3.py --label $1 --role raw --source /root/workspace/vllm-ascend-MuxPD-hust \
    --config root/raw-row-fusion-engine-args.json --frozen quality/frozen-v2 --program quality/run_quality.py > logs/$1.controller.log 2>&1
  rc=$?; echo "$(date -u +%FT%TZ) $1 rc=$rc"; return $rc
}
for label in sv2-w1 sv2-w2; do
  if ! one $label; then
    out=$(python3 -c "import json;print(len(json.load(open('runs/$label/results/run.json'))['outputs']))" 2>/dev/null || echo 0)
    [ "$out" = 0 ] && { echo "$label produced no output; relaunch once"; one $label-retry; }
  fi
done
echo SYNC_ROW_DONE
