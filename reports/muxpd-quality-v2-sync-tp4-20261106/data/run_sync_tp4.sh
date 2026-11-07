#!/usr/bin/env bash
# QUALITY-V2-SYNC-TP4.md part A (frozen): order N1 R1 G1 R2 N2 N3 R3 G2 R4 N4, devices 0-3, HCCL_DETERMINISTIC=true.
# A run that fails before any output is relaunched once with suffix -retry; the failed attempt is kept.
set -o pipefail
cd /root/workspace/muxpd-eval/sync-v2
[ "$(git -C /root/workspace/vllm-ascend-MuxPD-hust rev-parse HEAD)" = 43fcaf4279640bcc086119b0671cb7fd4228836a ] || { echo WRONG_HEAD; exit 1; }
source /root/workspace/runtime/env.sh >/dev/null 2>&1
export HCCL_DETERMINISTIC=true
unset VLLM_BATCH_INVARIANT
cfg() { case $1 in native) echo root/native-quality-fusion-engine-args.json;; raw) echo root/raw-fusion-engine-args.json;; group) echo root/group-fusion-engine-args.json;; esac; }
one() {
  label=$1; role=$2
  python root/run_quality_controlled_v3.py --label $label --role $role --source /root/workspace/vllm-ascend-MuxPD-hust \
    --config $(cfg $role) --frozen quality/frozen-v2 --program quality/run_quality.py > logs/$label.controller.log 2>&1
  rc=$?; echo "$(date -u +%FT%TZ) $label rc=$rc"
  return $rc
}
mkdir -p logs runs
for spec in sv2-n1:native sv2-r1:raw sv2-g1:group sv2-r2:raw sv2-n2:native sv2-n3:native sv2-r3:raw sv2-g2:group sv2-r4:raw sv2-n4:native; do
  label=${spec%%:*}; role=${spec##*:}
  if ! one $label $role; then
    out=$(python3 -c "import json,sys;d=json.load(open('runs/$label/results/run.json'));print(len(d['outputs']))" 2>/dev/null || echo 0)
    if [ "$out" = 0 ]; then echo "$label produced no output; relaunch once"; one $label-retry $role; fi
  fi
done
echo SYNC_TP4_DONE
