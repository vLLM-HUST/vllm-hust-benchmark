#!/bin/bash
# flock launcher for the zerocost-activation smoke (retry only on lock timeout).
set -u
EVID="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$EVID/raw/smoke_stage.log"
LOCK=/tmp/npu-card7.lock
while true; do
  ( exec 200>>"$LOCK"; flock -x -w 7200 200 || exit 99; bash "$EVID/raw/zerocost-smoke.sh" ) >>"$LOG" 2>&1
  rc=$?
  echo "----- smoke EXIT=$rc $(date -u +%FT%TZ) -----" >>"$LOG"
  [ "$rc" -eq 99 ] || break
  sleep 30
done
