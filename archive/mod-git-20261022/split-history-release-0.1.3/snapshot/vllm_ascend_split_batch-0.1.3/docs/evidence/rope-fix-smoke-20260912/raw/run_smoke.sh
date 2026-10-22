#!/bin/bash
# flock launcher for the P4 smoke (retry only on lock-acquisition timeout).
set -u
EVID="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$EVID/raw/smoke_stage.log"
LOCK=/tmp/w3-npu.lock
while true; do
  ( exec 200>>"$LOCK"; flock -x -w 7200 200 || exit 99; bash "$EVID/raw/rope-smoke.sh" ) >>"$LOG" 2>&1
  rc=$?
  echo "----- smoke EXIT=$rc $(date -u +%FT%TZ) -----" >>"$LOG"
  [ "$rc" -eq 99 ] || break
  sleep 30
done
