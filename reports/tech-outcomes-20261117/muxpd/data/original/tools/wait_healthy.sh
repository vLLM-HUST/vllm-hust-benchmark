#!/usr/bin/env bash
# Usage: wait_healthy.sh <name> <port> [max_seconds]
LOG=/root/workspace/muxpd-eval/runs/$1/server.log
for _ in $(seq 1 $(( ${3:-900} / 10 ))); do
  [ "$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' http://127.0.0.1:$2/health)" = 200 ] && { echo "$1 healthy"; exit 0; }
  grep -q -E 'initialization failed|Traceback' "$LOG" && { echo "$1 FAILED"; grep -E 'Error|Traceback' "$LOG" | tail -15; exit 1; }
  sleep 10
done
echo "$1 TIMEOUT"; exit 2
