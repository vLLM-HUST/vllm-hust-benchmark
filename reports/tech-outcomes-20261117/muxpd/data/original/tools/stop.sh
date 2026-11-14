#!/usr/bin/env bash
# Usage: stop.sh <name>  -- terminates the process group recorded by launch.sh
OUT=/root/workspace/muxpd-eval/runs/$1
pid=$(cat "$OUT/pid" 2>/dev/null) || { echo "no pid for $1"; exit 0; }
kill -TERM -- -"$pid" 2>/dev/null
for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
kill -KILL -- -"$pid" 2>/dev/null
rm -f "$OUT/pid"; echo "stopped $1"
