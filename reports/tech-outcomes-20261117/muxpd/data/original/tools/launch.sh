#!/usr/bin/env bash
# Usage: launch.sh <name> <devices> <port> <tp> <script> [extra args...]
# Starts a server detached, records pid/argv/env/git identity under runs/<name>.
set -eo pipefail
NAME=$1; DEV=$2; PORT=$3; TP=$4; SCRIPT=$5; shift 5
OUT=/root/workspace/muxpd-eval/runs/$NAME
mkdir -p "$OUT"
source /root/workspace/runtime/env.sh >/dev/null 2>&1
export ASCEND_RT_VISIBLE_DEVICES=$DEV VLLM_SERVER_PORT=$PORT MUXPD_TP_SIZE=$TP
{
  echo "script=$SCRIPT"; printf 'arg=%s\n' "$@"
  echo "muxpd=$(git -C /root/workspace/vllm-ascend-MuxPD-hust rev-parse HEAD) dirty=$(git -C /root/workspace/vllm-ascend-MuxPD-hust status --porcelain -- src scripts | wc -l)"
  echo "vllm=$(git -C /root/workspace/vllm-hust rev-parse HEAD)"
  echo "ascend=$(git -C /root/workspace/vllm-ascend-hust rev-parse HEAD)"
  env | grep -E '^(ASCEND_RT_VISIBLE_DEVICES|VLLM_|MUXPD_|HCCL_|OMP_)' | sort
} > "$OUT/identity.txt"
nohup setsid "$SCRIPT" "$@" > "$OUT/server.log" 2>&1 < /dev/null &
echo $! > "$OUT/pid"
echo "$NAME pid=$! port=$PORT dev=$DEV"
