#!/usr/bin/env bash
# P4 — rope carrier entry-point refresh smoke (2026-09-12).
#
#   1. default-off serve  (port 8453, mml 4096): /health + 2 chats, log must show
#      zero rope-carrier activity and no Traceback
#   2. VLLM_HUST_ROPE_FIX=1 serve (port 8454): /health + 2 chats, log must show the
#      ACTIVE / wrapped / overrides-in-place lines
#
# Card 7 only; run under flock -w 7200 /tmp/w3-npu.lock.
set -uo pipefail
EVID=$(cd "$(dirname "$0")/.." && pwd)
MODEL=/data/shared_models/Qwen--Qwen2.5-14B-Instruct
export ASCEND_RT_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1
export VLLM_DISABLE_COMPILE_CACHE=1
mkdir -p "$EVID/raw"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
hbm() { npu-smi info -t usages -i 7 2>/dev/null | grep 'HBM Usage Rate'; }

serve() {  # serve <tag> <port> <rope_env>
  local tag="$1" port="$2" rope="$3"
  local log="$EVID/raw/serve_${tag}.log"
  export VLLM_CACHE_ROOT="/tmp/p4_cache_${tag}"; rm -rf "$VLLM_CACHE_ROOT"
  if [ -n "$rope" ]; then export VLLM_HUST_ROPE_FIX="$rope"; else unset VLLM_HUST_ROPE_FIX || true; fi
  cd /tmp || return 1
  nohup vllm serve "$MODEL" --gpu-memory-utilization 0.85 --max-model-len 4096 \
    --served-model-name qwen14b --port "$port" \
    --compilation-config '{"cudagraph_mode":"FULL"}' > "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "/tmp/p4_serve_${tag}.pid"
  say "serve[$tag] pid=$pid port=$port VLLM_HUST_ROPE_FIX=${rope:-<unset>}"
  local i
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && { say "serve[$tag] healthy after $i polls"; return 0; }
    kill -0 "$pid" 2>/dev/null || { say "serve[$tag] DIED"; tail -30 "$log"; return 1; }
    sleep 5
  done
  say "serve[$tag] health timeout"; return 1
}

chats() {  # chats <tag> <port>
  local tag="$1" port="$2" n
  for n in 1 2; do
    curl -s "http://127.0.0.1:${port}/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"qwen14b\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok$n\"}],\"temperature\":0,\"max_tokens\":16}" \
      > "$EVID/raw/chat_${tag}_${n}.json"
    say "chat[$tag/$n] $(head -c 200 "$EVID/raw/chat_${tag}_${n}.json" | tr -d '\n')"
  done
}

stop() {  # stop <tag>
  local tag="$1"
  local pidfile="/tmp/p4_serve_${tag}.pid"
  [ -f "$pidfile" ] || return 0
  local pid
  pid=$(cat "$pidfile")
  # descendants of OUR serve pid only (never touch other tenants' engines)
  local kids
  kids=$(ps -eo pid,ppid,comm | awk -v p="$pid" '$2 == p && $3 ~ /^VLLM/ {print $1}')
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -9 "$pid" 2>/dev/null || true
  for k in $kids; do
    kill -0 "$k" 2>/dev/null && { say "reaping our engine pid=$k"; kill -9 "$k" 2>/dev/null || true; }
  done
  rm -f "$pidfile"
  sleep 3
  local left
  left=$(ps -eo pid,ppid,comm | awk -v p="$pid" '$2 == p {print $1" "$3}' | tr '\n' ' ')
  say "serve[$tag] stopped; our residual children: ${left:-<none>}"
}

say "HBM(before): $(hbm)"
say "=== LEG 1: default-off (no VLLM_HUST_ROPE_FIX) ==="
if serve off 8453 ""; then chats off 8453; fi
stop off
say "HBM(after off): $(hbm)"
say "=== LEG 2: VLLM_HUST_ROPE_FIX=1 ==="
if serve on 8454 "1"; then chats on 8454; fi
stop on
say "HBM(after on): $(hbm)"

say "=== evidence greps ==="
{
  echo "## default-off leg: rope traces (expect empty)"
  grep -nE "RoPE|rope-fix|rope_fix" "$EVID/raw/serve_off.log" || echo "  <none>"
  echo "## default-off leg: Traceback/Error (expect empty)"
  grep -nE "Traceback|ERROR" "$EVID/raw/serve_off.log" || echo "  <none>"
  echo "## ON leg: rope carrier lines"
  grep -nE "RoPE variant defect carrier is ACTIVE|is wrapped|rope overrides in place|RoPE fix" "$EVID/raw/serve_on.log" || echo "  <none>"
  echo "## ON leg: Traceback (expect empty)"
  grep -nE "Traceback" "$EVID/raw/serve_on.log" || echo "  <none>"
} | tee "$EVID/raw/evidence_greps.txt"
say "=== smoke done ==="
