#!/usr/bin/env bash
# zerocost-wiring activation smoke (release.md §3), 2026-09-13.
#
#   LEG 1 default-off (port 8352): plain `vllm serve`, NO zerocost env.
#     /health must turn 200, two chats must return 200, and the engine log must
#     show ZERO zerocost activity (the opt-in contract: unset => untouched).
#   LEG 2 ON (port 8353): `vllm-hust-ext run -- vllm serve` with the
#     org.vllm-hust.zerocost-wiring bundle enabled -> the manager injects
#     VLLM_HUST_FI_PREFILL_OUT=1 + VLLM_HUST_SKIP_COS_SIN=1; the engine log must
#     show the two "zero-cost wiring ①/② is ACTIVE" lines.
#     (cascade/demask are also manager-enabled, so the run gets them too; the
#      capture-size cap below is the release.md / cascade §3.5 quirk fix.)
#
# Card 7 only; run under flock -x -w 7200 /tmp/npu-card7.lock (run_smoke.sh).
set -uo pipefail
EVID=$(cd "$(dirname "$0")/.." && pwd)
MODEL=/data/shared_models/Qwen--Qwen2.5-14B-Instruct
export ASCEND_RT_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1
export VLLM_DISABLE_COMPILE_CACHE=1
mkdir -p "$EVID/raw"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
hbm() { npu-smi info -t usages -i 7 2>/dev/null | grep 'HBM Usage Rate'; }

wait_healthy() {  # wait_healthy <tag> <port> <pid>
  local tag="$1" port="$2" pid="$3" i
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && { say "serve[$tag] healthy after $i polls"; return 0; }
    kill -0 "$pid" 2>/dev/null || { say "serve[$tag] DIED"; tail -40 "$EVID/raw/serve_${tag}.log"; return 1; }
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
    say "chat[$tag/$n] $(head -c 220 "$EVID/raw/chat_${tag}_${n}.json" | tr -d '\n')"
  done
}

stop() {  # stop <tag>
  local tag="$1"
  local pidfile="/tmp/zc_serve_${tag}.pid"
  [ -f "$pidfile" ] || return 0
  local pid pgid
  pid=$(cat "$pidfile")
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  pgid=${pgid:-$pid}
  say "stopping serve[$tag] pid=$pid pgid=$pgid"
  kill -TERM -"$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  local _
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -KILL -"$pgid" 2>/dev/null || true
  rm -f "$pidfile"
  sleep 3
  local left
  left=$(ps -eo pid,pgid,comm | awk -v g="$pgid" '$2 == g {print $1" "$3}' | tr '\n' ' ')
  say "serve[$tag] stopped; our residual children: ${left:-<none>}"
}

say "HBM(before): $(hbm)"

say "=== LEG 1: default-off (plain vllm serve, no zerocost env) ==="
export VLLM_CACHE_ROOT="/tmp/zc_cache_off"; rm -rf "$VLLM_CACHE_ROOT"
cd /tmp || exit 1
unset VLLM_HUST_FI_PREFILL_OUT VLLM_HUST_SKIP_COS_SIN 2>/dev/null || true
setsid nohup vllm serve "$MODEL" --gpu-memory-utilization 0.85 --max-model-len 4096 \
  --served-model-name qwen14b --port 8352 \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  > "$EVID/raw/serve_off.log" 2>&1 < /dev/null &
OFF_PID=$!
echo "$OFF_PID" > /tmp/zc_serve_off.pid
say "serve[off] pid=$OFF_PID port=8352 (stock)"
if wait_healthy off 8352 "$OFF_PID"; then chats off 8352; fi
stop off
say "HBM(after off): $(hbm)"

say "=== LEG 2: zerocost ON (vllm-hust-ext run, manager injects the keys) ==="
export VLLM_CACHE_ROOT="/tmp/zc_cache_on"; rm -rf "$VLLM_CACHE_ROOT"
cd /tmp || exit 1
setsid nohup vllm-hust-ext run -- vllm serve "$MODEL" --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --served-model-name qwen14b --port 8353 \
  --compilation-config '{"cudagraph_capture_sizes":[32,64,128]}' \
  > "$EVID/raw/serve_on.log" 2>&1 < /dev/null &
ON_PID=$!
echo "$ON_PID" > /tmp/zc_serve_on.pid
say "serve[on] pid=$ON_PID port=8353 (vllm-hust-ext run)"
if wait_healthy on 8353 "$ON_PID"; then chats on 8353; fi
stop on
say "HBM(after on): $(hbm)"

say "=== evidence greps ==="
{
  echo "## OFF leg (stock): zerocost traces (expect empty)"
  grep -nE "zero-cost wiring|zerocost|VLLM_HUST_FI_PREFILL_OUT|VLLM_HUST_SKIP_COS_SIN" "$EVID/raw/serve_off.log" || echo "  <none>"
  echo "## OFF leg: Traceback/ERROR (expect empty)"
  grep -nE "Traceback|ERROR" "$EVID/raw/serve_off.log" || echo "  <none>"
  echo "## ON leg: zerocost ACTIVE lines"
  grep -nE "zero-cost wiring . is ACTIVE|zero-cost wiring ①|zero-cost wiring ②" "$EVID/raw/serve_on.log" || echo "  <none>"
  echo "## ON leg: injected keys visible in process env line"
  grep -nE "VLLM_HUST_FI_PREFILL_OUT|VLLM_HUST_SKIP_COS_SIN" "$EVID/raw/serve_on.log" || echo "  <none>"
  echo "## ON leg: Traceback (expect empty)"
  grep -nE "Traceback" "$EVID/raw/serve_on.log" || echo "  <none>"
} | tee "$EVID/raw/evidence_greps.txt"
say "=== smoke done ==="
