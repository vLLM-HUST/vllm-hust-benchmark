#!/usr/bin/env bash
# Isolation leg: plain `vllm serve` + the two zerocost envs (cascade graph OFF).
# This reproduces Y1's validated ON condition -- only cascade_plugin's pass-through
# wrapper is on the class, so zerocost ① must resolve the host body.  Card 7,
# flock-protected by run_smoke.sh's lock.
set -uo pipefail
EVID=$(cd "$(dirname "$0")/.." && pwd)
MODEL=/data/shared_models/Qwen--Qwen2.5-14B-Instruct
export ASCEND_RT_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HUST_FI_PREFILL_OUT=1 VLLM_HUST_SKIP_COS_SIN=1
mkdir -p "$EVID/raw"
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
export VLLM_CACHE_ROOT="/tmp/zc_cache_iso"; rm -rf "$VLLM_CACHE_ROOT"
cd /tmp || exit 1
setsid nohup vllm serve "$MODEL" --gpu-memory-utilization 0.85 --max-model-len 4096 \
  --served-model-name qwen14b --port 8354 \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  > "$EVID/raw/serve_iso.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > /tmp/zc_serve_iso.pid
say "serve[iso] pid=$PID port=8354 (plain serve + zerocost envs, cascade OFF)"
ok=1
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8354/health >/dev/null 2>&1 && { say "serve[iso] healthy after $i polls"; ok=0; break; }
  kill -0 "$PID" 2>/dev/null || { say "serve[iso] DIED"; break; }
  sleep 5
done
if [ "$ok" -eq 0 ]; then
  curl -s http://127.0.0.1:8354/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen14b","messages":[{"role":"user","content":"Reply with the single word: ok1"}],"temperature":0,"max_tokens":16}' \
    > "$EVID/raw/chat_iso_1.json"
  say "chat[iso/1] $(head -c 200 "$EVID/raw/chat_iso_1.json" | tr -d '\n')"
fi
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' '); PGID=${PGID:-$PID}
kill -TERM -"$PGID" 2>/dev/null || true
for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
kill -KILL -"$PGID" 2>/dev/null || true
rm -f /tmp/zc_serve_iso.pid
say "serve[iso] stopped"
grep -nE "zero-cost wiring . is ACTIVE|zero-cost wiring ① refused|cascade plugin loaded" "$EVID/raw/serve_iso.log" || echo "  <none>"
