#!/usr/bin/env bash
# zerocost-wiring §3 re-run AFTER the _host_func multi-wrapper fix (2026-09-13).
#
#   §3 ladder:  inspect / check / status  ->  enable  ->  run --dry-run
#   LEG OFF (port 8352): plain `vllm serve`, NO zerocost env -> /health + chats,
#     engine log must show ZERO zerocost traces (default-off = stock).
#   LEG ON  (port 8353): `vllm-hust-ext run -- vllm serve` (manager injects the
#     two keys; this workspace co-enables cascade graph -> the double-wrapper
#     path that the previous run failed on).  Engine log must show BOTH
#     "zero-cost wiring ① is ACTIVE" and "② is ACTIVE", zero "refused".
#
# Card 7 only; run under flock -x -w 7200 /tmp/npu-card7.lock.
set -uo pipefail
EVID=$(cd "$(dirname "$0")/.." && pwd)
RAW="$EVID/raw-fix"
MODEL=/data/shared_models/Qwen--Qwen2.5-14B-Instruct
export ASCEND_RT_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1
export VLLM_DISABLE_COMPILE_CACHE=1
mkdir -p "$RAW"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
hbm() { npu-smi info -t usages -i 7 2>/dev/null | grep 'HBM Usage Rate'; }

say "HBM(before): $(hbm)"
{
  echo "## host releases"
  python -c 'import importlib.metadata as m; print("vllm", m.version("vllm")); print("vllm_ascend", m.version("vllm-ascend")); print("vllm-ascend-split-batch", m.version("vllm-ascend-split-batch"))'
} | tee "$RAW/meta.txt"

say "=== §3 ladder (flipped manifest: active) ==="
cd /tmp || exit 1
vllm-hust-ext extension inspect org.vllm-hust.zerocost-wiring > "$RAW/ext_inspect.txt" 2>&1
vllm-hust-ext extension check   org.vllm-hust.zerocost-wiring > "$RAW/ext_check.txt" 2>&1
vllm-hust-ext extension status  org.vllm-hust.zerocost-wiring > "$RAW/ext_status_pre.txt" 2>&1
grep -E '"activation_ready"|"activation_blocker"|"status"' "$RAW/ext_inspect.txt" || true
grep -E '"states"' -A6 "$RAW/ext_check.txt" || true
vllm-hust-ext extension enable  org.vllm-hust.zerocost-wiring > "$RAW/ext_enable.txt" 2>&1
echo "enable rc=$?" | tee -a "$RAW/ext_enable.txt"
vllm-hust-ext extension status  org.vllm-hust.zerocost-wiring > "$RAW/ext_status_post.txt" 2>&1
vllm-hust-ext extension list > "$RAW/ext_list_enabled.txt" 2>&1
cat "$RAW/ext_list_enabled.txt"
vllm-hust-ext run --dry-run -- vllm serve "$MODEL" --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --served-model-name qwen14b --port 8353 \
  --compilation-config '{"cudagraph_capture_sizes":[32,64,128]}' \
  > "$RAW/ext_dryrun.txt" 2>&1
echo "--- run --dry-run injection preview ---"; cat "$RAW/ext_dryrun.txt"

wait_healthy() {  # wait_healthy <tag> <port> <pid>
  local tag="$1" port="$2" pid="$3" i
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && { say "serve[$tag] healthy after $i polls"; return 0; }
    kill -0 "$pid" 2>/dev/null || { say "serve[$tag] DIED"; tail -40 "$RAW/serve_${tag}.log"; return 1; }
    sleep 5
  done
  say "serve[$tag] health timeout"; return 1
}

chats() {  # chats <tag> <port> <model>
  local tag="$1" port="$2" model="$3" n
  for n in 1 2; do
    curl -s "http://127.0.0.1:${port}/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok$n\"}],\"temperature\":0,\"max_tokens\":16}" \
      > "$RAW/chat_${tag}_${n}.json"
    say "chat[$tag/$n] $(head -c 220 "$RAW/chat_${tag}_${n}.json" | tr -d '\n')"
  done
}

stop() {  # stop <tag>
  local tag="$1"
  local pidfile="/tmp/zcfix_serve_${tag}.pid"
  [ -f "$pidfile" ] || return 0
  local pid pgid _
  pid=$(cat "$pidfile")
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  pgid=${pgid:-$pid}
  say "stopping serve[$tag] pid=$pid pgid=$pgid"
  kill -TERM -"$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -KILL -"$pgid" 2>/dev/null || true
  rm -f "$pidfile"
  sleep 3
  local left
  left=$(ps -eo pid,pgid,comm | awk -v g="$pgid" '$2 == g {print $1" "$3}' | tr '\n' ' ')
  say "serve[$tag] stopped; our residual children: ${left:-<none>}"
}

say "=== LEG OFF: default-off (plain vllm serve, no zerocost env) ==="
export VLLM_CACHE_ROOT="/tmp/zcfix_cache_off"; rm -rf "$VLLM_CACHE_ROOT"
cd /tmp || exit 1
unset VLLM_HUST_FI_PREFILL_OUT VLLM_HUST_SKIP_COS_SIN 2>/dev/null || true
setsid nohup vllm serve "$MODEL" --gpu-memory-utilization 0.85 --max-model-len 4096 \
  --served-model-name qwen14b --port 8352 \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  > "$RAW/serve_off.log" 2>&1 < /dev/null &
OFF_PID=$!
echo "$OFF_PID" > /tmp/zcfix_serve_off.pid
say "serve[off] pid=$OFF_PID port=8352 (stock)"
if wait_healthy off 8352 "$OFF_PID"; then chats off 8352 qwen14b; fi
stop off
say "HBM(after off): $(hbm)"

say "=== LEG ON: zerocost ON (vllm-hust-ext run, cascade graph co-enabled) ==="
export VLLM_CACHE_ROOT="/tmp/zcfix_cache_on"; rm -rf "$VLLM_CACHE_ROOT"
cd /tmp || exit 1
setsid nohup vllm-hust-ext run -- vllm serve "$MODEL" --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --served-model-name qwen14b --port 8353 \
  --compilation-config '{"cudagraph_capture_sizes":[32,64,128]}' \
  > "$RAW/serve_on.log" 2>&1 < /dev/null &
ON_PID=$!
echo "$ON_PID" > /tmp/zcfix_serve_on.pid
say "serve[on] pid=$ON_PID port=8353 (vllm-hust-ext run)"
if wait_healthy on 8353 "$ON_PID"; then chats on 8353 qwen14b; fi
stop on
say "HBM(after on): $(hbm)"

say "=== evidence greps ==="
{
  echo "## OFF leg (stock): zerocost traces (expect <none>)"
  grep -nE "zero-cost wiring|zerocost|VLLM_HUST_FI_PREFILL_OUT|VLLM_HUST_SKIP_COS_SIN" "$RAW/serve_off.log" || echo "  <none>"
  echo "## OFF leg: FAIL/ERROR lines (expect <none>)"
  grep -nE "Traceback|CRITICAL" "$RAW/serve_off.log" || echo "  <none>"
  echo "## ON leg: zerocost ①/② lines"
  grep -nE "zero-cost wiring (①|②|\\(1\\)|\\(2\\))" "$RAW/serve_on.log" || echo "  <none>"
  echo "## ON leg: refused lines (expect <none>)"
  grep -nE "zero-cost wiring . refused" "$RAW/serve_on.log" || echo "  <none>"
  echo "## ON leg: cascade + demask markers"
  grep -nE "cascade plugin loaded|de-mask is ACTIVE" "$RAW/serve_on.log" || echo "  <none>"
  echo "## ON leg: Traceback/CRITICAL (expect <none>)"
  grep -nE "Traceback|CRITICAL" "$RAW/serve_on.log" || echo "  <none>"
} | tee "$RAW/evidence_greps.txt"

say "=== restore manager state ==="
vllm-hust-ext extension disable org.vllm-hust.zerocost-wiring > "$RAW/ext_disable_restore.txt" 2>&1
echo "disable rc=$?" | tee -a "$RAW/ext_disable_restore.txt"
vllm-hust-ext extension list > "$RAW/ext_list_restored.txt" 2>&1
cat "$RAW/ext_list_restored.txt"
say "HBM(final): $(hbm)"
say "=== fix smoke done ==="
