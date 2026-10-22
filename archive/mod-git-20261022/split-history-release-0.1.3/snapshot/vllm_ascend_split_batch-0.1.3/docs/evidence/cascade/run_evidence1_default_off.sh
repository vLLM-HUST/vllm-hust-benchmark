#!/bin/bash
# Evidence 1 (new baseline): default-off zero-regression smoke.
# Runs the plugin installed but NOT enabled, then proves zero cascade activation.
set -x
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs
mkdir -p "$OUT"
PORT=8341
LOG=$OUT/serve-default-off-v1.log

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust

# ensure the bundle is disabled (installed, not enabled)
cd /tmp && vllm-hust-ext extension disable org.vllm-hust.split-batch-full-graph 2>&1 | tail -1

cd /tmp
HF_HUB_OFFLINE=1 ASCEND_RT_VISIBLE_DEVICES=6 nohup vllm serve \
  /data/shared_models/Qwen--Qwen2.5-Coder-14B-Instruct \
  --max-model-len 4096 --enforce-eager --port $PORT \
  --gpu-memory-utilization 0.85 \
  > "$LOG" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID"

# wait for readiness (up to 10 min)
for i in $(seq 1 120); do
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
    echo "READY after ${i}0s"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "SERVER DIED"; tail -30 "$LOG"; exit 1
  fi
  sleep 5
done

echo "=== health ==="
curl -s -o /dev/null -w "health_http=%{http_code}\n" http://127.0.0.1:$PORT/health

echo "=== chat completions (3 prompts) ==="
for p in "hi" "what is 2+2" "write a haiku about NPUs"; do
  curl -s -o /dev/null -w "chat_http=%{http_code}\n" http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"/data/shared_models/Qwen--Qwen2.5-Coder-14B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":16}"
done

echo "=== zero-activation proof ==="
echo "cascade_active_marker_count=$(grep -c 'cascade-active' "$LOG")"
echo "cascade_loaded_marker_count=$(grep -c 'cascade plugin loaded' "$LOG")"
echo "wheel_unavailable_count=$(grep -c 'ascend_kernel wheel unavailable' "$LOG")"
grep -m2 'cascade plugin loaded' "$LOG"

kill $SERVER_PID 2>/dev/null
sleep 5
kill -9 $SERVER_PID 2>/dev/null
curl -s -o /dev/null -w "port_after_kill=%{http_code}\n" --max-time 3 http://127.0.0.1:$PORT/health
echo "=== DONE ==="
