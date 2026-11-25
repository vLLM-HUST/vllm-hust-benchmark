#!/usr/bin/env bash
# Option B: BF16-anchored long-context PPL for OUR W4A8+GDN artifact.
# Reuses the existing P2-2 BF16 and W8A8 per-token NLL (same tokenizer/windows),
# runs our W4A8 on the same 4K/32K/128K windows, then compares all three.
#
# Run as root inside the container:
#   CARD=7 bash /data/user8/tech-outcomes-20261117/ascend-quant/run_longctx_ppl_bf16_anchor_card7.sh
set -uo pipefail

P22=/data/gcy/quant-Qwen3.5-35B-A3B/p2-2-long-context
GROOT=/data/gcy/quant-Qwen3.5-35B-A3B
PY=/data/jxd/envs/vllm-hust-v1-cann91/bin/python
CHUNKS=$GROOT/logs/wikitext2_chunks_1000.json
CARD="${CARD:-7}"
PORT="${PORT:-8220}"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="/data/user8/tech-outcomes-20261117/ascend-quant/p2-2-bf16-anchor/run_${TS}"
OUT=$RUN_DIR/out
LOGS=$RUN_DIR/logs
H="http://127.0.0.1:${PORT}"
REF=$P22/out

if [ "$(id -u)" != "0" ]; then echo "ERROR: run as root" >&2; exit 1; fi
[ -x "$PY" ] || { echo "ERROR: python missing: $PY" >&2; exit 1; }
[ -f "$CHUNKS" ] || { echo "ERROR: chunks missing: $CHUNKS" >&2; exit 1; }
[ -f "$P22/score_long.py" ] || { echo "ERROR: score_long.py missing" >&2; exit 1; }
mkdir -p "$OUT" "$LOGS"
echo "=== BF16-anchored long-context PPL $TS card=$CARD port=$PORT ==="
echo "run_dir=$RUN_DIR"

# Reuse the already-published BF16 / W8A8 per-token NLL. Do not modify them.
for arm in bf16 w8a8; do
  for L in 4096 32768 131072; do
    cp -f "$REF/nll_${arm}_${L}.npz" "$OUT/" || exit 1
    cp -f "$REF/meta_${arm}_${L}.json" "$OUT/" || exit 1
  done
done
echo "copied BF16/W8A8 NLL references"

port_free() { ! curl -sf -o /dev/null -m 3 "$H/health" 2>/dev/null; }

stop_server() {
  local pidfile="$LOGS/swe_server_ab_w4a8.pid" pid i
  if [ -s "$pidfile" ]; then
    pid=$(cat "$pidfile")
    kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  fi
  for i in $(seq 1 60); do port_free && break; sleep 5; done
  if ! port_free; then
    pkill -f "vllm serve /data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot" 2>/dev/null || true
  fi
  sleep 5
}

if ! port_free; then
  echo "ERROR: port $PORT busy; stop the old 8220 server first" >&2
  exit 1
fi

# P2-2 run_server.sh: TP1, max-model-len 133120, KV 8 GiB, no prefix caching,
# bt 4096, MNS2. Override only the model/served name for our W4A8+GDN.
LABEL=w4a8 MODEL_PATH=/data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot \
SERVED_NAME=qwen35-a3b-w8a8nr TP=1 \
ASCEND_RT_VISIBLE_DEVICES="$CARD" PORT="$PORT" \
MAX_MODEL_LEN=133120 GPU_MEM_UTIL=0.9 \
PERF_CACHE_ROOT=/data/jxd/cache/p22-w4a8ours LOG_DIR="$LOGS" \
  nohup "$P22/run_server.sh" > "$LOGS/server_w4a8ours.log" 2>&1 < /dev/null &
launcher=$!
echo "[w4a8ours] launcher=$launcher"

ready=0
for i in $(seq 1 240); do
  if curl -sf -o /dev/null -m 3 "$H/health"; then ready=1; echo "[w4a8ours] ready after $((i*10))s"; break; fi
  if ! kill -0 "$launcher" 2>/dev/null; then
    echo "ERROR: launcher exited; see $LOGS/server_w4a8ours.log" >&2
    tail -80 "$LOGS/server_w4a8ours.log" >&2
    exit 1
  fi
  sleep 10
done
if [ "$ready" != "1" ]; then
  echo "ERROR: not ready after 2400s; see $LOGS/server_w4a8ours.log" >&2
  tail -80 "$LOGS/server_w4a8ours.log" >&2
  exit 1
fi

"$PY" -I "$P22/score_long.py" \
  --base-url "$H" --model qwen35-a3b-w8a8nr --arm w4a8ours \
  --chunks "$CHUNKS" --lengths 4096,32768,131072 --out-dir "$OUT"
rc=$?

stop_server

"$PY" -I "$P22/compare_long.py" \
  --dir "$OUT" --arms bf16,w8a8,w4a8ours --base bf16 || true

echo "=== DONE score_rc=$rc ==="
echo "run_dir=$RUN_DIR"
echo "ppl=$OUT/longctx_ppl.csv"
echo "difficulty=$OUT/longctx_difficulty.csv"
echo "json=$OUT/longctx.json"
exit "$rc"
