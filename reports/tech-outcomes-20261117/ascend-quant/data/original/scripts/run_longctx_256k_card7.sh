#!/usr/bin/env bash
# 256K-class long-context extension on card 7.
# 260,000-token prompts + 1,024 generation = ~261K, under max-model-len 262144.
# Reuses the P2-3 NIAH/longgen clients with 8 GiB KV, TP1, single card.
#
# Run as root inside the container:
#   CARD=7 bash /data/user8/tech-outcomes-20261117/ascend-quant/run_longctx_256k_card7.sh
set -uo pipefail

SRC=/data/gcy/quant-Qwen3.5-35B-A3B/p2-3-long-context-stability
GROOT=/data/gcy/quant-Qwen3.5-35B-A3B
PY=/data/jxd/envs/vllm-hust-v1-cann91/bin/python
TOK=/data/jxd/Qwen3.5-35B-A3B/tokenizer.json
CHUNKS=$GROOT/logs/wikitext2_chunks_1000.json
CARD="${CARD:-7}"
PORT="${PORT:-8231}"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="/data/user8/tech-outcomes-20261117/ascend-quant/long-context-256k/run_${TS}"
OUT=$RUN_DIR/out
LOGS=$RUN_DIR/logs
H="http://127.0.0.1:${PORT}"

if [ "$(id -u)" != "0" ]; then echo "ERROR: run as root" >&2; exit 1; fi
[ -x "$PY" ] || { echo "ERROR: python missing: $PY" >&2; exit 1; }
[ -f "$CHUNKS" ] || { echo "ERROR: chunks missing: $CHUNKS" >&2; exit 1; }
[ -f "$TOK" ] || { echo "ERROR: tokenizer missing: $TOK" >&2; exit 1; }
mkdir -p "$OUT" "$LOGS"
echo "=== 260K long-context extension $TS card=$CARD port=$PORT ==="
echo "run_dir=$RUN_DIR"

# 260,000-token bodies, 5 samples per depth, plus 5 multi-needle cases.
"$PY" -I "$SRC/build_niah.py" \
  --chunks "$CHUNKS" --tokenizer "$TOK" --out "$OUT" \
  --lengths 260000 --depths 0.0,0.25,0.5,0.75,1.0 --samples 260000:5 \
  --multi-lengths 260000 --multi-samples 260000:5 --multi-depths 0.1,0.5,0.9 || exit 1
[ -s "$OUT/niah_cases.jsonl" ] || { echo "ERROR: cases not built" >&2; exit 1; }

port_free() { ! curl -sf -o /dev/null -m 3 "$H/health" 2>/dev/null; }

stop_server() {
  local label="$1" model="$2" pidfile="$LOGS/swe_server_ab_${label}.pid" pid i
  if [ -s "$pidfile" ]; then
    pid=$(cat "$pidfile")
    kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  fi
  for i in $(seq 1 60); do port_free && break; sleep 5; done
  if ! port_free; then
    # The pid file can point at the APIServer child rather than the vllm
    # launcher; kill by exact model path as a fallback.
    pkill -f "vllm serve ${model}" 2>/dev/null || true
  fi
  sleep 5
}

run_arm() {
  local label="$1" model="$2" served="$3" arm="$4"
  local slog="$LOGS/server_${arm}_${TS}.log"
  echo "=== [$arm] label=$label model=$model ==="
  port_free || { echo "ERROR: port $PORT busy" >&2; return 1; }

  LABEL="$label" MODEL_PATH="$model" SERVED_NAME="$served" TP=1 \
  ASCEND_RT_VISIBLE_DEVICES="$CARD" PORT="$PORT" \
  MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.9 \
  PERF_CACHE_ROOT="/data/jxd/cache/p23-256k-${arm}" LOG_DIR="$LOGS" \
  EXTRA_SERVER_ARGS='--no-enable-prefix-caching --max-num-seqs 2 --max-num-batched-tokens 4096 --kv-cache-memory-bytes 8589934592' \
    nohup "$GROOT/run_swe_server_ab.sh" > "$slog" 2>&1 < /dev/null &
  local launcher=$!
  echo "[$arm] launcher=$launcher"

  local ready=0 i
  for i in $(seq 1 240); do
    if curl -sf -o /dev/null -m 3 "$H/health"; then ready=1; echo "[$arm] ready after $((i*10))s"; break; fi
    if ! kill -0 "$launcher" 2>/dev/null; then
      echo "ERROR: launcher exited; see $slog" >&2
      tail -80 "$slog" >&2
      return 1
    fi
    sleep 10
  done
  if [ "$ready" != "1" ]; then
    echo "ERROR: not ready after 2400s; see $slog" >&2
    tail -80 "$slog" >&2
    stop_server "$label" "$model"
    return 1
  fi

  echo "[$arm] NIAH 260K ..."
  "$PY" -I "$SRC/run_niah.py" --base-url "$H" --model "$served" --arm "$arm" \
    --cases "$OUT/niah_cases.jsonl" --out "$OUT/niah_%s.jsonl" \
    --max-tokens 1024 --skip-existing || { stop_server "$label" "$model"; return 1; }

  echo "[$arm] longgen 260K ..."
  "$PY" -I "$SRC/run_longgen.py" --base-url "$H" --model "$served" --arm "$arm" \
    --chunks "$CHUNKS" --tokenizer "$TOK" --out "$OUT/longgen_%s.jsonl" \
    --lengths 260000:5 --max-tokens 1024 --skip-existing || { stop_server "$label" "$model"; return 1; }

  stop_server "$label" "$model"
  echo "[$arm] done"
}

rc=0
run_arm w8a8 /data/jxd/Qwen3.5-35B-A3B-W8A8 qwen35-a3b-w8a8 w8a8 || rc=1
run_arm w4a8 /data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot qwen35-a3b-w8a8nr w4a8 || rc=1

cp "$SRC/compare_stability.py" "$RUN_DIR/compare_stability.py"
"$PY" -I "$RUN_DIR/compare_stability.py" || true

echo "=== DONE rc=$rc ==="
echo "run_dir=$RUN_DIR"
echo "summary=$OUT/stability.json"
exit "$rc"
