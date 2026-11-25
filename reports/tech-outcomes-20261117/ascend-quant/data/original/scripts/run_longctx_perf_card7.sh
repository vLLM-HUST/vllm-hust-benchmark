#!/usr/bin/env bash
# Long-context performance on card 7, single request (C1), same KV 18.85 GiB.
# Inputs: 8K / 32K / 128K / 260K; output 256 tokens; 1 warmup + 3 measured runs.
# Arms: W8A8 and W4A8+GDN, TP1. Uses vllm bench serve (openai backend).
#
# Run as root inside the container:
#   CARD=7 bash /data/user8/tech-outcomes-20261117/ascend-quant/run_longctx_perf_card7.sh
set -uo pipefail

WS=/data/jxd/quant-Qwen3.5-35B-A3B
GROOT=/data/gcy/quant-Qwen3.5-35B-A3B
VLLM=/data/jxd/envs/vllm-hust-v1-cann91/bin/vllm
PY=/data/jxd/envs/vllm-hust-v1-cann91/bin/python
TOK=/data/jxd/Qwen3.5-35B-A3B
CARD="${CARD:-7}"
PORT="${PORT:-8240}"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="/data/user8/tech-outcomes-20261117/ascend-quant/long-context-perf/run_${TS}"
OUT=$RUN_DIR/out
LOGS=$RUN_DIR/logs
H="http://127.0.0.1:${PORT}"
LENGTHS="${LENGTHS:-8192 32768 131072 260000}"
REPS="${REPS:-3}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"

if [ "$(id -u)" != "0" ]; then echo "ERROR: run as root" >&2; exit 1; fi
[ -x "$VLLM" ] || { echo "ERROR: vllm missing: $VLLM" >&2; exit 1; }
[ -x "$PY" ] || { echo "ERROR: python missing: $PY" >&2; exit 1; }
mkdir -p "$OUT" "$LOGS"
echo "=== long-context perf $TS card=$CARD port=$PORT lengths=[$LENGTHS] reps=$REPS ==="
echo "run_dir=$RUN_DIR"

port_free() { ! curl -sf -o /dev/null -m 3 "$H/health" 2>/dev/null; }

stop_server() {
  local label="$1" model="$2" pidfile="$LOGS/swe_server_ab_${label}.pid" pid i
  if [ -s "$pidfile" ]; then
    pid=$(cat "$pidfile")
    kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  fi
  for i in $(seq 1 60); do port_free && break; sleep 5; done
  if ! port_free; then
    pkill -f "vllm serve ${model}" 2>/dev/null || true
  fi
  sleep 5
}

bench_once() {
  local served="$1" L="$2" outlen="$3" fname="$4"
  local args=(bench serve --backend openai --base-url "$H" --model "$served"
    --tokenizer "$TOK" --trust-remote-code
    --dataset-name random --random-input-len "$L" --random-output-len "$outlen"
    --num-prompts 1 --request-rate inf --ignore-eos --seed 0
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99)
  if [ -n "$fname" ]; then
    args+=(--save-result --result-dir "$OUT" --result-filename "$fname")
  fi
  "$VLLM" "${args[@]}"
}

run_arm() {
  local label="$1" model="$2" served="$3" arm="$4"
  local slog="$LOGS/server_${arm}_${TS}.log"
  echo "=== [$arm] label=$label model=$model ==="
  port_free || { echo "ERROR: port $PORT busy" >&2; return 1; }

  LABEL="$label" MODEL_PATH="$model" SERVED_NAME="$served" TP=1 \
  ASCEND_RT_VISIBLE_DEVICES="$CARD" PORT="$PORT" \
  MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.95 \
  PERF_CACHE_ROOT="/data/jxd/cache/longperf-${arm}" LOG_DIR="$LOGS" \
  EXTRA_SERVER_ARGS='--no-enable-prefix-caching --max-num-seqs 1 --max-num-batched-tokens 4096 --kv-cache-memory-bytes 20242376704' \
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

  for L in $LENGTHS; do
    echo "[$arm] warmup L=$L out=16"
    bench_once "$served" "$L" 16 "" > "$LOGS/${arm}_L${L}_warmup.log" 2>&1 || {
      echo "ERROR: warmup failed L=$L; see $LOGS/${arm}_L${L}_warmup.log" >&2
      stop_server "$label" "$model"; return 1; }
    for r in $(seq 1 "$REPS"); do
      local fname="${arm}_L${L}_r${r}.json"
      echo "[$arm] measured L=$L rep=$r -> $fname"
      bench_once "$served" "$L" "$OUTPUT_TOKENS" "$fname" \
        > "$LOGS/${arm}_L${L}_r${r}.log" 2>&1 || {
        echo "ERROR: bench failed L=$L rep=$r" >&2
        stop_server "$label" "$model"; return 1; }
      sleep 2
    done
  done

  stop_server "$label" "$model"
  echo "[$arm] done"
}

rc=0
run_arm w8a8 /data/jxd/Qwen3.5-35B-A3B-W8A8 qwen35-a3b-w8a8 w8a8 || rc=1
run_arm w4a8 /data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot qwen35-a3b-w8a8nr w4a8 || rc=1

cp "$(dirname "${BASH_SOURCE[0]}")/summarize_longperf.py" "$RUN_DIR/summarize_longperf.py" 2>/dev/null || true
"$PY" -I "$RUN_DIR/summarize_longperf.py" --dir "$OUT" || true

echo "=== DONE rc=$rc ==="
echo "run_dir=$RUN_DIR"
echo "summary=$OUT/longperf_summary.csv"
exit "$rc"
