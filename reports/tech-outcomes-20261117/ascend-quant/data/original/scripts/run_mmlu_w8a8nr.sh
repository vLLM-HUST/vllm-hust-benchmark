#!/usr/bin/env bash
# ==============================================================================
# Single-arm MMLU-Pro 12,032: GDN-W8A8 (Qwen3.5-35B-A3B-W8A8-NoRot)
#
# Protocol is copied from mmlu-pro-accuracy/run_accuracy_pair.sh, which mirrors
# the published BF16 precheck: /v1/completions, frozen token-ID prompts,
# max_tokens=2048, temperature=0, seed=17, concurrency=16, no retries, no APC,
# no MTP, 32K context, FULL_DECODE_ONLY.
#
# Run as root inside a container that can read /data/jxd (user8-vllm-ascend).
# The only difference from the pair runner is that this arm is the W4A8+GDN
# mixed checkpoint (label w8a8nr) instead of the official W8A8 / gcy W4A8.
# ==============================================================================
set -uo pipefail

WS=/data/gcy/quant-Qwen3.5-35B-A3B
MMLU="$WS/mmlu-pro-accuracy"
ENV_PY=/data/jxd/envs/vllm-hust-v1-cann91/bin/python
MODEL=/data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot
CARD="${CARD:-7}"
PORT="${PORT:-8203}"
LABEL=w8a8nr
SERVED=qwen35-a3b-w8a8nr
OUTROOT="${OUTROOT:-/data/user8/tech-outcomes-20261117/ascend-quant/mmlu-w8a8nr}"
LOGS="$OUTROOT/logs"
RESULTS="$OUTROOT/results/w8a8nr-run1"
SMOKE="$OUTROOT/results/w8a8nr-smoke"
# Reuse the published W8A8 MMLU cache: it contains the precompiled npu_utils.so
# that this host cannot rebuild; a fresh cache directory breaks the worker.
CACHE="${CACHE:-/data/jxd/cache/cann91-w8a8-mmluacc}"
BF16_CSV=/data/jxd/whb/reports/mmlu-pro-qwen35-b0-precheck-20261012/results/per-case-outcomes.csv.gz
STATUS="$OUTROOT/RUN_STATUS.txt"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

[ "$(id -u)" = 0 ] || { echo "ERROR: run as root (model files are mode 600 root)" >&2; exit 1; }
for p in "$ENV_PY" "$MMLU/run_accuracy.py" "$MMLU/plan/contract.json" "$MODEL/config.json" "$BF16_CSV"; do
  [ -e "$p" ] || { echo "ERROR: missing $p" >&2; exit 1; }
done
mkdir -p "$LOGS" "$OUTROOT/results" "$CACHE"

health() { curl -sf -o /dev/null -m 3 "http://127.0.0.1:$PORT/health"; }

stop_server() {
  local pidfile="$LOGS/swe_server_ab_${LABEL}.pid" pid i
  if [ -s "$pidfile" ]; then
    pid="$(cat "$pidfile")"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  fi
  for i in $(seq 1 60); do health || break; sleep 5; done
  if [ -s "$pidfile" ]; then
    pid="$(cat "$pidfile")"
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  sleep 3
}
trap stop_server EXIT

log "START single-arm MMLU-Pro label=$LABEL card=$CARD port=$PORT"
log "model=$MODEL"
echo "started=$(date -u +%FT%TZ) label=$LABEL card=$CARD port=$PORT model=$MODEL" > "$STATUS"

if [ "${CLIENT_ONLY:-0}" = "1" ]; then
  log "CLIENT_ONLY: attach to the existing service on :$PORT"
  if ! health; then
    log "ERROR: CLIENT_ONLY but no healthy service on :$PORT"
    exit 1
  fi
else
  if health; then
    log "ERROR: port $PORT already has a healthy service"
    exit 1
  fi

  EXTRA='--quantization ascend --dtype bfloat16 --block-size 128 --max-num-seqs 16 --max-num-batched-tokens 8192 --no-enable-prefix-caching --enable-chunked-prefill --scheduling-policy fcfs --compilation-config {"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"} --cudagraph-capture-sizes 1 2 4 8 16'

  # run_swe_server_ab.sh ends with `wait "${SERVER_PID}"`, so it must be
  # backgrounded here; otherwise this driver blocks before the health check.
  ( cd "$WS" && \
    LABEL="$LABEL" MODEL_PATH="$MODEL" SERVED_NAME="$SERVED" PORT="$PORT" \
    ASCEND_RT_VISIBLE_DEVICES="$CARD" TP=1 MAX_MODEL_LEN=32768 GPU_MEM_UTIL=0.90 \
    PERF_CACHE_ROOT="$CACHE" LOG_DIR="$LOGS" EXTRA_SERVER_ARGS="$EXTRA" \
    ./run_swe_server_ab.sh ) >> "$LOGS/launcher.log" 2>&1 &
  log "server launcher started in background (pid $!)"
fi

log "waiting for health on :$PORT (cold compile may take 10-20 min)"
ready=0
for i in $(seq 1 360); do
  if health; then ready=1; break; fi
  sleep 10
done
if [ "$ready" != 1 ]; then
  log "ERROR: server not healthy after 3600s"
  ls -t "$LOGS"/swe_server_ab_${LABEL}_*.log 2>/dev/null | head -1 | xargs -r tail -120 >&2
  echo "server_not_healthy" >> "$STATUS"
  exit 1
fi
log "server ready"
echo "server_ready=$(date -u +%FT%TZ)" >> "$STATUS"

# ---- 20-case smoke (same go/no-go as the pair runner) ------------------------
if [ ! -d "$SMOKE" ]; then
  log "smoke: 20 cases -> $SMOKE"
  "$ENV_PY" -I "$MMLU/run_accuracy.py" \
    --plan "$MMLU/plan" --base-url "http://127.0.0.1:$PORT" \
    --served-model "$SERVED" --label "$LABEL" \
    --output "$SMOKE" --concurrency 16 --limit 20
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "ERROR: smoke client rc=$rc"
    echo "smoke_rc=$rc" >> "$STATUS"
    exit "$rc"
  fi
  "$ENV_PY" -I - "$SMOKE" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
s = json.loads((out / "summary.json").read_text())
sc = json.loads((out / "score.json").read_text())
recs = json.loads((out / "results.json").read_text())
bad = [r for r in recs if not r["success"]]
if bad:
    raise SystemExit("FAIL smoke: %d transport failures, e.g. %r" % (len(bad), bad[0]["error"]))
garbage = sum(1 for r in recs if not any(c.isalpha() for c in r["output"][:200]))
if garbage:
    raise SystemExit("FAIL smoke: %d outputs contain no letters" % garbage)
print("ok smoke: %d/%d success, %d correct, %d invalid, %d length-limited"
      % (len(recs) - len(bad), len(recs), sc["correct"], sc["invalid_answers"], sc["length_limited"]))
PY
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "ERROR: smoke check failed"
    echo "smoke_check_failed" >> "$STATUS"
    exit "$rc"
  fi
fi

# ---- full 12,032-case run ----------------------------------------------------
log "full client start -> $RESULTS"
echo "full_start=$(date -u +%FT%TZ)" >> "$STATUS"
CLIENT_ARGS=(--plan "$MMLU/plan" --base-url "http://127.0.0.1:$PORT" \
  --served-model "$SERVED" --label "$LABEL" --output "$RESULTS" --concurrency 16)
if [ "${RESUME:-0}" = 1 ]; then CLIENT_ARGS+=(--resume); fi
"$ENV_PY" -I "$MMLU/run_accuracy.py" "${CLIENT_ARGS[@]}"
rc=$?
log "full client rc=$rc"
echo "client_rc=$rc finished=$(date -u +%FT%TZ)" >> "$STATUS"

if [ "$rc" -eq 0 ]; then
  log "analysis -> $OUTROOT/analysis"
  "$ENV_PY" -I "$MMLU/analyze_accuracy.py" \
    --bf16-csv "$BF16_CSV" \
    --run "$LABEL=$RESULTS/per-case-outcomes.csv" \
    --outdir "$OUTROOT/analysis"
  log "analysis rc=$?"
  echo "analysis_done=$(date -u +%FT%TZ)" >> "$STATUS"
fi

exit "$rc"
