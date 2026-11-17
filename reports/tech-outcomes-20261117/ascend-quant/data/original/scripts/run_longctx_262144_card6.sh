#!/usr/bin/env bash
# Near-max long-context extension on card 6:
#   - exact full-context capacity smoke: prompt 262,143 tokens + 1 generated token
#   - task-level NIAH/longgen at body 261,000 / 260,800 (total budget ~262.0K)
# Both arms (official W8A8, W4A8/GDN), TP1, 8 GiB KV, max-model-len 262144.
# Run as root inside user8-vllm-ascend.
set -uo pipefail

SRC=/data/gcy/quant-Qwen3.5-35B-A3B/p2-3-long-context-stability
GROOT=/data/gcy/quant-Qwen3.5-35B-A3B
PY=/data/jxd/envs/vllm-hust-v1-cann91/bin/python
TOK=/data/jxd/Qwen3.5-35B-A3B/tokenizer.json
CHUNKS=$GROOT/logs/wikitext2_chunks_1000.json
CARD="${CARD:-6}"
PORT="${PORT:-8232}"
L_NIAH="${L_NIAH:-261000}"
L_LONG="${L_LONG:-260800}"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="/data/user8/tech-outcomes-20261117/ascend-quant/long-context-262144/run_${TS}"
OUT=$RUN_DIR/out
LOGS=$RUN_DIR/logs
H="http://127.0.0.1:${PORT}"

[ "$(id -u)" = 0 ] || { echo "ERROR: run as root" >&2; exit 1; }
[ -x "$PY" ] || { echo "ERROR: python missing: $PY" >&2; exit 1; }
[ -f "$CHUNKS" ] || { echo "ERROR: chunks missing: $CHUNKS" >&2; exit 1; }
[ -f "$TOK" ] || { echo "ERROR: tokenizer missing: $TOK" >&2; exit 1; }
mkdir -p "$OUT" "$LOGS"
echo "=== 262K-class long-context extension $TS card=$CARD port=$PORT ==="
echo "run_dir=$RUN_DIR  L_NIAH=$L_NIAH L_LONG=$L_LONG"

"$PY" -I "$SRC/build_niah.py" \
  --chunks "$CHUNKS" --tokenizer "$TOK" --out "$OUT" \
  --lengths "$L_NIAH" --depths 0.0,0.25,0.5,0.75,1.0 --samples "${L_NIAH}:5" \
  --multi-lengths "$L_NIAH" --multi-samples "${L_NIAH}:5" --multi-depths 0.1,0.5,0.9 || exit 1
[ -s "$OUT/niah_cases.jsonl" ] || { echo "ERROR: cases not built" >&2; exit 1; }

port_free() { ! curl -sf -o /dev/null -m 3 "$H/health" 2>/dev/null; }

stop_server() {
    local label="$1" pidfile="$LOGS/swe_server_ab_${label}.pid" pid i
    if [ -s "$pidfile" ]; then
        pid=$(cat "$pidfile")
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
    for i in $(seq 1 60); do port_free && break; sleep 5; done
    if ! port_free; then
        # Port-scoped only; never pkill by model path (the 8241 W8A8 service shares it).
        pkill -f -- "--port ${PORT}" 2>/dev/null || true
    fi
    sleep 5
}

capacity_smoke() {
    local arm="$1" served="$2"
    echo "=== [$arm] full-context capacity smoke (262,143 prompt + 1 token) ==="
    "$PY" -I - "$H" "$served" "$OUT/capacity_smoke_${arm}.json" "$CHUNKS" <<'PY'
import json, sys, time, urllib.request
base, model, out, chunks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
pieces = json.load(open(chunks))
toks = [t for p in pieces for t in p]
length = 262143
if len(toks) < length:
    print("corpus only %d tokens" % len(toks))
    length = len(toks)
prompt = toks[:length]
body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 1,
                   "temperature": 0, "ignore_eos": True, "stream": False}).encode()
req = urllib.request.Request(base + "/v1/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    rec = {"ok": True, "http": r.status, "prompt_len": length,
           "elapsed_s": time.time() - t0, "usage": d.get("usage"),
           "text": (d.get("choices") or [{}])[0].get("text", "")[:120]}
except Exception as e:
    rec = {"ok": False, "prompt_len": length, "elapsed_s": time.time() - t0,
           "error": repr(e)}
json.dump(rec, open(out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(rec, ensure_ascii=False))
PY
    echo "[$arm] capacity smoke rc=$? (file: $OUT/capacity_smoke_${arm}.json)"
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
        stop_server "$label"
        return 1
    fi

    capacity_smoke "$arm" "$served" || true

    echo "[$arm] NIAH body $L_NIAH ..."
    "$PY" -I "$SRC/run_niah.py" --base-url "$H" --model "$served" --arm "$arm" \
      --cases "$OUT/niah_cases.jsonl" --out "$OUT/niah_%s.jsonl" \
      --max-tokens 1024 --skip-existing || { stop_server "$label"; return 1; }

    echo "[$arm] longgen body $L_LONG ..."
    "$PY" -I "$SRC/run_longgen.py" --base-url "$H" --model "$served" --arm "$arm" \
      --chunks "$CHUNKS" --tokenizer "$TOK" --out "$OUT/longgen_%s.jsonl" \
      --lengths "${L_LONG}:5" --max-tokens 1024 --skip-existing || { stop_server "$label"; return 1; }

    stop_server "$label"
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
