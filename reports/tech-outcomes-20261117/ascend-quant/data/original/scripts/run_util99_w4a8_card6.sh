#!/usr/bin/env bash
# 19.22 GiB same-KV appendix: W4A8/GDN arm, card 6, port 8210.
# Pairs with the existing W8A8 util99 runs (auto-estimated 19.22 GiB / 820,136 tokens).
# Run as root inside user8-vllm-ascend.
set -uo pipefail

WS=/data/jxd/quant-Qwen3.5-35B-A3B
MODEL=/data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot
CARD="${CARD:-6}"
PORT="${PORT:-8210}"
LABEL=w8a8nr
SERVED=qwen35-a3b-w8a8nr
OUTROOT=/data/user8/tech-outcomes-20261117/ascend-quant/util99
LOGS=$OUTROOT/logs
CACHE="${CACHE:-/data/jxd/cache/p23-256k-w4a8}"
CLI_WRAPPER=/data/user8/tech-outcomes-20261117/ascend-quant/swe-prefix-reuse-wrapper.sh
META=$OUTROOT/server-metadata-w4a8ours-tp1-util99.json
TAG=w4a8ours-tp1-util99
KV_BYTES=20637317857
STATUS=$OUTROOT/RUN_STATUS.txt

[ "$(id -u)" = 0 ] || { echo "ERROR: root required" >&2; exit 1; }
mkdir -p "$LOGS" "$OUTROOT"
log() { echo "[$(date -u +%FT%TZ)] $*"; }

# Accurate server metadata for the client record.
python3 - "$META" "$PORT" "$CARD" "$CACHE" "$KV_BYTES" <<'PY'
import datetime, json, sys
dst, port, card, cache, kv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
src = "/data/jxd/swe-prefix-reuse/server-metadata-w4a8ours-tp1.json"
j = json.load(open(src))
j["recorded_at_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
j["recorded_by"] = ("19.22 GiB same-KV appendix: W4A8/GDN pinned to 20637317857 B (19.22 GiB). "
                    "The W8A8 util99 arm is the existing 2026-10-23 auto-estimated 19.22 GiB / "
                    "820,136-token run; both use the same workload/protocol/client.")
serv = j.setdefault("serving", {})
serv["gpu_memory_utilization"] = "0.95 (explicit KV pin; W8A8 util99 arm used 0.99 auto)"
serv["kv_cache_memory_bytes"] = "20637317857 (19.22 GiB, explicit pin)"
serv["explicitly_set_args"] = (
    "显式设置: --kv-cache-memory-bytes 20637317857 / --max-num-seqs 16 / "
    "--max-num-batched-tokens 4096 / --speculative-config {\"method\":\"mtp\","
    "\"num_speculative_tokens\":2} / --compilation-config "
    "{\"max_cudagraph_capture_size\":256,\"cudagraph_mode\":\"FULL\"}")
j["endpoint"] = f"http://127.0.0.1:{port}/v1/completions"
j["launch_port"] = int(port)
j["launch_script"] = ("/data/gcy/quant-Qwen3.5-35B-A3B/run_swe_server_ab.sh "
    "(CANN 9.1 + VENDOR_OPP=1 + TORCHINDUCTOR_AUTOGRAD_CACHE=0)")
j["launch_command_redacted"] = (
    f"ASCEND_RT_VISIBLE_DEVICES={card} TP=1 PORT={port} MAX_MODEL_LEN=262144 "
    f"GPU_MEM_UTIL=0.95 LABEL=w8a8nr PERF_CACHE_ROOT={cache} "
    f"VENDOR_OPP=1 CANN_SET_ENV=/data/jxd/Ascend-9.1.0/ascend-toolkit/set_env.sh "
    f"ATB_SET_ENV=/data/jxd/Ascend-9.1.0/nnal/atb/set_env.sh "
    f"EXTRA_SERVER_ARGS='--quantization ascend --kv-cache-memory-bytes {kv} ...' ./run_swe_server_ab.sh")
json.dump(j, open(dst, "w"), ensure_ascii=False, indent=1)
print("wrote", dst)
PY

pidfile="$LOGS/swe_server_ab_${LABEL}.pid"
health() { curl -sf -o /dev/null -m 3 "http://127.0.0.1:${PORT}/health"; }
stop_server() {
    local pid i
    if [ -s "$pidfile" ]; then
        pid=$(cat "$pidfile")
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
    for i in $(seq 1 60); do health || break; sleep 5; done
    if health; then
        # Only our port; never pkill by model path (the 8241 W8A8 service shares it).
        pkill -f -- "--port ${PORT}" 2>/dev/null || true
    fi
    sleep 3
}
trap stop_server EXIT

log "START 19.22 GiB W4A8/GDN arm card=$CARD port=$PORT kv=$KV_BYTES"
echo "started=$(date -u +%FT%TZ) card=$CARD port=$PORT kv=$KV_BYTES" > "$STATUS"
if health; then log "ERROR: port $PORT already healthy"; exit 1; fi
rm -f "$pidfile"

( cd "$WS" && \
  LABEL="$LABEL" MODEL_PATH="$MODEL" SERVED_NAME="$SERVED" TP=1 \
  ASCEND_RT_VISIBLE_DEVICES="$CARD" PORT="$PORT" MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.95 \
  PERF_CACHE_ROOT="$CACHE" LOG_DIR="$LOGS" \
  EXTRA_SERVER_ARGS="--quantization ascend --kv-cache-memory-bytes $KV_BYTES --max-num-seqs 16 --max-num-batched-tokens 4096 --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2} --compilation-config {\"max_cudagraph_capture_size\":256,\"cudagraph_mode\":\"FULL\"}" \
  nohup /data/gcy/quant-Qwen3.5-35B-A3B/run_swe_server_ab.sh > "$LOGS/launcher.log" 2>&1 < /dev/null & )
log "server launcher started"

ready=0
for i in $(seq 1 240); do
    if health; then ready=1; break; fi
    sleep 10
done
if [ "$ready" != 1 ]; then
    log "ERROR: server not healthy after 2400s"
    ls -t "$LOGS"/swe_server_${LABEL}_*.log 2>/dev/null | head -1 | xargs -r tail -100 >&2
    echo "server_not_healthy" >> "$STATUS"
    exit 1
fi
log "server ready"
grep -E 'Available KV cache memory|GPU KV cache size' "$LOGS"/swe_server_ab_${LABEL}_*.log 2>/dev/null | tail -2 | tee -a "$STATUS"
echo "ready=$(date -u +%FT%TZ)" >> "$STATUS"

cd /data/gcy/quant-Qwen3.5-35B-A3B
CLI="$CLI_WRAPPER" TAG="$TAG" MODEL="$SERVED" PORT="$PORT" META="$META" \
OUT_ROOT=/data/jxd/swe-prefix-reuse/results LOG_DIR="$LOGS" \
bash ./run_swe_ab_sweep.sh
rc=$?
log "sweep rc=$rc"
echo "sweep_rc=$rc finished=$(date -u +%FT%TZ)" >> "$STATUS"
exit "$rc"
