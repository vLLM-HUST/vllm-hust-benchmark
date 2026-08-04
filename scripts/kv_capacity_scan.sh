#!/bin/bash
# Issue #134: KV capacity scan and tiering state machine analysis.
#
# Part A: KV capacity scan — 4 capacities × 3 workloads × 3 reps = 36 runs
# Part B: Tiering comparison — 3 configs × 3 reps = 9 runs (at 8 GiB KV)
#
# Usage:
#   ./kv_capacity_scan.sh [--reps N] [--part A|B|both] [--dry-run]
#
# Output:
#   /data/issue134-results/raw_results/<workload>/<kv_gib>/rep-<N>/{raw.json,server.log,metrics.json,run_info.txt}
#   /data/issue134-results/tiering/<config>/rep-<N>/{raw.json,server.log,metrics.json,run_info.txt}
#   /data/issue134-results/summary.json

set -euo pipefail

REPS=${REPS:-3}
PART="both"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reps) REPS="$2"; shift 2 ;;
        --part) PART="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH="/data/vllm-hust-benchmark-issue97/models/Qwen2.5-14B-Instruct"
PYTHON="/root/miniconda3/envs/vllm-hust-dev/bin/python"
VLLM_HUST_REPO="/root/vllm/vllm-hust"
ASCEND_REPO="/root/vllm/vllm-ascend-hust"
RESULT_DIR="/data/issue134-results"
PORT=8420
HOST="127.0.0.1"

# ShareGPT dataset (search common locations)
SHAREGPT_DATASET="${SHAREGPT_DATASET:-}"
for candidate in \
    "/root/vllm/vllm-hust/benchmarks/sharegpt.json" \
    "/root/vllm/vllm-hust-benchmark/data/sharegpt.json" \
    "/data/sharegpt.json"; do
    if [ -f "$candidate" ]; then
        SHAREGPT_DATASET="$candidate"
        break
    fi
done

# KV capacity targets: GiB → gpu_memory_utilization
# Calibrated from server log: 0.6→8.04GiB, weights=27.54, overhead≈0.91
# KV = util*60.96 - 27.54 - 0.91
declare -A KV_UTIL_MAP
KV_UTIL_MAP[8]="0.60"
KV_UTIL_MAP[16]="0.73"
KV_UTIL_MAP[24]="0.86"
KV_UTIL_MAP[32]="0.95"

KV_CAPACITIES=(8 16 24 32)
SCAN_WORKLOADS=("random-online" "sharegpt-online" "prefix-repetition-online")

# Tiering configs for Part B
TIERING_CONFIGS=("hbm-only" "kv-constrained" "kv-constrained-utility")
TIERING_WORKLOAD="prefix-repetition-online"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_VISIBLE_DEVICES=0
export VLLM_USE_V1=1
export VLLM_TARGET_DEVICE=npu
export VLLM_PLUGINS=ascend
export PYTHONDONTWRITEBYTECODE=1
export HF_ENDPOINT="https://hf-mirror.com"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
# Unset proxy env vars to prevent curl from routing localhost through proxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"

_atb_home="/usr/local/Ascend/nnal/atb/9.0.0/atb"
_cxx_abi_dir="cxx_abi_1"
_conda_lib="$(dirname "$(dirname "$PYTHON")")/lib"
# Include Ascend driver libs (libascend_hal.so) and CANN toolkit libs for NPU runtime
export LD_LIBRARY_PATH="${_conda_lib}:${_atb_home}/${_cxx_abi_dir}/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}"
export ATB_HOME_PATH="${_atb_home}/${_cxx_abi_dir}"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*"; }

kill_leftover_processes() {
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "run_engine_core" 2>/dev/null || true
    pkill -9 -f "api_server" 2>/dev/null || true
    # Kill anything on our port
    local pid
    pid=$(lsof -ti ":${PORT}" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        kill -9 $pid 2>/dev/null || true
    fi
    sleep 2
}

wait_for_npu_idle() {
    local max_wait=60
    local waited=0
    while [ $waited -lt $max_wait ]; do
        local used_mb
        used_mb=$(npu-smi info -t usages -i 0 2>/dev/null \
            | grep 'Device HBM Used' \
            | head -1 \
            | awk -F: '{print $2}' \
            | tr -d ' %' || echo "0")
        if [ "$used_mb" -lt 5000 ] 2>/dev/null; then
            return 0
        fi
        log "  NPU still using ${used_mb}MB HBM, waiting..."
        sleep 5
        waited=$((waited + 5))
    done
    log "  WARNING: NPU not idle after ${max_wait}s, proceeding anyway"
}

wait_for_server() {
    local max_wait=600
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl --noproxy "*" -s "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
            log "  Server is ready (waited ${waited}s)"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    log "  ERROR: Server not ready after ${max_wait}s"
    return 1
}

collect_metrics() {
    local output_file="$1"
    curl --noproxy "*" -s "http://${HOST}:${PORT}/metrics" > "$output_file" 2>/dev/null || true
}

build_serve_cmd() {
    local gpu_util="$1"
    echo "$PYTHON -m vllm.entrypoints.cli.main serve" \
        "$MODEL_PATH" \
        "--host $HOST" \
        "--port $PORT" \
        "--dtype float16" \
        "--gpu-memory-utilization $gpu_util" \
        "--max-model-len 32768" \
        "--enable-prefix-caching"
}

build_bench_cmd() {
    local workload="$1"
    local output_dir="$2"
    local cmd="$PYTHON -m vllm.entrypoints.cli.main bench serve"
    cmd="$cmd --backend vllm --endpoint /v1/completions"
    cmd="$cmd --model $MODEL_PATH"
    cmd="$cmd --num-prompts 200 --request-rate 1"
    cmd="$cmd --save-result --result-dir $output_dir --result-filename raw.json"

    case "$workload" in
        random-online)
            cmd="$cmd --dataset-name random --random-input-len 1024 --random-output-len 256"
            ;;
        sharegpt-online)
            cmd="$cmd --dataset-name sharegpt"
            if [ -n "$SHAREGPT_DATASET" ]; then
                cmd="$cmd --dataset-path $SHAREGPT_DATASET"
            fi
            ;;
        prefix-repetition-online)
            cmd="$cmd --dataset-name prefix_repetition --random-input-len 4096 --random-output-len 256"
            ;;
    esac
    echo "$cmd"
}

run_single_experiment() {
    local workload="$1"
    local kv_gib="$2"
    local rep="$3"
    local output_dir="$4"
    local extra_env="$5"

    local gpu_util="${KV_UTIL_MAP[$kv_gib]}"
    log "  Running: workload=$workload kv=${kv_gib}GiB util=$gpu_util rep=$rep"

    mkdir -p "$output_dir"

    # Write run info
    cat > "$output_dir/run_info.txt" <<EOF
workload=$workload
kv_target_gib=$kv_gib
gpu_memory_utilization=$gpu_util
rep=$rep
timestamp=$(date -u '+%Y%m%dT%H%M%SZ')
port=$PORT
model=$MODEL_PATH
extra_env=$extra_env
EOF

    if [ $DRY_RUN -eq 1 ]; then
        log "  [DRY RUN] Skipping actual execution"
        return 0
    fi

    kill_leftover_processes
    wait_for_npu_idle

    # Start server
    local serve_cmd
    serve_cmd=$(build_serve_cmd "$gpu_util")
    log "  Starting server: $serve_cmd"

    # Apply extra env vars (e.g., utility victim selection)
    if [ -n "$extra_env" ]; then
        eval "export $extra_env"
    else
        unset VLLM_ASCEND_ENABLE_UTILITY_VICTIM_SELECTION 2>/dev/null || true
    fi

    $serve_cmd > "$output_dir/server.log" 2>&1 &
    local server_pid=$!

    # Reset extra env
    if [ -n "$extra_env" ]; then
        eval "unset ${extra_env%%=*}" 2>/dev/null || true
    fi

    # Wait for server
    if ! wait_for_server; then
        log "  ERROR: Server failed to start"
        kill -9 $server_pid 2>/dev/null || true
        kill_leftover_processes
        return 1
    fi

    # Collect pre-benchmark metrics
    collect_metrics "$output_dir/metrics_pre.json"

    # Run benchmark
    local bench_cmd
    bench_cmd=$(build_bench_cmd "$workload" "$output_dir")
    log "  Running benchmark: $bench_cmd"

    if ! $bench_cmd 2>&1 | tee "$output_dir/bench.log"; then
        log "  ERROR: Benchmark failed"
        kill -9 $server_pid 2>/dev/null || true
        kill_leftover_processes
        return 1
    fi

    # Collect post-benchmark metrics
    collect_metrics "$output_dir/metrics_post.json"

    # Kill server
    kill -9 $server_pid 2>/dev/null || true
    kill_leftover_processes
    wait_for_npu_idle

    log "  Completed: $output_dir"
    return 0
}

run_part_a() {
    log "=== Part A: KV Capacity Scan ==="
    log "  ${#KV_CAPACITIES[@]} capacities × ${#SCAN_WORKLOADS[@]} workloads × $REPS reps"

    for workload in "${SCAN_WORKLOADS[@]}"; do
        for kv_gib in "${KV_CAPACITIES[@]}"; do
            for rep in $(seq 1 "$REPS"); do
                local output_dir="$RESULT_DIR/raw_results/$workload/$kv_gib/rep-$rep"
                run_single_experiment \
                    "$workload" "$kv_gib" "$rep" "$output_dir" ""
            done
        done
    done
}

run_part_b() {
    log "=== Part B: Tiering Comparison ==="
    log "  ${#TIERING_CONFIGS[@]} configs × $REPS reps"
    log "  Workload: $TIERING_WORKLOAD (fixed 8 GiB KV for constrained configs)"

    for config in "${TIERING_CONFIGS[@]}"; do
        for rep in $(seq 1 "$REPS"); do
            local output_dir="$RESULT_DIR/tiering/$config/rep-$rep"
            local extra_env=""

            case "$config" in
                hbm-only)
                    # Use 32 GiB KV (no pressure, no preemption)
                    run_single_experiment \
                        "$TIERING_WORKLOAD" "32" "$rep" "$output_dir" ""
                    ;;
                kv-constrained)
                    # Use 8 GiB KV (pressure, standard preemption)
                    run_single_experiment \
                        "$TIERING_WORKLOAD" "8" "$rep" "$output_dir" ""
                    ;;
                kv-constrained-utility)
                    # Use 8 GiB KV + utility victim selection (BidKV)
                    run_single_experiment \
                        "$TIERING_WORKLOAD" "8" "$rep" "$output_dir" \
                        "VLLM_ASCEND_ENABLE_UTILITY_VICTIM_SELECTION=1"
                    ;;
            esac
        done
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log "Issue #134 KV Capacity Scan"
log "  Model: $MODEL_PATH"
log "  Python: $PYTHON"
log "  Result dir: $RESULT_DIR"
log "  Reps: $REPS"
log "  Part: $PART"
log "  ShareGPT dataset: ${SHAREGPT_DATASET:-NOT FOUND}"

if [ $DRY_RUN -eq 1 ]; then
    log "  DRY RUN mode"
fi

mkdir -p "$RESULT_DIR"

case "$PART" in
    A|a) run_part_a ;;
    B|b) run_part_b ;;
    both|BOTH)
        run_part_a
        run_part_b
        ;;
    *) echo "Invalid part: $PART"; exit 2 ;;
esac

log "=== All experiments complete ==="
log "Results: $RESULT_DIR"
log "Run analysis: python scripts/analyze_kv_capacity_scan.py --results-dir $RESULT_DIR"
