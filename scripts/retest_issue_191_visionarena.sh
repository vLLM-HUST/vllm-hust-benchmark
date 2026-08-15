#!/usr/bin/env bash
# =============================================================================
# Issue #191 re-test: visionarena-online single-card leaderboard jump.
#
# Re-tests the ec4847981f -> ceec19abb0 interval with 3 interleaved
# repetitions per side to determine whether the reported TPOT +71.4% jump is a
# reproducible code regression or original-data/environment noise.
#
# Design inherited from retest_issue_151_regression.sh, adapted for the
# visionarena-online multimodal workload (Qwen2.5-VL-7B-Instruct, 910B2
# single-card, openai-chat backend over the lmarena-ai/VisionArena-Chat HF
# dataset).
#
# Usage:
#   ./retest_issue_191_visionarena.sh [--reps N] [--dry-run]
#
# Retest contract (same metric/config contract as PR #165):
#   - median-based comparison
#   - 3 interleaved reps per side
#   - fixed NPU/model/CANN/torch_npu/dtype/graph mode/concurrency/RPS
#   - capture full dual-repo backend provenance (no backend=N/A)
#
# Acceptance thresholds (from issue #151/#165):
#   - TTFT/TPOT should not stably exceed 20% higher after the change
#   - Throughput should not stably decrease by more than 10%
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH="/data/shared_datasets/strict-models/Qwen2.5-VL-7B-Instruct"
VLLM_HUST_REPO="/root/vllm/vllm-hust"
ASCEND_REPO="/root/vllm/vllm-ascend-hust"
RESULT_DIR="/data/issue191-retest-results"
PYTHON="/root/miniconda3/envs/vllm-hust-dev/bin/python"
VISIONARENA_DATASET="/data/shared_datasets/strict-inputs/VisionArena-Chat"

# Fixed plugin commit (vllm-ascend-hust)
PLUGIN_COMMIT="312ca80a"

# Official fixed-target spec (official-ascend-jan-2026-v0.18.0-
# visionarena-online-qwen25-vl-7b-910b2)
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.6
DTYPE="float16"

# Server port (reused across runs with stop_server between them)
PORT=8000

# Server startup timeout (seconds)
SERVER_TIMEOUT=1200

# NPU device to run on (physical device index). Must be verified idle before
# launch. Selectable via --npu to avoid occupied devices on shared servers.
NPU_DEVICE=0

# Map NPU device index -> bus-id used by stop_server's read-only memory
# monitor (npu-smi reports bus-id per device).
_npu_bus_id() {
    case "$NPU_DEVICE" in
        0) echo "0000:C1:00.0" ;;
        1) echo "0000:01:00.0" ;;
        2) echo "0000:C2:00.0" ;;
        3) echo "0000:02:00.0" ;;
        4) echo "0000:81:00.0" ;;
        5) echo "0000:41:00.0" ;;
        6) echo "0000:82:00.0" ;;
        7) echo "0000:42:00.0" ;;
        *) echo "0000:C1:00.0" ;;
    esac
}

# (commit, workload) pairs for interleaved execution.
# Each pair maps to a specific workload — NOT a full cross-product.
#   visionarena-online:  ec4847981f -> ceec19abb0  (TPOT +71.4% jump)
COMMIT_WORKLOAD_PAIRS=(
    "ec4847981f:visionarena-online"
    "ceec19abb0:visionarena-online"
)

# ---------------------------------------------------------------------------
# Globals for PID tracking (triple verification: PID + starttime + cmdline)
# ---------------------------------------------------------------------------
_CURRENT_SERVER_PID=""
_LAUNCHER_PID=""
_LAUNCHER_STARTTIME=""
_LAUNCHER_CMDLINE_HASH=""
_VERIFIED_DESCENDANTS=()

# =============================================================================
# Helper functions
# =============================================================================

# 1. Timestamped logging to stderr.
log() {
    echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" >&2
}

# 1b. Verify the selected NPU is idle (memory below threshold) before launch.
#     Fail-closed: exit 1 if the device is occupied, to prevent resource
#     conflicts on shared servers.
verify_npu_idle() {
    local npu_bus_id
    npu_bus_id=$(_npu_bus_id)
    local npu_mem
    npu_mem=$(npu-smi info 2>/dev/null \
        | grep "$npu_bus_id" \
        | grep -o '[0-9]* / 65536' \
        | grep -o '^[0-9]*' || echo "65536")
    log "NPU $NPU_DEVICE ($npu_bus_id) memory: ${npu_mem} MB"
    if [[ "$npu_mem" -ge 5000 ]]; then
        log "ERROR: NPU $NPU_DEVICE is NOT idle (${npu_mem} MB used). Aborting to avoid resource conflict."
        return 1
    fi
    return 0
}

# 2. Get process start time from /proc/PID/stat field 22 (after comm).
#    macOS fallback via `ps -o lstart=`.
_get_starttime() {
    local pid="$1"
    if [[ -r "/proc/$pid/stat" ]]; then
        # /proc/PID/stat: pid (comm) state ppid ... field 22 = starttime.
        # comm can contain spaces/parens, so strip everything up to and
        # including the last ')'. After that, starttime is field 20
        # (22 - 2 for the stripped pid and comm fields).
        local stat_content
        stat_content=$(cat "/proc/$pid/stat" 2>/dev/null) || return 1
        local after_comm="${stat_content##*)}"
        echo "$after_comm" | awk '{print $20}'
    else
        # macOS fallback
        ps -o lstart= -p "$pid" 2>/dev/null | tr -s ' '
    fi
}

# 3. SHA-256 of /proc/PID/cmdline (null-separated).
#    macOS fallback via `ps -o args=`.
_get_cmdline_hash() {
    local pid="$1"
    if [[ -r "/proc/$pid/cmdline" ]]; then
        sha256sum "/proc/$pid/cmdline" 2>/dev/null | awk '{print $1}'
    else
        # macOS fallback
        ps -o args= -p "$pid" 2>/dev/null | sha256sum | awk '{print $1}'
    fi
}

# 4. Recursively snapshot children PID+starttime+cmdline into
#    _VERIFIED_DESCENDANTS.
_snapshot_descendants() {
    local pid="$1"
    _VERIFIED_DESCENDANTS=()
    _snapshot_descendants_recursive "$pid"
}

_snapshot_descendants_recursive() {
    local pid="$1"
    local children=()
    if [[ -r "/proc/$pid/task/$pid/children" ]]; then
        # Linux: /proc/PID/task/PID/children lists direct children PIDs
        local children_str
        children_str=$(cat "/proc/$pid/task/$pid/children" 2>/dev/null || true)
        read -ra children <<< "$children_str"
    else
        # Fallback: pgrep -P
        local child
        while IFS= read -r child; do
            [[ -n "$child" ]] && children+=("$child")
        done < <(pgrep -P "$pid" 2>/dev/null || true)
    fi

    local child
    for child in "${children[@]:-}"; do
        [[ -n "$child" ]] || continue
        local st ch
        st=$(_get_starttime "$child" 2>/dev/null) || continue
        ch=$(_get_cmdline_hash "$child" 2>/dev/null) || continue
        _VERIFIED_DESCENDANTS+=("$child:$st:$ch")
        _snapshot_descendants_recursive "$child"
    done
}

# 5. Verify PID still has same starttime AND cmdline hash.
#    Returns 0 if identity matches, 1 otherwise.
_verify_pid_identity() {
    local pid="$1"
    local recorded_starttime="$2"
    local recorded_cmdline_hash="$3"

    # Check if PID still exists
    if ! kill -0 "$pid" 2>/dev/null; then
        return 1
    fi

    local current_starttime current_cmdline_hash
    current_starttime=$(_get_starttime "$pid" 2>/dev/null) || return 1
    current_cmdline_hash=$(_get_cmdline_hash "$pid" 2>/dev/null) || return 1

    [[ "$current_starttime" == "$recorded_starttime" ]] || return 1
    [[ "$current_cmdline_hash" == "$recorded_cmdline_hash" ]] || return 1

    return 0
}

# 6. Kill launcher + descendants whose PID+starttime+cmdline still match.
#    Sends TERM, waits, then KILL any survivors.
kill_owned_server() {
    local pids=()

    # Add launcher PID if set and identity still matches
    if [[ -n "${_LAUNCHER_PID:-}" ]]; then
        if _verify_pid_identity \
            "$_LAUNCHER_PID" \
            "${_LAUNCHER_STARTTIME:-}" \
            "${_LAUNCHER_CMDLINE_HASH:-}"; then
            pids+=("$_LAUNCHER_PID")
        fi
    fi

    # Add verified descendants whose identity still matches
    local entry
    for entry in "${_VERIFIED_DESCENDANTS[@]:-}"; do
        [[ -n "$entry" ]] || continue
        local pid="${entry%%:*}"
        local rest="${entry#*:}"
        local recorded_st="${rest%%:*}"
        local recorded_ch="${rest#*:}"
        if _verify_pid_identity "$pid" "$recorded_st" "$recorded_ch"; then
            pids+=("$pid")
        fi
    done

    # Send TERM
    local pid
    for pid in "${pids[@]:-}"; do
        [[ -n "$pid" ]] || continue
        kill -TERM "$pid" 2>/dev/null || true
    done

    sleep 3

    # Send KILL to any still alive
    for pid in "${pids[@]:-}"; do
        [[ -n "$pid" ]] || continue
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    # Reset tracking state
    _CURRENT_SERVER_PID=""
    _LAUNCHER_PID=""
    _LAUNCHER_STARTTIME=""
    _LAUNCHER_CMDLINE_HASH=""
    _VERIFIED_DESCENDANTS=()
}

# Install trap so cleanup runs even if the script is interrupted (SIGINT,
# SIGTERM) or exits unexpectedly (per repo convention). kill_owned_server is
# identity-verified and only reaps the tracked server session/descendants.
trap kill_owned_server EXIT TERM INT

# 7. Combine `git diff HEAD` + untracked files, save to patch_file.
#    Returns SHA-256 of the patch file.
capture_patch_identity() {
    local repo="$1"
    local patch_file="$2"
    local temp_diff
    temp_diff=$(mktemp)

    # Capture tracked changes
    git -C "$repo" diff HEAD > "$temp_diff" 2>/dev/null || true

    # Capture untracked files
    local untracked
    untracked=$(git -C "$repo" ls-files --others --exclude-standard 2>/dev/null || true)
    local file
    for file in $untracked; do
        if [[ -f "$repo/$file" ]]; then
            echo "" >> "$temp_diff"
            echo "=== Untracked: $file ===" >> "$temp_diff"
            cat "$repo/$file" >> "$temp_diff" 2>/dev/null || true
        fi
    done

    cp "$temp_diff" "$patch_file"
    rm -f "$temp_diff"

    sha256sum "$patch_file" | awk '{print $1}'
}

# 8. Reset --hard, clean -fdx, checkout -f.
#    For ASCEND_REPO: call ensure_build_info + patch_triton_compat.
#    For VLLM_HUST_REPO: call patch_openai_conflict.
checkout_repo() {
    local repo="$1"
    local commit="$2"
    local name="$3"

    log "Checking out $name at $commit"
    git -C "$repo" reset --hard HEAD >/dev/null 2>&1 || true
    git -C "$repo" clean -fdx >/dev/null 2>&1 || true
    git -C "$repo" checkout -f "$commit" >/dev/null 2>&1

    if [[ "$repo" == "$ASCEND_REPO" ]]; then
        ensure_build_info
        patch_triton_compat
    elif [[ "$repo" == "$VLLM_HUST_REPO" ]]; then
        patch_openai_conflict
    fi

    clear_pycache
}

# 9. Create _build_info.py in ascend repo if missing.
#    Some ascend versions don't have this file, which causes import errors.
ensure_build_info() {
    local build_info_file="$ASCEND_REPO/vllm_ascend/_build_info.py"
    if [[ ! -f "$build_info_file" ]]; then
        log "Creating _build_info.py (missing in this ascend version)"
        mkdir -p "$(dirname "$build_info_file")"
        cat > "$build_info_file" << 'BUILD_INFO_EOF'
# Auto-generated fallback for missing _build_info.py
__version__ = "0.0.0"
__device_type__ = "A2"
__commit__ = ""
BUILD_INFO_EOF
    fi
}

# 10. Placeholder for triton compatibility patches (no-op for now).
patch_triton_compat() {
    true
}

# 11. Handle vllm.entrypoints.cli.openai vs openai library naming conflict.
#     If openai.py exists and openai_cmd.py doesn't, rename openai.py to
#     openai_cmd.py. Then sed main.py to update imports.
patch_openai_conflict() {
    local cli_dir="$VLLM_HUST_REPO/vllm/entrypoints/cli"
    local openai_py="$cli_dir/openai.py"
    local openai_cmd_py="$cli_dir/openai_cmd.py"

    if [[ -f "$openai_py" && ! -f "$openai_cmd_py" ]]; then
        log "Renaming openai.py -> openai_cmd.py to avoid openai library conflict"
        mv "$openai_py" "$openai_cmd_py"
    fi

    local main_py="$cli_dir/main.py"
    if [[ -f "$main_py" ]]; then
        sed -i \
            -e 's|import vllm\.entrypoints\.cli\.openai\b|import vllm.entrypoints.cli.openai_cmd|g' \
            -e 's|vllm\.entrypoints\.cli\.openai,|vllm.entrypoints.cli.openai_cmd,|g' \
            "$main_py"
    fi
}

# 12. Clear __pycache__ directories to avoid stale bytecode.
clear_pycache() {
    find "$VLLM_HUST_REPO" "$ASCEND_REPO" \
        -name __pycache__ -type d \
        -exec rm -rf {} + 2>/dev/null || true
}

# 13. Write env-manifest.json with full provenance information.
write_env_manifest() {
    local outdir="$1"
    local engine_short="$2"
    local engine_full="$3"
    local plugin_full="$4"

    local engine_patch_sha256=""
    local engine_patch_file=""
    local plugin_patch_sha256=""
    local plugin_patch_file=""

    if [[ -f "$outdir/engine.patch" ]]; then
        engine_patch_file="$outdir/engine.patch"
        engine_patch_sha256=$(sha256sum "$engine_patch_file" | awk '{print $1}')
    fi
    if [[ -f "$outdir/plugin.patch" ]]; then
        plugin_patch_file="$outdir/plugin.patch"
        plugin_patch_sha256=$(sha256sum "$plugin_patch_file" | awk '{print $1}')
    fi

    local python_version
    python_version=$("$PYTHON" --version 2>&1 || echo "unknown")

    local cann_version="unknown"
    local cann_version_file="/usr/local/Ascend/ascend-toolkit/latest/version.cfg"
    if [[ -f "$cann_version_file" ]]; then
        cann_version=$(cat "$cann_version_file" 2>/dev/null | head -1 || echo "unknown")
    fi

    local driver_version="unknown"
    if command -v npu-smi &>/dev/null; then
        driver_version=$(npu-smi info 2>/dev/null \
            | grep -i "driver" | head -1 || echo "unknown")
    fi

    local npu_device="${ASCEND_RT_VISIBLE_DEVICES:-0}"

    cat > "$outdir/env-manifest.json" << MANIFEST_EOF
{
    "engine_commit_requested": "$engine_short",
    "engine_commit_observed": "$engine_full",
    "engine_patch_sha256": "$engine_patch_sha256",
    "engine_patch_file": "$engine_patch_file",
    "plugin_commit_requested": "$PLUGIN_COMMIT",
    "plugin_commit_observed": "$plugin_full",
    "plugin_patch_sha256": "$plugin_patch_sha256",
    "plugin_patch_file": "$plugin_patch_file",
    "python_version": "$python_version",
    "cann_version": "$cann_version",
    "driver_version": "$driver_version",
    "npu_device": "$npu_device",
    "max_model_len": $MAX_MODEL_LEN,
    "gpu_memory_utilization": $GPU_MEM_UTIL,
    "dtype": "$DTYPE",
    "artifact_class": "diagnostic_historical_retest",
    "official_target": false,
    "note": "Diagnostic historical retest for issue #191 regression investigation. Results are NOT official benchmark results."
}
MANIFEST_EOF
}

# 14. Validate raw.json has a valid mean_ttft_ms > 0.
#     Online serving benchmark output from vllm bench serve has fields like
#     mean_ttft_ms, mean_tpot_ms, output_throughput.
validate_raw_json() {
    local raw_file="$1"
    local workload="$2"

    if [[ ! -f "$raw_file" ]]; then
        log "ERROR: raw.json not found at $raw_file"
        return 1
    fi

    "$PYTHON" -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
mt = data.get('mean_ttft_ms', 0)
if not isinstance(mt, (int, float)) or mt <= 0:
    print(f'ERROR: invalid mean_ttft_ms={mt}', file=sys.stderr)
    sys.exit(1)
tpot = data.get('mean_tpot_ms', 'N/A')
tput = data.get('output_throughput', 'N/A')
print(f'OK: mean_ttft_ms={mt:.2f} mean_tpot_ms={tpot} output_throughput={tput}')
" "$raw_file" || {
        log "ERROR: raw.json validation failed for $raw_file"
        return 1
    }
}

# 15. Start cmd in background, record PID+starttime, snapshot descendants,
#     wait for cmd to finish, return its exit code.
run_bench_tracked() {
    local log_file="$1"
    shift

    log "Starting tracked bench: $*"

    "$@" > "$log_file" 2>&1 &
    local bench_pid=$!
    local bench_starttime
    bench_starttime=$(_get_starttime "$bench_pid")
    local bench_cmdline_hash
    bench_cmdline_hash=$(_get_cmdline_hash "$bench_pid")

    _snapshot_descendants "$bench_pid"

    local exit_code=0
    wait "$bench_pid" || exit_code=$?

    log "Bench finished with exit code $exit_code"
    return "$exit_code"
}

# =============================================================================
# Server management (NEW for online serving, not in #146)
# =============================================================================

# 16. Start vllm server in background with start_new_session.
#     Poll /health until ready (max SERVER_TIMEOUT seconds).
#     Sets _CURRENT_SERVER_PID and snapshots descendants.
start_server() {
    local commit="$1"
    local port="$2"
    local server_log="$3"

    log "Starting vllm server (commit $commit) on port $port"

    # Ensure no stale process on this port
    local stale_pid
    stale_pid=$(lsof -ti :"$port" 2>/dev/null || true)
    if [[ -n "$stale_pid" ]]; then
        log "Killing stale process on port $port: $stale_pid"
        kill -9 "$stale_pid" 2>/dev/null || true
        sleep 3
    fi

    # Bypass proxy for localhost (server has http_proxy that breaks /health)
    export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
    export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

    # Start server with start_new_session (setsid creates a new session
    # so we can cleanly manage the process group).
    setsid "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port "$port" \
        --dtype "$DTYPE" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --limit-mm-per-prompt '{"image": 1}' \
        --enable-prefix-caching \
        > "$server_log" 2>&1 &

    _LAUNCHER_PID=$!
    _LAUNCHER_STARTTIME=$(_get_starttime "$_LAUNCHER_PID")
    _LAUNCHER_CMDLINE_HASH=$(_get_cmdline_hash "$_LAUNCHER_PID")
    log "Server launcher PID: $_LAUNCHER_PID"

    # Poll health endpoint
    local ready=0
    local elapsed=0
    local interval=5
    while (( elapsed < SERVER_TIMEOUT )); do
        # Check if launcher is still alive
        if ! kill -0 "$_LAUNCHER_PID" 2>/dev/null; then
            log "ERROR: Server process died. Check $server_log"
            tail -50 "$server_log" 2>/dev/null || true
            return 1
        fi

        local health_code
        health_code=$(LD_LIBRARY_PATH= curl -sf --noproxy "*" -o /dev/null -w "%{http_code}" \
            "http://127.0.0.1:$port/health" 2>/dev/null || echo "000")
        if [[ "$health_code" == "200" ]]; then
            ready=1
            log "Server ready after ${elapsed}s"
            break
        fi

        # Also check server log for startup completion
        if grep -q "Application startup complete" "$server_log" 2>/dev/null; then
            sleep 5
            elapsed=$((elapsed + 5))
            health_code=$(LD_LIBRARY_PATH= curl -sf --noproxy "*" -o /dev/null -w "%{http_code}" \
                "http://127.0.0.1:$port/health" 2>/dev/null || echo "000")
            if [[ "$health_code" == "200" ]]; then
                ready=1
                log "Server ready (startup complete + health=200) after ${elapsed}s"
                break
            fi
        fi

        log "  [${elapsed}s] health=$health_code, waiting..."
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done

    if [[ "$ready" -ne 1 ]]; then
        log "ERROR: Server failed to start within ${SERVER_TIMEOUT}s. Check $server_log"
        tail -80 "$server_log" 2>/dev/null || true
        return 1
    fi

    # Snapshot descendants (EngineCore subprocess, resource_tracker, etc.)
    _snapshot_descendants "$_LAUNCHER_PID"
    _CURRENT_SERVER_PID="$_LAUNCHER_PID"

    log "Server PID: $_LAUNCHER_PID, descendants: ${#_VERIFIED_DESCENDANTS[@]}"
    return 0
}

# 17. Stop the server. Cleanup is STRICTLY scoped to the tracked launcher
#     session (started via setsid) plus its identity-verified descendants.
#     No broad pkill patterns or /proc scans are used, so processes owned by
#     other tenants on a shared NPU server are never touched. Only the
#     read-only NPU memory wait is retained.
stop_server() {
    log "Stopping server..."

    # Capture tracked launcher identity before it is reset below.
    local launcher_pid="${_LAUNCHER_PID:-}"
    local launcher_st="${_LAUNCHER_STARTTIME:-}"
    local launcher_ch="${_LAUNCHER_CMDLINE_HASH:-}"

    # Scoped process-group kill: because the launcher was started with
    # setsid, its PGID equals its PID, so `kill -- -<pgid>` reaps the whole
    # tracked session (including any descendant not captured by the
    # recursive snapshot) without matching other processes by name.
    if [[ -n "$launcher_pid" ]] \
        && _verify_pid_identity "$launcher_pid" "$launcher_st" "$launcher_ch"; then
        local pgid
        pgid=$(ps -o pgid= -p "$launcher_pid" 2>/dev/null | tr -d ' ')
        if [[ -n "$pgid" && "$pgid" =~ ^[0-9]+$ ]]; then
            log "Killing tracked session/process group $pgid"
            kill -TERM -- "-$pgid" 2>/dev/null || true
            sleep 3
            kill -KILL -- "-$pgid" 2>/dev/null || true
        fi
    fi

    # Identity-verified cleanup of the launcher + descendants (also resets
    # tracking state).
    kill_owned_server

    # Wait for NPU memory to be released (up to 60s). Read-only monitor —
    # never kills other processes.
    local npu_bus_id
    npu_bus_id=$(_npu_bus_id)
    local wait_sec=0
    while (( wait_sec < 60 )); do
        local npu_mem
        npu_mem=$(npu-smi info 2>/dev/null \
            | grep "$npu_bus_id" \
            | grep -o '[0-9]* / 65536' \
            | grep -o '^[0-9]*' || echo "65536")
        if [[ "$npu_mem" -lt 5000 ]]; then
            log "NPU $NPU_DEVICE memory released: ${npu_mem} MB"
            break
        fi
        log "Waiting for NPU $NPU_DEVICE memory release: ${npu_mem} MB (${wait_sec}s)"
        sleep 5
        wait_sec=$((wait_sec + 5))
    done
}

# =============================================================================
# Benchmark functions
# =============================================================================

# 18. visionarena-online: backend=openai-chat,
#     endpoint=/v1/chat/completions, dataset=hf (VisionArena-Chat)
#     params: num_prompts=1000, request_rate=1, hf_split=train,
#             enable_multimodal_chat (official target client params)
run_visionarena_online() {
    local commit="$1"
    local rep="$2"
    local outdir="$3"
    local port="$4"

    local server_log="$outdir/server.log"
    local bench_log="$outdir/benchmark.log"

    log "=== visionarena-online: commit=$commit rep=$rep ==="

    # Start server
    if ! start_server "$commit" "$port" "$server_log"; then
        log "ERROR: failed to start server for visionarena-online"
        return 1
    fi

    # Run benchmark client
    local bench_exit=0
    run_bench_tracked "$bench_log" \
        "$PYTHON" -m vllm.entrypoints.cli.main bench serve \
            --backend openai-chat \
            --endpoint /v1/chat/completions \
            --host 127.0.0.1 \
            --port "$port" \
            --model "$MODEL_PATH" \
            --dataset-name hf \
            --dataset-path "$VISIONARENA_DATASET" \
            --hf-name lmarena-ai/VisionArena-Chat \
            --hf-split train \
            --num-prompts 1000 \
            --request-rate 1 \
            --enable-multimodal-chat \
            --save-result \
            --result-dir "$outdir" \
            --result-filename raw.json \
        || bench_exit=$?

    if [[ $bench_exit -ne 0 ]]; then
        log "ERROR: visionarena-online benchmark failed (exit $bench_exit)"
        log "=== Last 30 lines of bench log ==="
        tail -30 "$bench_log" 2>/dev/null || true
    fi

    # Stop server
    stop_server

    return "$bench_exit"
}

# 21. Clear stale artifacts, write run_info, checkout engine, run workload,
#     write env-manifest, touch .completed (ONLY after success — fail-closed).
run_single_benchmark() {
    local commit="$1"
    local workload="$2"
    local rep="$3"
    local plugin_full="$4"

    local outdir="$RESULT_DIR/$commit/$workload/rep-$rep"
    mkdir -p "$outdir"

    # Skip if already completed (resumable)
    if [[ -f "$outdir/.completed" ]]; then
        log "Already completed, skipping: $commit / $workload / rep-$rep"
        return 0
    fi

    # Clear stale artifacts (fail-closed: remove any previous .completed)
    rm -f "$outdir/raw.json" "$outdir/.completed" "$outdir/env-manifest.json"
    rm -f "$outdir/engine.patch" "$outdir/plugin.patch"

    # Write run_info.txt
    {
        echo "=== Issue #191 re-test ==="
        echo "timestamp=$(date -u +%Y%m%dT%H%M%SZ)"
        echo "engine_commit=$commit"
        echo "workload=$workload"
        echo "rep=$rep"
        echo "model_path=$MODEL_PATH"
        echo "max_model_len=$MAX_MODEL_LEN"
        echo "gpu_memory_utilization=$GPU_MEM_UTIL"
        echo "dtype=$DTYPE"
        echo "port=$PORT"
        echo "plugin_commit=$PLUGIN_COMMIT"
        echo "plugin_full=$plugin_full"
    } > "$outdir/run_info.txt"

    # Capture engine patch identity (before checkout — preserves any local
    # changes as provenance, then checkout -f discards them)
    capture_patch_identity "$VLLM_HUST_REPO" "$outdir/engine.patch" \
        > /dev/null

    # Checkout engine commit
    checkout_repo "$VLLM_HUST_REPO" "$commit" "vllm-hust"
    local engine_full
    engine_full=$(git -C "$VLLM_HUST_REPO" rev-parse HEAD)
    log "Engine at $engine_full (requested: $commit)"

    # Capture plugin patch identity
    capture_patch_identity "$ASCEND_REPO" "$outdir/plugin.patch" \
        > /dev/null

    # Run the appropriate workload
    local workload_exit=0
    if [[ "$workload" == "visionarena-online" ]]; then
        run_visionarena_online "$commit" "$rep" "$outdir" "$PORT" \
            || workload_exit=$?
    else
        log "ERROR: unknown workload '$workload'"
        return 1
    fi

    if [[ $workload_exit -ne 0 ]]; then
        log "ERROR: workload $workload failed for commit $commit rep $rep"
        return 1
    fi

    # Validate raw.json (fail-closed)
    if ! validate_raw_json "$outdir/raw.json" "$workload"; then
        log "ERROR: raw.json validation failed for $commit / $workload / rep-$rep"
        return 1
    fi

    # Write env-manifest
    write_env_manifest "$outdir" "$commit" "$engine_full" "$plugin_full"

    # Touch .completed (ONLY after success — fail-closed)
    touch "$outdir/.completed"
    log "SUCCESS: $commit / $workload / rep-$rep completed"

    return 0
}

# 20. Main entry point.
main() {
    # Parse args
    local reps=3
    local dry_run=false
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --reps)
                reps="$2"
                shift 2
                ;;
            --npu)
                NPU_DEVICE="$2"
                shift 2
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            *)
                log "ERROR: unknown argument '$1'"
                log "Usage: $0 [--reps N] [--npu DEVICE] [--dry-run]"
                exit 1
                ;;
        esac
    done

    log "========================================"
    log "Issue #191 re-test: visionarena-online"
    log "  Model:    $MODEL_PATH"
    log "  Engine:   $VLLM_HUST_REPO"
    log "  Plugin:   $ASCEND_REPO @ $PLUGIN_COMMIT"
    log "  Results:  $RESULT_DIR"
    log "  NPU:      $NPU_DEVICE"
    log "  Reps:     $reps"
    log "  Dry run:  $dry_run"
    log "========================================"

    if $dry_run; then
        log "DRY RUN — printing plan only"
        log ""
        log "Interleaved execution plan:"
        local rep pair
        for rep in $(seq 1 "$reps"); do
            log "  Rep $rep:"
            for pair in "${COMMIT_WORKLOAD_PAIRS[@]}"; do
                local commit="${pair%%:*}"
                local workload="${pair#*:}"
                log "    $commit / $workload"
            done
        done
        log ""
        log "Would source CANN env, checkout plugin at $PLUGIN_COMMIT,"
        log "then run $reps interleaved reps of ${#COMMIT_WORKLOAD_PAIRS[@]} pairs."
        log "Results would be saved to $RESULT_DIR"
        return 0
    fi

    # Verify selected NPU is idle (fail-closed)
    if ! verify_npu_idle; then
        return 1
    fi

    # Source CANN env
    if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
        log "Sourcing CANN environment"
        set +u; source /usr/local/Ascend/ascend-toolkit/set_env.sh; set -u
    else
        log "WARNING: CANN set_env.sh not found"
    fi

    # Export NPU env
    export ASCEND_RT_VISIBLE_DEVICES=$NPU_DEVICE
    export VLLM_USE_V1=1
    export VLLM_TARGET_DEVICE=npu
    export VLLM_ASCEND_TORCH_PREFLIGHT=0  # Skip preflight guard (verified torch_npu works; preflight takes ~22s > 20s timeout)
    export VLLM_PLUGINS=ascend
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    # VisionArena-Chat is ~82 GB across 43 parquet shards. Load it in offline
    # streaming mode with mmap disabled to avoid the known pyarrow futex
    # deadlock in the datasets library (see project lessons learned).
    export HF_DATASETS_OFFLINE=1
    export HF_DATASETS_DISABLE_MMAP=1
    export NO_PROXY="127.0.0.1,localhost"
    export no_proxy="127.0.0.1,localhost"

    # Export LD_LIBRARY_PATH (conda lib + ATB + CANN)
    local conda_lib
    conda_lib=$(dirname "$(dirname "$PYTHON")")/lib
    local atb_home="/usr/local/Ascend/nnal/atb/9.0.0/atb"
    local cxx_abi_dir
    cxx_abi_dir=$("$PYTHON" -c \
        "import torch; print('cxx_abi_1' if torch.compiled_with_cxx11_abi() else 'cxx_abi_0')" \
        2>/dev/null || echo "cxx_abi_0")
    local atb_lib_path="$atb_home/$cxx_abi_dir/lib"
    export LD_LIBRARY_PATH="$conda_lib:$atb_lib_path:/usr/local/Ascend/ascend-toolkit/lib64:/usr/local/Ascend/cann-9.0.0/lib64:${LD_LIBRARY_PATH:-}"
    export ATB_HOME_PATH="$atb_home/$cxx_abi_dir"

    log "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    log "ATB_HOME_PATH=$ATB_HOME_PATH"

    # Checkout plugin repo at PLUGIN_COMMIT
    log "Checking out plugin repo at $PLUGIN_COMMIT"
    checkout_repo "$ASCEND_REPO" "$PLUGIN_COMMIT" "vllm-ascend-hust"
    local plugin_full
    plugin_full=$(git -C "$ASCEND_REPO" rev-parse HEAD)
    log "Plugin at $plugin_full (requested: $PLUGIN_COMMIT)"

    # Interleaved execution: round-robin across reps
    local rep pair
    for rep in $(seq 1 "$reps"); do
        log ""
        log "======== Rep $rep / $reps ========"
        for pair in "${COMMIT_WORKLOAD_PAIRS[@]}"; do
            local commit="${pair%%:*}"
            local workload="${pair#*:}"

            log ""
            log "--- Rep $rep: $commit / $workload ---"

            # run_single_benchmark handles checkout + bench + validate +
            # env-manifest + .completed. If it fails, we continue to the
            # next pair (the .completed marker will be absent for failed
            # runs, so the analyzer skips them).
            run_single_benchmark "$commit" "$workload" "$rep" "$plugin_full" \
                || log "WARNING: $commit / $workload / rep-$rep failed"

            # Extra stop_server for safety (the workload function already
            # stops the server, but we double-check here)
            stop_server

            sleep 5
        done
    done

    # Restore repos to main
    log ""
    log "Restoring repos to main"
    git -C "$VLLM_HUST_REPO" checkout main >/dev/null 2>&1 || true
    git -C "$ASCEND_REPO" checkout main >/dev/null 2>&1 || true
    clear_pycache

    log ""
    log "========================================"
    log "Issue #191 re-test complete"
    log "Results: $RESULT_DIR"
    log "========================================"
    log ""
    log "Analyze with:"
    log "  $PYTHON scripts/analyze_issue_191_visionarena.py --result-dir $RESULT_DIR"
}

# Run main only when executed directly, so the script can be sourced in tests.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
