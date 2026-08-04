#!/bin/bash
# Issue #146 regression re-test script.
# Runs sonnet-throughput and random-latency benchmarks at 3 vllm-hust commits
# with 3 repetitions each, using a fixed vllm-ascend-hust plugin commit.
#
# Key design decisions per reviewer feedback (round 2):
#   1. Commits are interleaved (round-robin) across reps, not run sequentially.
#   2. Benchmark failures are NOT swallowed (no || true); temporary output is
#      atomically moved to the final location only after field validation.
#   3. Process cleanup is job-owned: the bench command runs in its own process
#      group (setsid) and _CURRENT_SERVER_PID tracks the real PGID so
#      kill_owned_server kills the entire process tree, not a no-op.
#   4. Provenance captures observed full SHA, full derived patch content
#      (tracked diff + untracked files like _build_info.py) saved to disk
#      and bound by SHA-256, plus Python/CANN/driver versions.
#   5. Each rep's final directory is cleared before the run and a .completed
#      marker is written only after successful validation, preventing stale
#      results from being consumed by the analysis if a rerun fails.
#   6. Results are diagnostic/historical re-test artifacts, NOT official targets.
#
# Usage:
#   ./retest_issue_146_regression.sh [--reps N] [--dry-run]
#
# Output:
#   /data/issue146-retest-results/<commit>/<workload>/rep-<N>/{raw.json,bench.log,run_info.txt,env-manifest.json,derived_patch.diff,.completed}

set -euo pipefail

NL=$'\n'

REPS=${REPS:-3}
DRY_RUN=0
NPU_DEVICE="${NPU_DEVICE:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reps) REPS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Configuration (matches original backfill parameters)
# ---------------------------------------------------------------------------

MODEL_PATH="/data/vllm-hust-benchmark-issue97/models/Qwen2.5-14B-Instruct"
VLLM_HUST_REPO="/root/vllm/vllm-hust"
ASCEND_REPO="/root/vllm/vllm-ascend-hust"
RESULT_DIR="/data/issue146-retest-results"
PYTHON="/root/miniconda3/envs/vllm-hust-dev/bin/python"
SONNET_DATASET="/root/vllm/vllm-hust/benchmarks/sonnet.txt"

# Fixed plugin commit (same as original July backfill)
PLUGIN_COMMIT="b2328661bd54079ce95eee78037ed9166d52e983"  # pragma: allowlist secret

# Three engine commits from issue #146 (full SHAs resolved at checkout)
ENGINE_COMMITS=("2206f1f7b7" "7a63f81e86" "83cf83ff20")

# Benchmark parameters (matching original backfill)
MAX_MODEL_LEN=30720
GPU_MEM_UTIL=0.6

# Export env for NPU access (matches backfill_single_gpu.py:_build_env)
export ASCEND_RT_VISIBLE_DEVICES=$NPU_DEVICE
export ASCEND_VISIBLE_DEVICES=$NPU_DEVICE
export VLLM_USE_V1=1
export VLLM_TARGET_DEVICE=npu
export VLLM_PLUGINS=ascend
export PYTHONDONTWRITEBYTECODE=1
export HF_ENDPOINT="https://hf-mirror.com"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"

# LD_LIBRARY_PATH: conda lib first (for CXXABI_1.3.15), then ATB, then CANN
_atb_home="/usr/local/Ascend/nnal/atb/9.0.0/atb"
_cxx_abi_dir="cxx_abi_1"
_conda_lib="$(dirname "$(dirname "$PYTHON")")/lib"
export LD_LIBRARY_PATH="${_conda_lib}:${_atb_home}/${_cxx_abi_dir}/lib:/usr/local/Ascend/ascend-toolkit/lib64:/usr/local/Ascend/cann-9.0.0/lib64:${LD_LIBRARY_PATH:-}"
export ATB_HOME_PATH="${_atb_home}/${_cxx_abi_dir}"

mkdir -p "$RESULT_DIR"

# Track verified descendant PIDs and their start times (in centiseconds).
# Populated by _snapshot_descendants() while the launcher is alive and the
# server is ready.  Used by kill_owned_server() to kill only processes whose
# PID AND start time still match — preventing PID-reuse false kills.
#
# Format: "pid1:starttime1 pid2:starttime2 ..."
_CURRENT_BENCH_PID=""
_VERIFIED_DESCENDANTS=""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%Y%m%dT%H%M%SZ')] $*"; }

# Get the start time of a PID (in centiseconds since boot, from /proc).
# Returns empty string if PID doesn't exist or /proc is unavailable.
_get_starttime() {
    local pid="$1"
    local stat_file="/proc/$pid/stat"
    if [ -r "$stat_file" ]; then
        # Field 22 is starttime in clock ticks.  Use a robust parse that
        # handles comm fields containing spaces/parens.
        local content
        content=$(cat "$stat_file" 2>/dev/null) || return
        local after_comm
        after_comm="${content##*)}"
        # after_comm now starts with " state ..." — field 22 from the start
        # of the full stat, but after the comm it's field 20.
        echo "$after_comm" | awk '{print $20}'
    else
        # macOS fallback: use ps -o lstart (less precise but works).
        ps -o lstart= -p "$pid" 2>/dev/null | tr ' ' '_' | tr -s '_'
    fi
}

# Recursively snapshot all descendants of a PID: record PID + start time.
# Called while the launcher is alive so the process tree is intact.
# Stores results in _VERIFIED_DESCENDANTS.
_snapshot_descendants() {
    local parent_pid="$1"
    local child
    for child in $(pgrep -P "$parent_pid" 2>/dev/null || true); do
        # Recurse first (depth-first) to capture grandchildren.
        _snapshot_descendants "$child"
        local st
        st=$(_get_starttime "$child")
        if [ -n "$st" ]; then
            _VERIFIED_DESCENDANTS+=" ${child}:${st}"
        fi
    done
}

# Verify that a PID still has the same start time as when we snapshotted it.
# Returns 0 (true) if PID exists and start time matches, 1 (false) otherwise.
# This prevents killing a recycled PID that now belongs to a different process.
_verify_pid() {
    local pid="$1" recorded_starttime="$2"
    local current_starttime
    current_starttime=$(_get_starttime "$pid")
    [ -n "$current_starttime" ] && [ "$current_starttime" = "$recorded_starttime" ]
}

# Job-owned cleanup: kill processes whose PID + start time match the
# snapshot taken while the launcher was alive.
# Per reviewer round 4: "在 launcher 存活、服务 ready 时持久化已验证后代的 PID
# 和 start time，cleanup 时以 PID/start time/cmdline 等身份再次核验后终止".
kill_owned_server() {
    # 1. Kill the launcher itself if still alive (it may have already exited).
    if [ -n "$_CURRENT_BENCH_PID" ]; then
        local launcher_st
        launcher_st=$(_get_starttime "$_CURRENT_BENCH_PID")
        if [ -n "$launcher_st" ]; then
            kill -TERM "$_CURRENT_BENCH_PID" 2>/dev/null || true
        fi
    fi

    # 2. Kill each verified descendant whose PID + start time still match.
    #    This avoids PID-reuse false kills: if a PID was recycled to a new
    #    process, its start time will differ and we skip it.
    if [ -n "$_VERIFIED_DESCENDANTS" ]; then
        local entry pid recorded_st
        for entry in $_VERIFIED_DESCENDANTS; do
            pid="${entry%%:*}"
            recorded_st="${entry##*:}"
            if _verify_pid "$pid" "$recorded_st"; then
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
        sleep 1
        # Force-kill any that survived TERM
        for entry in $_VERIFIED_DESCENDANTS; do
            pid="${entry%%:*}"
            recorded_st="${entry##*:}"
            if _verify_pid "$pid" "$recorded_st"; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
    fi

    _CURRENT_BENCH_PID=""
    _VERIFIED_DESCENDANTS=""
    sleep 2
}

# Install trap so cleanup runs even if the script is interrupted (SIGINT,
# SIGTERM) or exits unexpectedly.  Per reviewer round 3: "脚本没有 EXIT/TERM
# trap".
trap kill_owned_server EXIT TERM INT

clear_pycache() {
    find "$VLLM_HUST_REPO" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$ASCEND_REPO" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
}

# Regenerate the auto-generated _build_info.py for vllm-ascend-hust.
ensure_build_info() {
    local build_info="$ASCEND_REPO/vllm_ascend/_build_info.py"
    cat > "$build_info" <<'EOF'
# Auto-generated file
__device_type__ = 'A2'
EOF
}

# Patch triton-ascend API compatibility.
patch_triton_compat() {
    local penalties="$ASCEND_REPO/vllm_ascend/worker/v2/sample/penalties.py"
    if [ -f "$penalties" ] && grep -q 'triton.language.extra.ascend.libdevice' "$penalties"; then
        sed -i 's/triton\.language\.extra\.ascend\.libdevice/triton.language.extra.cann.libdevice/' "$penalties"
    fi
}

# Fix the openai naming conflict.
patch_openai_conflict() {
    local cli_dir="$VLLM_HUST_REPO/vllm/entrypoints/cli"
    local openai_py="$cli_dir/openai.py"
    local openai_cmd_py="$cli_dir/openai_cmd.py"

    if [ -f "$openai_py" ] && [ -f "$openai_cmd_py" ]; then
        rm -f "$openai_py"
    elif [ -f "$openai_py" ] && [ ! -f "$openai_cmd_py" ]; then
        mv "$openai_py" "$openai_cmd_py"
    fi

    local main_py="$cli_dir/main.py"
    if [ -f "$main_py" ]; then
        sed -i \
            -e 's/import vllm\.entrypoints\.cli\.openai$/import vllm.entrypoints.cli.openai_cmd/' \
            -e 's/vllm\.entrypoints\.cli\.openai,/vllm.entrypoints.cli.openai_cmd,/' \
            "$main_py" 2>/dev/null || true
    fi
}

# Capture the full derived patch content (tracked diff + untracked files)
# and save it to a .diff file.  Returns the SHA-256 of the combined content.
# This captures modifications that ensure_build_info (_build_info.py) and
# patch_triton_compat / patch_openai_conflict introduce, which a plain
# `git diff HEAD` would miss for untracked files.
capture_patch_identity() {
    local repo="$1"
    local patch_file="$2"
    local combined=""

    # 1. Tracked modifications (git diff HEAD)
    local tracked_diff
    tracked_diff=$(git -C "$repo" diff HEAD 2>/dev/null || true)

    # 2. Untracked files (e.g. _build_info.py generated by ensure_build_info)
    local untracked_files
    untracked_files=$(git -C "$repo" ls-files --others --exclude-standard 2>/dev/null || true)

    # Build the combined patch content
    if [ -n "$tracked_diff" ]; then
        combined+="=== TRACKED DIFF (git diff HEAD) ===${NL}"
        combined+="${tracked_diff}${NL}"
    fi

    if [ -n "$untracked_files" ]; then
        combined+="=== UNTRACKED FILES ===${NL}"
        local f
        while IFS= read -r f; do
            local filepath="$repo/$f"
            if [ -f "$filepath" ]; then
                combined+="--- /dev/null${NL}"
                combined+="+++ b/$f${NL}"
                combined+="$(cat "$filepath")${NL}"
            fi
        done <<< "$untracked_files"
    fi

    if [ -z "$combined" ]; then
        echo "clean" > "$patch_file"
        echo "clean"
    else
        # Save the full reproducible patch content to disk
        printf '%s' "$combined" > "$patch_file"
        # Return SHA-256 of the combined content
        printf '%s' "$combined" | sha256sum | awk '{print $1}'
    fi
}

checkout_repo() {
    local repo="$1" commit="$2" name="$3"
    log "  Checking out $name at $commit..."
    git -C "$repo" reset --hard HEAD --quiet 2>/dev/null || true
    git -C "$repo" clean -fdx --quiet 2>/dev/null || true
    if ! git -C "$repo" checkout -f "$commit" 2>/dev/null; then
        log "  Retrying with aggressive clean..."
        git -C "$repo" ls-files --others --ignored --exclude-standard -z 2>/dev/null | \
            xargs -0 rm -f 2>/dev/null || true
        git -C "$repo" checkout -f "$commit" 2>&1
    fi
    if [ "$repo" = "$ASCEND_REPO" ]; then
        ensure_build_info
        patch_triton_compat
    fi
    if [ "$repo" = "$VLLM_HUST_REPO" ]; then
        patch_openai_conflict
    fi
}

# Write a comprehensive env-manifest.json with full provenance.
# Saves the full derived patch content to derived_patch_{engine,plugin}.diff
# and binds it with SHA-256 (not MD5) so tracked AND untracked modifications
# (e.g. _build_info.py from ensure_build_info) are captured and reproducible.
write_env_manifest() {
    local outdir="$1"
    local engine_short="$2"
    local engine_full="$3"
    local plugin_full="$4"

    local engine_patch_file="$outdir/derived_patch_engine.diff"
    local plugin_patch_file="$outdir/derived_patch_plugin.diff"

    local engine_patch_sha256 plugin_patch_sha256
    engine_patch_sha256=$(capture_patch_identity "$VLLM_HUST_REPO" "$engine_patch_file")
    plugin_patch_sha256=$(capture_patch_identity "$ASCEND_REPO" "$plugin_patch_file")

    local python_version cann_version driver_version
    python_version=$("$PYTHON" --version 2>&1 | awk '{print $2}')
    cann_version=$(cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | head -1 || echo "unknown")
    driver_version=$(cat /usr/local/Ascend/driver/version.info 2>/dev/null | head -1 || echo "unknown")

    cat > "$outdir/env-manifest.json" <<EOF
{
  "engine_commit_requested": "$engine_short",
  "engine_commit_observed": "$engine_full",
  "engine_patch_sha256": "$engine_patch_sha256",
  "engine_patch_file": "derived_patch_engine.diff",
  "plugin_commit_requested": "$PLUGIN_COMMIT",
  "plugin_commit_observed": "$plugin_full",
  "plugin_patch_sha256": "$plugin_patch_sha256",
  "plugin_patch_file": "derived_patch_plugin.diff",
  "python_version": "$python_version",
  "cann_version": "$cann_version",
  "driver_version": "$driver_version",
  "npu_device": "$NPU_DEVICE",
  "max_model_len": "$MAX_MODEL_LEN",
  "gpu_memory_utilization": "$GPU_MEM_UTIL",
  "artifact_class": "diagnostic_historical_retest",
  "official_target": false,
  "note": "Re-test used max_model_len=30720 (original backfill value); official fixed-target spec requires 32768. Results are diagnostic only. Patch identity uses SHA-256 over tracked diff + untracked files."
}
EOF
}

# Validate that the raw.json contains a valid primary metric.
# Exits non-zero if validation fails.
validate_raw_json() {
    local raw_file="$1" workload="$2"
    "$PYTHON" -c "
import json, sys, math
with open('$raw_file') as f:
    raw = json.load(f)
if '$workload' == 'sonnet-throughput':
    val = raw.get('tokens_per_second') or raw.get('throughput_tps') or raw.get('tokens/s')
    if not val and isinstance(raw.get('throughput'), dict):
        val = raw['throughput'].get('tokens/s')
elif '$workload' == 'random-latency':
    val = raw.get('mean_ttft_ms') or raw.get('ttft_ms')
    if val is None:
        val = raw.get('avg_latency')
        if val is not None:
            val = float(val) * 1000.0
    if val is None:
        val = raw.get('p50')
else:
    print('Unknown workload: $workload', file=sys.stderr)
    sys.exit(1)
if val is None:
    print('Missing primary metric in $raw_file', file=sys.stderr)
    sys.exit(1)
val = float(val)
if not math.isfinite(val) or val <= 0:
    print(f'Invalid metric value {val} in $raw_file', file=sys.stderr)
    sys.exit(1)
print(f'Validated: {val}')
" || return 1
}

# Run a benchmark command, tracking its PID and snapshotting descendants.
# The command's stdout/stderr are tee'd to the log file.
# Sets _CURRENT_BENCH_PID to the launcher PID, and snapshots all descendant
# PIDs + start times into _VERIFIED_DESCENDANTS while the launcher is alive
# (so detached descendants like EngineCore are captured even if they later
# reparent or setsid).
# Exits non-zero if the benchmark command fails.
# Per reviewer round 4: "在 launcher 存活、服务 ready 时持久化已验证后代的 PID
# 和 start time，cleanup 时以 PID/start time/cmdline 等身份再次核验后终止".
run_bench_tracked() {
    local log_file="$1"
    shift

    # Reset state for this run.
    _CURRENT_BENCH_PID=""
    _VERIFIED_DESCENDANTS=""

    # Start the bench command.  NOT in a new session — we want it in our
    # process group so we can walk its descendants via pgrep -P.
    "$@" > >(tee "$log_file") 2>&1 &
    local bench_pid=$!
    _CURRENT_BENCH_PID=$bench_pid

    # Background monitor: continuously snapshot descendants while the launcher
    # is alive.  This captures EngineCore workers even if they later setsid
    # or reparent to init after the launcher exits.
    # The monitor writes to a temp file (append mode) to avoid subshell
    # variable scoping issues.
    local snapshot_file
    snapshot_file=$(mktemp)
    : > "$snapshot_file"  # truncate
    (
        while kill -0 "$bench_pid" 2>/dev/null; do
            _VERIFIED_DESCENDANTS=""
            _snapshot_descendants "$bench_pid"
            if [ -n "$_VERIFIED_DESCENDANTS" ]; then
                echo "$_VERIFIED_DESCENDANTS" >> "$snapshot_file"
            fi
            sleep 2
        done
        # Final snapshot right after launcher exits — children may still be
        # alive and traceable via pgrep before they reparent.
        _VERIFIED_DESCENDANTS=""
        _snapshot_descendants "$bench_pid"
        if [ -n "$_VERIFIED_DESCENDANTS" ]; then
            echo "$_VERIFIED_DESCENDANTS" >> "$snapshot_file"
        fi
    ) &
    local monitor_pid=$!

    # Wait for the bench command to finish; propagate its exit code.
    local rc=0
    wait "$bench_pid" || rc=$?

    # Stop the monitor and collect the accumulated snapshots.
    wait "$monitor_pid" 2>/dev/null || true
    # Read all snapshot lines, merge unique PID:starttime pairs.
    # Each line is a space-separated list of "pid:starttime" entries.
    _VERIFIED_DESCENDANTS=$(tr ' ' '\n' < "$snapshot_file" | sort -u | tr '\n' ' ')
    rm -f "$snapshot_file"

    return $rc
}

run_sonnet_throughput() {
    local commit="$1" rep="$2" outdir="$3"
    local tmpdir="$outdir/.tmp"
    local tmp_out="$tmpdir/raw.json"
    log "  [sonnet-throughput] commit=$commit rep=$rep"

    rm -rf "$tmpdir"
    mkdir -p "$tmpdir"

    # Run in tracked process group — no || true, failures propagate.
    run_bench_tracked "$tmpdir/bench.log" \
        "$PYTHON" -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL_PATH" \
        --dataset-name sonnet \
        --num-prompts 200 \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MAX_MODEL_LEN \
        --output-json "$tmp_out" \
        --dataset-path "$SONNET_DATASET"

    # Validate before atomic move
    if [ ! -f "$tmp_out" ]; then
        log "  ERROR: raw.json not produced for sonnet-throughput/$commit/rep-$rep"
        return 1
    fi
    validate_raw_json "$tmp_out" "sonnet-throughput" || return 1

    # Atomic move to final location
    mv "$tmp_out" "$outdir/raw.json"
    mv "$tmpdir/bench.log" "$outdir/bench.log"
    rm -rf "$tmpdir"
}

run_random_latency() {
    local commit="$1" rep="$2" outdir="$3"
    local tmpdir="$outdir/.tmp"
    local tmp_out="$tmpdir/raw.json"
    log "  [random-latency] commit=$commit rep=$rep"

    rm -rf "$tmpdir"
    mkdir -p "$tmpdir"

    # Run in tracked process group — no || true, failures propagate.
    run_bench_tracked "$tmpdir/bench.log" \
        "$PYTHON" -m vllm.entrypoints.cli.main bench latency \
        --model "$MODEL_PATH" \
        --input-len 1024 \
        --output-len 128 \
        --batch-size 8 \
        --num-iters-warmup 10 \
        --num-iters 30 \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MAX_MODEL_LEN \
        --output-json "$tmp_out"

    # Validate before atomic move
    if [ ! -f "$tmp_out" ]; then
        log "  ERROR: raw.json not produced for random-latency/$commit/rep-$rep"
        return 1
    fi
    validate_raw_json "$tmp_out" "random-latency" || return 1

    # Atomic move to final location
    mv "$tmp_out" "$outdir/raw.json"
    mv "$tmpdir/bench.log" "$outdir/bench.log"
    rm -rf "$tmpdir"
}

run_single_benchmark() {
    local commit="$1" workload="$2" rep="$3"
    local engine_full="$4" plugin_full="$5"
    local outdir="$RESULT_DIR/$commit/$workload/rep-$rep"

    # Clear stale final artifacts before starting this rep.
    # Per reviewer: "请在开始 rep 前清空最终目录" — if a previous run left
    # raw.json/bench.log/env-manifest.json and the new run fails, the analysis
    # must not read stale results.
    rm -f "$outdir/raw.json" "$outdir/bench.log" "$outdir/env-manifest.json" \
          "$outdir/run_info.txt" "$outdir/.completed" \
          "$outdir/derived_patch_engine.diff" "$outdir/derived_patch_plugin.diff"
    rm -rf "$outdir/.tmp"
    mkdir -p "$outdir"

    # Write run metadata
    cat > "$outdir/run_info.txt" <<EOF
engine_commit=$commit
plugin_commit=$PLUGIN_COMMIT
workload=$workload
rep=$rep
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
gpu_mem_util=$GPU_MEM_UTIL
max_model_len=$MAX_MODEL_LEN
npu_device=$NPU_DEVICE
artifact_class=diagnostic_historical_retest
official_target=false
EOF

    if [ "$DRY_RUN" -eq 1 ]; then
        log "  [DRY RUN] Skipping $workload/$commit/rep-$rep"
        # Dry-run writes manifest + completion marker so analysis can proceed
        write_env_manifest "$outdir" "$commit" "$engine_full" "$plugin_full"
        touch "$outdir/.completed"
        return 0
    fi

    kill_owned_server
    ensure_build_info

    if [ "$workload" = "sonnet-throughput" ]; then
        run_sonnet_throughput "$commit" "$rep" "$outdir"
    else
        run_random_latency "$commit" "$rep" "$outdir"
    fi

    # Write env-manifest.json with full provenance BEFORE the .completed marker.
    # Per reviewer round 3: "run_single_benchmark 在 write_env_manifest 之前就
    # 写入 .completed，后者失败时 marker 会保留" — this was fail-open.
    # Now: manifest must succeed first, then .completed is written.
    write_env_manifest "$outdir" "$commit" "$engine_full" "$plugin_full"

    # Write completion marker ONLY after successful benchmark + validation +
    # manifest write.  collect_results + validate_env_manifest enforce that
    # the manifest is valid before consuming results.
    touch "$outdir/.completed"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log "========================================"
log "Issue #146 Regression Re-test"
log "Reps: $REPS"
log "Engine commits: ${ENGINE_COMMITS[*]}"
log "Plugin commit: $PLUGIN_COMMIT (fixed)"
log "Model: $MODEL_PATH"
log "Result dir: $RESULT_DIR"
log "NPU device: $NPU_DEVICE"
log "Artifact class: diagnostic_historical_retest (NOT official target)"
log "========================================"

if [ "$DRY_RUN" -eq 1 ]; then
    log "[dry-run] Would run $(( ${#ENGINE_COMMITS[@]} * 2 * REPS )) benchmarks"
    exit 0
fi

# Step 1: Fix plugin commit and capture full SHA
log ""
log "=== Step 1: Checkout vllm-ascend-hust at $PLUGIN_COMMIT ==="
checkout_repo "$ASCEND_REPO" "$PLUGIN_COMMIT" "vllm-ascend-hust"
clear_pycache
PLUGIN_FULL_SHA=$(git -C "$ASCEND_REPO" rev-parse HEAD)
log "  Plugin observed full SHA: $PLUGIN_FULL_SHA"

# Step 2: Run benchmarks with INTERLEAVED execution (round-robin across reps)
# Per reviewer: "把三个 commit 交替执行，不要整段跑完一个 commit 后再跑另一个"
log ""
log "=== Step 2: Interleaved benchmark execution ==="
for rep in $(seq 1 "$REPS"); do
    for workload in "sonnet-throughput" "random-latency"; do
        for commit in "${ENGINE_COMMITS[@]}"; do
            log ""
            log "--- Rep $rep / $workload / commit $commit ---"

            # Checkout engine commit for this round
            checkout_repo "$VLLM_HUST_REPO" "$commit" "vllm-hust"
            clear_pycache
            ENGINE_FULL_SHA=$(git -C "$VLLM_HUST_REPO" rev-parse HEAD)

            # run_single_benchmark now writes env-manifest BEFORE .completed
            # so manifest failure prevents the completion marker (fail-closed).
            run_single_benchmark "$commit" "$workload" "$rep" \
                "$ENGINE_FULL_SHA" "$PLUGIN_FULL_SHA"

            kill_owned_server
            sleep 5
        done
    done
done

# Step 3: Restore repos to main
log ""
log "=== Restoring repos to main ==="
git -C "$VLLM_HUST_REPO" checkout main --quiet 2>&1 || true
git -C "$ASCEND_REPO" checkout main --quiet 2>&1 || true
kill_owned_server

log ""
log "========================================"
log "Re-test complete. Results in $RESULT_DIR"
log "Artifact class: diagnostic_historical_retest"
log "Run analysis: python scripts/analyze_issue_146_regression.py --result-dir $RESULT_DIR"
log "========================================"
