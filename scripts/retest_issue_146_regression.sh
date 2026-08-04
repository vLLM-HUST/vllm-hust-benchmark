#!/bin/bash
# Issue #146 regression re-test script.
# Runs sonnet-throughput and random-latency benchmarks at 3 vllm-hust commits
# with 3 repetitions each, using a fixed vllm-ascend-hust plugin commit.
#
# Key design decisions per reviewer feedback:
#   1. Commits are interleaved (round-robin) across reps, not run sequentially.
#   2. Benchmark failures are NOT swallowed (no || true); temporary output is
#      atomically moved to the final location only after field validation.
#   3. Process cleanup is job-owned (tracks the server PID it started) rather
#      than global pkill.
#   4. Provenance captures observed full SHA, dirty diff/patch identity, and
#      Python/CANN/driver versions in an env-manifest.json.
#   5. Results are diagnostic/historical re-test artifacts, NOT official targets.
#
# Usage:
#   ./retest_issue_146_regression.sh [--reps N] [--dry-run]
#
# Output:
#   /data/issue146-retest-results/<commit>/<workload>/rep-<N>/{raw.json,bench.log,run_info.txt,env-manifest.json}

set -euo pipefail

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

# Track the server PID we start so cleanup is job-owned.
_CURRENT_SERVER_PID=""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%Y%m%dT%H%M%SZ')] $*"; }

# Job-owned cleanup: only kill processes we started, not global pkill.
kill_owned_server() {
    if [ -n "$_CURRENT_SERVER_PID" ]; then
        # Kill the server process and its children
        kill -9 "$_CURRENT_SERVER_PID" 2>/dev/null || true
        # Kill children (engine core workers spawned by the server)
        local child
        for child in $(pgrep -P "$_CURRENT_SERVER_PID" 2>/dev/null || true); do
            kill -9 "$child" 2>/dev/null || true
        done
        _CURRENT_SERVER_PID=""
        sleep 2
    fi
}

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

# Capture the dirty diff / patch identity after applying derived patches.
# Returns a short hash identifying the working-tree modification state.
capture_patch_identity() {
    local repo="$1"
    local diff
    diff=$(git -C "$repo" diff HEAD 2>/dev/null || true)
    if [ -z "$diff" ]; then
        echo "clean"
    else
        # Use md5sum of the diff as a patch identity
        echo "$diff" | md5sum | awk '{print $1}'
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
write_env_manifest() {
    local outdir="$1"
    local engine_short="$2"
    local engine_full="$3"
    local plugin_full="$4"

    local engine_patch plugin_patch
    engine_patch=$(capture_patch_identity "$VLLM_HUST_REPO")
    plugin_patch=$(capture_patch_identity "$ASCEND_REPO")

    local python_version cann_version driver_version
    python_version=$("$PYTHON" --version 2>&1 | awk '{print $2}')
    cann_version=$(cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | head -1 || echo "unknown")
    driver_version=$(cat /usr/local/Ascend/driver/version.info 2>/dev/null | head -1 || echo "unknown")

    cat > "$outdir/env-manifest.json" <<EOF
{
  "engine_commit_requested": "$engine_short",
  "engine_commit_observed": "$engine_full",
  "engine_patch_identity": "$engine_patch",
  "plugin_commit_requested": "$PLUGIN_COMMIT",
  "plugin_commit_observed": "$plugin_full",
  "plugin_patch_identity": "$plugin_patch",
  "python_version": "$python_version",
  "cann_version": "$cann_version",
  "driver_version": "$driver_version",
  "npu_device": "$NPU_DEVICE",
  "max_model_len": "$MAX_MODEL_LEN",
  "gpu_memory_utilization": "$GPU_MEM_UTIL",
  "artifact_class": "diagnostic_historical_retest",
  "official_target": false,
  "note": "Re-test used max_model_len=30720 (original backfill value); official fixed-target spec requires 32768. Results are diagnostic only."
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

run_sonnet_throughput() {
    local commit="$1" rep="$2" outdir="$3"
    local tmpdir="$outdir/.tmp"
    local tmp_out="$tmpdir/raw.json"
    log "  [sonnet-throughput] commit=$commit rep=$rep"

    rm -rf "$tmpdir"
    mkdir -p "$tmpdir"

    # No || true: let failures propagate
    $PYTHON -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL_PATH" \
        --dataset-name sonnet \
        --num-prompts 200 \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MAX_MODEL_LEN \
        --output-json "$tmp_out" \
        --dataset-path "$SONNET_DATASET" \
        2>&1 | tee "$tmpdir/bench.log"

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

    # No || true: let failures propagate
    $PYTHON -m vllm.entrypoints.cli.main bench latency \
        --model "$MODEL_PATH" \
        --input-len 1024 \
        --output-len 128 \
        --batch-size 8 \
        --num-iters-warmup 10 \
        --num-iters 30 \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MAX_MODEL_LEN \
        --output-json "$tmp_out" \
        2>&1 | tee "$tmpdir/bench.log"

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
    local outdir="$RESULT_DIR/$commit/$workload/rep-$rep"
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
        return 0
    fi

    kill_owned_server
    ensure_build_info

    if [ "$workload" = "sonnet-throughput" ]; then
        run_sonnet_throughput "$commit" "$rep" "$outdir"
    else
        run_random_latency "$commit" "$rep" "$outdir"
    fi
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

            run_single_benchmark "$commit" "$workload" "$rep"

            # Write env manifest with full provenance
            write_env_manifest "$RESULT_DIR/$commit/$workload/rep-$rep" \
                "$commit" "$ENGINE_FULL_SHA" "$PLUGIN_FULL_SHA"

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
