#!/bin/bash
# Issue #146 regression re-test script.
# Runs sonnet-throughput and random-latency benchmarks at 3 vllm-hust commits
# with 3 repetitions each, using a fixed vllm-ascend-hust plugin commit.
#
# Usage:
#   ./retest_issue_146_regression.sh [--reps N] [--dry-run]
#
# Output:
#   /data/issue146-retest-results/<commit>/<workload>/rep-<N>/raw.json
#   /data/issue146-retest-results/summary.json

set -euo pipefail

REPS=${REPS:-3}
DRY_RUN=0

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

# Three engine commits from issue #146
ENGINE_COMMITS=("2206f1f7b7" "7a63f81e86" "83cf83ff20")

# Benchmark parameters (matching original backfill)
MAX_MODEL_LEN=30720
GPU_MEM_UTIL=0.6

# Export env for NPU access (matches backfill_single_gpu.py:_build_env)
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_VISIBLE_DEVICES=0
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

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

kill_leftover_processes() {
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "run_engine_core" 2>/dev/null || true
    pkill -9 -f "from multiprocessing.resource_tracker import main" 2>/dev/null || true
    pkill -9 -f "$MODEL_PATH" 2>/dev/null || true
    sleep 3
    # Kill any process on NPU 0
    local pids
    pids=$(npu-smi info 2>/dev/null | grep "| 0       0" | grep -o '| [0-9]*' | head -5 | tr -d '| ' || true)
    if [ -n "$pids" ]; then
        echo "  Killing leftover NPU 0 processes: $pids"
        for pid in $pids; do kill -9 "$pid" 2>/dev/null || true; done
        sleep 3
    fi
}

clear_pycache() {
    find "$VLLM_HUST_REPO" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$ASCEND_REPO" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
}

# Regenerate the auto-generated _build_info.py for vllm-ascend-hust.
# git checkout / clean removes it because it is created at build time by
# setup.py:gen_build_info().  Without it, vllm_ascend.utils fails with
# ImportError: cannot import name '_build_info' from 'vllm_ascend'.
ensure_build_info() {
    local build_info="$ASCEND_REPO/vllm_ascend/_build_info.py"
    echo "  Ensuring $build_info exists for 910B2 (A2)..."
    cat > "$build_info" <<'EOF'
# Auto-generated file
__device_type__ = 'A2'
EOF
}

# Patch triton-ascend API compatibility: vllm-ascend-hust plugin commit
# b2328661bd references triton.language.extra.ascend.libdevice.pow, but
# triton-ascend 3.5.0+ renamed the `ascend` extra submodule to `cann`.
# This patches the single offending line in penalties.py.
patch_triton_compat() {
    local penalties="$ASCEND_REPO/vllm_ascend/worker/v2/sample/penalties.py"
    if [ -f "$penalties" ] && grep -q 'triton.language.extra.ascend.libdevice' "$penalties"; then
        sed -i 's/triton\.language\.extra\.ascend\.libdevice/triton.language.extra.cann.libdevice/' "$penalties"
        echo "  Patched: triton.language.extra.ascend -> cann in $penalties"
    fi
}

# Fix the openai naming conflict: vllm/entrypoints/cli/openai.py shadows
# the pip-installed openai package, causing circular imports.  Rename it
# to openai_cmd.py and update main.py references (same fix as
# backfill_single_gpu.py).
patch_openai_conflict() {
    local cli_dir="$VLLM_HUST_REPO/vllm/entrypoints/cli"
    local openai_py="$cli_dir/openai.py"
    local openai_cmd_py="$cli_dir/openai_cmd.py"

    # If both exist, remove openai.py (openai_cmd.py is already the renamed copy).
    if [ -f "$openai_py" ] && [ -f "$openai_cmd_py" ]; then
        rm -f "$openai_py"
        echo "  Patched: removed duplicate $openai_py, keeping $openai_cmd_py"
    fi

    if [ -f "$openai_cmd_py" ] && [ ! -f "$openai_py" ]; then
        # Already renamed, just ensure main.py is up to date.
        :
    elif [ -f "$openai_py" ] && [ ! -f "$openai_cmd_py" ]; then
        mv "$openai_py" "$openai_cmd_py"
        echo "  Patched: renamed $openai_py -> $openai_cmd_py"
    fi

    # Update all references in main.py.
    local main_py="$cli_dir/main.py"
    if [ -f "$main_py" ]; then
        local content
        content=$(cat "$main_py")
        local orig="$content"
        content="${content//import vllm.entrypoints.cli.openai\n/import vllm.entrypoints.cli.openai_cmd\n}"
        # Use sed for the replacement since bash string replacement is tricky
        sed -i \
            -e 's/import vllm\.entrypoints\.cli\.openai$/import vllm.entrypoints.cli.openai_cmd/' \
            -e 's/vllm\.entrypoints\.cli\.openai,/vllm.entrypoints.cli.openai_cmd,/' \
            "$main_py" 2>/dev/null || true
        if [ "$content" != "$orig" ]; then
            echo "  Patched: updated imports in $main_py"
        fi
    fi
}

checkout_repo() {
    local repo="$1" commit="$2" name="$3"
    echo "  Checking out $name at $commit..."
    git -C "$repo" reset --hard HEAD --quiet 2>/dev/null || true
    git -C "$repo" clean -fdx --quiet 2>/dev/null || true
    # If checkout still fails due to untracked files, force remove them
    if ! git -C "$repo" checkout -f "$commit" 2>/dev/null; then
        echo "  Retrying with aggressive clean..."
        git -C "$repo" ls-files --others --ignored --exclude-standard -z 2>/dev/null | \
            xargs -0 rm -f 2>/dev/null || true
        git -C "$repo" checkout -f "$commit" 2>&1
    fi
    git -C "$repo" rev-parse HEAD
    # vllm-ascend-hust needs its build-time _build_info.py regenerated
    # after every checkout because clean -fdx removes it.
    if [ "$repo" = "$ASCEND_REPO" ]; then
        ensure_build_info
        patch_triton_compat
    fi
    # vllm-hust needs the openai.py naming conflict patched after every
    # checkout because reset --hard restores the original file names.
    if [ "$repo" = "$VLLM_HUST_REPO" ]; then
        patch_openai_conflict
    fi
}

run_sonnet_throughput() {
    local commit="$1" rep="$2" outdir="$3"
    local outfile="$outdir/raw.json"
    echo "  [sonnet-throughput] commit=$commit rep=$rep -> $outfile"

    $PYTHON -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL_PATH" \
        --dataset-name sonnet \
        --num-prompts 200 \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MAX_MODEL_LEN \
        --output-json "$outfile" \
        --dataset-path "$SONNET_DATASET" \
        2>&1 | tee "$outdir/bench.log" || true
}

run_random_latency() {
    local commit="$1" rep="$2" outdir="$3"
    local outfile="$outdir/raw.json"
    echo "  [random-latency] commit=$commit rep=$rep -> $outfile"

    $PYTHON -m vllm.entrypoints.cli.main bench latency \
        --model "$MODEL_PATH" \
        --input-len 1024 \
        --output-len 128 \
        --batch-size 8 \
        --num-iters-warmup 10 \
        --num-iters 30 \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MAX_MODEL_LEN \
        --output-json "$outfile" \
        2>&1 | tee "$outdir/bench.log" || true
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "========================================"
echo "Issue #146 Regression Re-test"
echo "Reps: $REPS"
echo "Engine commits: ${ENGINE_COMMITS[*]}"
echo "Plugin commit: $PLUGIN_COMMIT (fixed)"
echo "Model: $MODEL_PATH"
echo "Result dir: $RESULT_DIR"
echo "========================================"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] Would run $(( ${#ENGINE_COMMITS[@]} * 2 * REPS )) benchmarks"
    exit 0
fi

# Step 1: Fix plugin commit
echo ""
echo "=== Step 1: Checkout vllm-ascend-hust at $PLUGIN_COMMIT ==="
checkout_repo "$ASCEND_REPO" "$PLUGIN_COMMIT" "vllm-ascend-hust"
clear_pycache

# Step 2: Run benchmarks
for commit in "${ENGINE_COMMITS[@]}"; do
    echo ""
    echo "=== Engine commit: $commit ==="
    checkout_repo "$VLLM_HUST_REPO" "$commit" "vllm-hust"
    clear_pycache

    for workload in "sonnet-throughput" "random-latency"; do
        for rep in $(seq 1 "$REPS"); do
            outdir="$RESULT_DIR/$commit/$workload/rep-$rep"
            mkdir -p "$outdir"

            # Save run metadata
            {
                echo "engine_commit=$commit"
                echo "plugin_commit=$PLUGIN_COMMIT"
                echo "workload=$workload"
                echo "rep=$rep"
                echo "timestamp=$(date -u +%Y%m%dT%H%M%SZ)"
                echo "gpu_mem_util=$GPU_MEM_UTIL"
                echo "max_model_len=$MAX_MODEL_LEN"
            } > "$outdir/run_info.txt"

            kill_leftover_processes
            ensure_build_info

            if [ "$workload" = "sonnet-throughput" ]; then
                run_sonnet_throughput "$commit" "$rep" "$outdir"
            else
                run_random_latency "$commit" "$rep" "$outdir"
            fi

            sleep 5
        done
    done
done

# Step 3: Restore repos to main
echo ""
echo "=== Restoring repos to main ==="
git -C "$VLLM_HUST_REPO" checkout main --quiet 2>&1 || true
git -C "$ASCEND_REPO" checkout main --quiet 2>&1 || true

kill_leftover_processes

echo ""
echo "========================================"
echo "Re-test complete. Results in $RESULT_DIR"
echo "========================================"
