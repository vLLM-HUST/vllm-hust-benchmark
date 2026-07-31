#!/bin/bash
set -uo pipefail

# Source conda so prepare-official-ascend-baseline-env.sh can find it
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
fi
export PATH="/root/miniconda3/bin:/root/miniconda3/condabin:$PATH"

# Force offline mode: datasets are cached locally, prevent HF network access
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT="https://hf-mirror.com"

# Re-run v0.18.0 baseline 3x repeats for vision-7b and coder-14b only.
# core-text-14b already has 3 reps (with high error rate, to be addressed later).
# - coder-14b: NPUs 0,1,2 (3 parallel)
# - vision-7b: NPU 7 (sequential, 3 runs)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=${VLLM_HUST_HOST_WORKSPACE_ROOT:-/root/vllm}
RESULT_ROOT=${BASELINE_RESULT_ROOT:-"$REPO_ROOT/.benchmarks/baseline-rerun-vision-coder-$(date -u +%Y%m%dT%H%M%SZ)"}
RUNNER="$SCRIPT_DIR/run-official-ascend-goal-baseline.sh"
ENV_PREFIX=${GOAL_BASELINE_ENV_PREFIX:-/root/miniconda3/envs/vllm-ascend-official-v0180}
ASCEND_TOOLKIT_SET_ENV=${ASCEND_TOOLKIT_SET_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}
ASCEND_ATB_SET_ENV=${ASCEND_ATB_SET_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}
HF_HOME=${HF_HOME:-/root/.cache/huggingface}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
BASE_PORT=${BASE_PORT:-8440}

# Writable cache and dataset roots
CACHE_ROOT=${BASELINE_CACHE_ROOT:-"/root/vllm/baseline-cache"}
OFFICIAL_VLLM_CACHE_ROOT=${CACHE_ROOT}/vllm-cache
OFFICIAL_BENCHMARK_DATASET_ROOT=${CACHE_ROOT}/datasets
OFFICIAL_RUNTIME_CWD=${CACHE_ROOT}/runtime-cwd
mkdir -p "$OFFICIAL_VLLM_CACHE_ROOT" "$OFFICIAL_BENCHMARK_DATASET_ROOT" "$OFFICIAL_RUNTIME_CWD"

CODER_SPEC="$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0180-instructcoder-online-qwen25-coder-14b-910b2.json"
VISION_SPEC="$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0180-visionarena-online-qwen25-vl-7b-910b2.json"

CODER_MODEL="/data/shared_models/Qwen--Qwen2.5-Coder-14B-Instruct"
VISION_MODEL="/data/shared_models/Qwen--Qwen2.5-VL-7B-Instruct"

mkdir -p "$RESULT_ROOT"

RUN_ONE_PID=""
run_one() {
  local spec_name=$1
  local rep_index=$2
  local npu=$3
  local port=$4
  local spec_file=$5
  local model_path=$6
  local result_dir="$RESULT_ROOT/${spec_name}-rep${rep_index}"
  local run_id="baseline-${spec_name}-rep${rep_index}-$(date -u +%Y%m%dT%H%M%SZ)"
  local cache_root="$CACHE_ROOT/${spec_name}-rerun-rep${rep_index}"

  mkdir -p "$result_dir" "$cache_root" "$cache_root/runtime-cwd"

  echo "[rerun] launching ${spec_name} rep${rep_index} on NPU${npu} port${port} -> $result_dir"

  ASCEND_VISIBLE_DEVICES="$npu" \
  ASCEND_RT_VISIBLE_DEVICES="$npu" \
  GOAL_BASELINE_ENV_PREFIX="$ENV_PREFIX" \
  OFFICIAL_SERVER_PORT="$port" \
  OFFICIAL_CLIENT_PORT="$port" \
  RESULT_DIR="$result_dir" \
  RUN_ID="$run_id" \
  VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
  VLLM_HUST_HOST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
  ASCEND_TOOLKIT_SET_ENV="$ASCEND_TOOLKIT_SET_ENV" \
  ASCEND_ATB_SET_ENV="$ASCEND_ATB_SET_ENV" \
  OFFICIAL_MODEL_PATH="$model_path" \
  HF_HOME="$HF_HOME" \
  HF_HUB_CACHE="$HF_HUB_CACHE" \
  TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  OFFICIAL_VLLM_CACHE_ROOT="$cache_root/vllm-cache" \
  OFFICIAL_BENCHMARK_DATASET_ROOT="$OFFICIAL_BENCHMARK_DATASET_ROOT" \
  OFFICIAL_RUNTIME_CWD="$cache_root/runtime-cwd" \
  MANAGED_SERVER_PORT_FILE="" \
  MANAGED_SERVER_WRAPPER_PID_FILE="" \
  MANAGED_SERVER_LISTENER_PIDS_FILE="" \
  SKIP_OFFICIAL_ASCEND_C_EXTENSION_BUILD=1 \
  OFFICIAL_VLLM_WORKTREE=/tmp/vllm-v0180 \
  OFFICIAL_VLLM_ASCEND_WORKTREE=/tmp/vllm-ascend-v0180 \
  OFFICIAL_VLLM_REPO="$WORKSPACE_ROOT/reference-repos/vllm" \
  OFFICIAL_VLLM_ASCEND_REPO="$WORKSPACE_ROOT/reference-repos/vllm-ascend" \
  bash "$RUNNER" "$spec_file" >"$result_dir/runner.log" 2>&1 &
  RUN_ONE_PID=$!
}

# Launch coder-14b on NPUs 0,1,2
CODER_PIDS=()
for REP in 1 2 3; do
  NPU=$((REP - 1))
  PORT=$((BASE_PORT + NPU))
  run_one "coder-14b" "$REP" "$NPU" "$PORT" "$CODER_SPEC" "$CODER_MODEL"
  CODER_PIDS+=("$RUN_ONE_PID")
done

# Launch vision-7b on NPU 7 (sequential - 3 runs)
VISION_PORT=$((BASE_PORT + 7))
run_vision_sequential() {
  for REP in 1 2 3; do
    local result_dir="$RESULT_ROOT/vision-7b-rep${REP}"
    local run_id="baseline-vision-7b-rep${REP}-$(date -u +%Y%m%dT%H%M%SZ)"
    local cache_root="$CACHE_ROOT/vision-7b-rerun-rep${REP}"
    mkdir -p "$result_dir" "$cache_root" "$cache_root/runtime-cwd"
    echo "[rerun] launching vision-7b rep${REP} on NPU7 port${VISION_PORT} -> $result_dir" >&2
    ASCEND_VISIBLE_DEVICES=7 \
    ASCEND_RT_VISIBLE_DEVICES=7 \
    GOAL_BASELINE_ENV_PREFIX="$ENV_PREFIX" \
    OFFICIAL_SERVER_PORT="$VISION_PORT" \
    OFFICIAL_CLIENT_PORT="$VISION_PORT" \
    RESULT_DIR="$result_dir" \
    RUN_ID="$run_id" \
    VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
    VLLM_HUST_HOST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
    ASCEND_TOOLKIT_SET_ENV="$ASCEND_TOOLKIT_SET_ENV" \
    ASCEND_ATB_SET_ENV="$ASCEND_ATB_SET_ENV" \
    OFFICIAL_MODEL_PATH="$VISION_MODEL" \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
    TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OFFICIAL_VLLM_CACHE_ROOT="$cache_root/vllm-cache" \
    OFFICIAL_BENCHMARK_DATASET_ROOT="$OFFICIAL_BENCHMARK_DATASET_ROOT" \
    OFFICIAL_RUNTIME_CWD="$cache_root/runtime-cwd" \
    MANAGED_SERVER_PORT_FILE="" \
    MANAGED_SERVER_WRAPPER_PID_FILE="" \
    MANAGED_SERVER_LISTENER_PIDS_FILE="" \
    SKIP_OFFICIAL_ASCEND_C_EXTENSION_BUILD=1 \
    OFFICIAL_VLLM_WORKTREE=/tmp/vllm-v0180 \
    OFFICIAL_VLLM_ASCEND_WORKTREE=/tmp/vllm-ascend-v0180 \
    OFFICIAL_VLLM_REPO="$WORKSPACE_ROOT/reference-repos/vllm" \
    OFFICIAL_VLLM_ASCEND_REPO="$WORKSPACE_ROOT/reference-repos/vllm-ascend" \
    bash "$RUNNER" "$VISION_SPEC" >"$result_dir/runner.log" 2>&1
    echo "[rerun] vision-7b rep${REP} exit=$?" >&2
  done
}

run_vision_sequential &
VISION_PID=$!

# Wait for all parallel tasks
echo "[rerun] waiting for coder-14b parallel runs..."
FAILED=0
for PID in "${CODER_PIDS[@]}"; do
  if ! wait "$PID"; then
    echo "[rerun] PID $PID failed"
    FAILED=$((FAILED + 1))
  fi
done

echo "[rerun] waiting for vision-7b sequential runs..."
if ! wait "$VISION_PID"; then
  echo "[rerun] vision-7b had failures"
  FAILED=$((FAILED + 1))
fi

echo "[rerun] completed with $FAILED failures"
echo "[rerun] results at: $RESULT_ROOT"

# Summary
echo "[rerun] === SUMMARY ==="
for SPEC in coder-14b vision-7b; do
  for REP in 1 2 3; do
    DIR="$RESULT_ROOT/${SPEC}-rep${REP}"
    if [[ -f "$DIR/raw_benchmark_result.json" ]]; then
      TPS=$(python3 -c "import json; d=json.load(open('$DIR/raw_benchmark_result.json')); print(d.get('output_throughput','N/A'))" 2>/dev/null || echo "N/A")
      echo "  $SPEC rep$REP: BENCHMARK DONE (tps=$TPS)"
    elif [[ -f "$DIR/runner.log" ]]; then
      echo "  $SPEC rep$REP: RUNNER FINISHED (no result)"
    else
      echo "  $SPEC rep$REP: NOT STARTED"
    fi
  done
done
