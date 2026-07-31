#!/bin/bash
set -uo pipefail

# Re-run vision-7b rep2 and rep3 after coder-14b has completed.
# rep2/3 failed earlier due to admission check detecting coder-14b processes.

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
fi
export PATH="/root/miniconda3/bin:/root/miniconda3/condabin:$PATH"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT="https://hf-mirror.com"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=${VLLM_HUST_HOST_WORKSPACE_ROOT:-/root/vllm}
RESULT_ROOT=${BASELINE_RESULT_ROOT:-"$REPO_ROOT/.benchmarks/baseline-rerun-vision-coder-20260731T020724Z"}
RUNNER="$SCRIPT_DIR/run-official-ascend-goal-baseline.sh"
ENV_PREFIX=${GOAL_BASELINE_ENV_PREFIX:-/root/miniconda3/envs/vllm-ascend-official-v0180}
ASCEND_TOOLKIT_SET_ENV=${ASCEND_TOOLKIT_SET_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}
ASCEND_ATB_SET_ENV=${ASCEND_ATB_SET_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}
HF_HOME=${HF_HOME:-/root/.cache/huggingface}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
BASE_PORT=${BASE_PORT:-8440}

CACHE_ROOT=${BASELINE_CACHE_ROOT:-"/root/vllm/baseline-cache"}
OFFICIAL_BENCHMARK_DATASET_ROOT=${CACHE_ROOT}/datasets
mkdir -p "$OFFICIAL_BENCHMARK_DATASET_ROOT"

VISION_SPEC="$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0180-visionarena-online-qwen25-vl-7b-910b2.json"
VISION_MODEL="/data/shared_models/Qwen--Qwen2.5-VL-7B-Instruct"
VISION_PORT=$((BASE_PORT + 7))

# Wait for coder-14b to finish (no residual benchmark processes)
echo "[rerun-vision] waiting for coder-14b to complete..."
while true; do
  residual=$(ps -ef | grep run_vllm_cli_compat.py | grep -v grep | wc -l)
  if [ "$residual" -eq 0 ]; then
    echo "[rerun-vision] no residual benchmark processes, proceeding"
    break
  fi
  echo "[rerun-vision] still $residual residual processes, waiting 30s..."
  sleep 30
done

# Run vision-7b rep2 and rep3 sequentially on NPU 7
for REP in 2 3; do
  result_dir="$RESULT_ROOT/vision-7b-rep${REP}"
  run_id="baseline-vision-7b-rep${REP}-$(date -u +%Y%m%dT%H%M%SZ)"
  cache_root="$CACHE_ROOT/vision-7b-rerun-rep${REP}"
  mkdir -p "$result_dir" "$cache_root" "$cache_root/runtime-cwd"

  echo "[rerun-vision] launching vision-7b rep${REP} on NPU7 port${VISION_PORT} -> $result_dir"

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
  echo "[rerun-vision] vision-7b rep${REP} exit=$?"
done

echo "[rerun-vision] all vision-7b reps complete"
