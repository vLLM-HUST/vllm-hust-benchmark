#!/bin/bash
# Layer 2 repetition campaign: run additional reps for current main on 6 NPUs in parallel.
# Each subshell gets its own environment. Rep 1 already exists in submissions/.
set -eo pipefail

REPO_ROOT="/root/vllm/vllm-hust-benchmark"
cd "$REPO_ROOT"

RESULT_BASE="/root/vllm/vllm-hust-benchmark/.benchmarks/repetition-current"
mkdir -p "$RESULT_BASE"

run_one() {
  local SPEC_NAME=$1
  local REP=$2
  local NPU_ID=$3
  local PORT=$4
  local SPEC_FILE=$5
  local MODEL_PATH=$6
  local RUN_ID="current-${SPEC_NAME}-rep${REP}"
  local RESULT_DIR="$RESULT_BASE/$RUN_ID"

  mkdir -p "$RESULT_DIR" \
    "/root/vllm/benchmark-cache/rep-cache-npu${NPU_ID}" \
    "/root/vllm/benchmark-cache/rep-tmp-npu${NPU_ID}"

  echo "[$(date '+%H:%M:%S')] Starting $SPEC_NAME rep$REP on NPU$NPU_ID port$PORT"

  (
    # Independent subshell with its own environment
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate vllm-hust-dev
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
    set -u

    export VLLM_HUST_WORKSPACE_ROOT=/root/vllm
    export CURRENT_ENV_PREFIX=/root/miniconda3/envs/vllm-hust-dev
    export CURRENT_VLLM_HUST_REPO=/root/vllm/vllm-hust
    export CURRENT_VLLM_ASCEND_HUST_REPO=/root/vllm/vllm-ascend-hust
    export CURRENT_GIT_COMMIT=e4ce33646f2ef1781289e6dc651fad0d00177c55
    export CURRENT_PLUGIN_GIT_COMMIT=03a12f9bddd944952bd029c6b62e23d68fa3a28e
    export VLLM_ENGINE_READY_TIMEOUT_S=1200
    export ASCEND_VISIBLE_DEVICES=$NPU_ID
    export ASCEND_RT_VISIBLE_DEVICES=$NPU_ID
    export CURRENT_SERVER_PORT=$PORT
    export CURRENT_CLIENT_PORT=$PORT
    export CURRENT_MODEL_PATH="$MODEL_PATH"
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_HOME=/root/.cache/huggingface
    export HF_HUB_CACHE=/root/.cache/huggingface/hub
    export CURRENT_VLLM_CACHE_ROOT=/root/vllm/benchmark-cache/rep-cache-npu${NPU_ID}
    export CURRENT_BENCHMARK_DATASET_ROOT=/root/vllm/benchmark-cache/current-benchmark-datasets
    export CURRENT_RUNTIME_CWD=/root/vllm/benchmark-cache/rep-tmp-npu${NPU_ID}
    export RESULT_DIR="$RESULT_DIR"

    bash scripts/run-current-ascend-same-spec.sh "$SPEC_FILE" \
      > "$RESULT_DIR/campaign.log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE $SPEC_NAME rep$REP (exit=$?)" >> "$RESULT_DIR/campaign.log"
  ) &
}

# Find model snapshot paths
CORE_TEXT_MODEL=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/*/  2>/dev/null | head -1 | sed 's:/$::')
CODER_MODEL=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-14B-Instruct/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
VISION_MODEL=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')

echo "Model paths:"
echo "  core-text-14b: $CORE_TEXT_MODEL"
echo "  coder-14b: $CODER_MODEL"
echo "  vision-7b: $VISION_MODEL"

# Verify models exist
[[ -f "$CORE_TEXT_MODEL/config.json" ]] || { echo "ERROR: core-text-14b model not found"; exit 1; }
[[ -f "$CODER_MODEL/config.json" ]] || { echo "ERROR: coder-14b model not found"; exit 1; }
[[ -f "$VISION_MODEL/config.json" ]] || { echo "ERROR: vision-7b model not found"; exit 1; }

# Launch rep 2 on NPUs 0-2, rep 3 on NPUs 3-5 (6 parallel)
NPU=0
PORT=8010
for REP in 2 3; do
  run_one "core-text-14b" $REP $NPU $PORT \
    "$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json" \
    "$CORE_TEXT_MODEL"
  NPU=$((NPU+1)); PORT=$((PORT+1))

  run_one "coder-14b" $REP $NPU $PORT \
    "$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0180-instructcoder-online-qwen25-coder-14b-910b2.json" \
    "$CODER_MODEL"
  NPU=$((NPU+1)); PORT=$((PORT+1))

  run_one "vision-7b" $REP $NPU $PORT \
    "$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0180-visionarena-online-qwen25-vl-7b-910b2.json" \
    "$VISION_MODEL"
  NPU=$((NPU+1)); PORT=$((PORT+1))
done

echo "[$(date '+%H:%M:%S')] All 6 jobs launched, waiting..."
wait
echo "[$(date '+%H:%M:%S')] All repetitions complete!"

echo ""
echo "=== Results ==="
find "$RESULT_BASE" -name "run_leaderboard.json" | sort
echo ""
echo "=== Status ==="
for d in "$RESULT_BASE"/*/; do
  if [[ -f "$d/submission/run_leaderboard.json" ]]; then
    echo "  OK: $(basename $d)"
  else
    echo "  FAIL: $(basename $d)"
    tail -3 "$d/campaign.log" 2>/dev/null
  fi
done
