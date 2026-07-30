#!/usr/bin/env bash
set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────────────────
PY=/root/miniconda3/envs/vllm-hust-dev/bin/python
BENCH_REPO=/root/vllm/vllm-hust-benchmark
HUST_REPO=/root/vllm/vllm-hust
MODEL=Qwen/Qwen2.5-14B-Instruct
HUST_COMMIT=$(cd "$HUST_REPO" && git rev-parse HEAD)
ASCEND_COMMIT=$(cd /root/vllm/vllm-ascend-hust && git rev-parse HEAD)
TODAY=$(date -u +%Y%m%d)
RUN_PREFIX="single-gpu-backfill"

export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_USE_V1=1
export PYTHONDONTWRITEBYTECODE=1

echo "========================================================"
echo " vLLM-Hust commit : ${HUST_COMMIT:0:12}"
echo " Ascend  commit   : ${ASCEND_COMMIT:0:12}"
echo " Model            : $MODEL"
echo " NPU              : NPU 0 (single card)"
echo " Date             : $TODAY"
echo "========================================================"

mkdir -p "$BENCH_REPO/.benchmarks/runs"

# ── 辅助函数 ──────────────────────────────────────────────────────────────
submit_artifact() {
  local scenario=$1 result_file=$2 run_id=$3
  local input_len=${4:-} output_len=${5:-} batch_size=${6:-}

  local cmd_args=()
  [[ -n "$input_len"   ]] && cmd_args+=(--input-length "$input_len")
  [[ -n "$output_len"  ]] && cmd_args+=(--output-length "$output_len")
  [[ -n "$batch_size"  ]] && cmd_args+=(--batch-size "$batch_size")

  echo ">>> Submitting $scenario → submissions/$run_id/"
  cd "$BENCH_REPO"
  $PY -m vllm_hust_benchmark.cli submit "$scenario" \
    --benchmark-result-file "$result_file" \
    --constraints-file docs/examples/constraints_metrics.sample.json \
    --run-id "$run_id" \
    --engine vllm-hust \
    --engine-version "0.23.1.post1" \
    --model-name "$MODEL" \
    --model-parameters 14B \
    --model-precision FP16 \
    --hardware-vendor Huawei \
    --hardware-chip-model 910B2 \
    --chip-count 1 --node-count 1 \
    --submitter vllm-hust-org-member \
    --data-source vllm-hust-benchmark \
    --git-commit "$HUST_COMMIT" \
    --github-repository vllm-hust/vllm-hust \
    --github-ref "${HUST_COMMIT:0:10}" \
    --engine-source-repository vllm-hust/vllm-hust \
    --engine-source-ref "${HUST_COMMIT:0:10}" \
    --engine-source-commit "$HUST_COMMIT" \
    --plugin-source-engine vllm-ascend-hust \
    --plugin-source-repository vllm-hust/vllm-ascend-hust \
    --plugin-source-ref "${ASCEND_COMMIT:0:10}" \
    --plugin-source-commit "$ASCEND_COMMIT" \
    "${cmd_args[@]}"
}

# ── 1. random-latency ───────────────────────────────────────────────────
echo ""
echo "=== [1/3] random-latency ==="
RUN_ID="${RUN_PREFIX}-random-latency-${HUST_COMMIT:0:9}-${TODAY}"
RESULT_DIR="$BENCH_REPO/.benchmarks/runs/$RUN_ID"
mkdir -p "$RESULT_DIR"

cd "$HUST_REPO"
$PY -m vllm.entrypoints.cli.main bench latency \
  --model "$MODEL" \
  --input-len 1024 \
  --output-len 128 \
  --batch-size 8 \
  --num-iters-warmup 10 \
  --num-iters 30 \
  --output-json "$RESULT_DIR/latency.json" 2>&1 | tee "$RESULT_DIR/bench.log"

submit_artifact random-latency "$RESULT_DIR/latency.json" "$RUN_ID" 1024 128 8

# ── 2. sharegpt-throughput ──────────────────────────────────────────────
echo ""
echo "=== [2/3] sharegpt-throughput ==="
RUN_ID="${RUN_PREFIX}-sharegpt-throughput-${HUST_COMMIT:0:9}-${TODAY}"
RESULT_DIR="$BENCH_REPO/.benchmarks/runs/$RUN_ID"
mkdir -p "$RESULT_DIR"
SHAREGPT=/data/shared_datasets/vllm-hust-benchmark/current-benchmark-datasets/ShareGPT_V3_unfiltered_cleaned_split.json

cd "$HUST_REPO"
$PY -m vllm.entrypoints.cli.main bench throughput \
  --model "$MODEL" \
  --dataset-name sharegpt \
  --dataset-path "$SHAREGPT" \
  --num-prompts 200 \
  --output-json "$RESULT_DIR/throughput.json" 2>&1 | tee "$RESULT_DIR/bench.log"

submit_artifact sharegpt-throughput "$RESULT_DIR/throughput.json" "$RUN_ID"

# ── 3. sonnet-throughput ────────────────────────────────────────────────
echo ""
echo "=== [3/3] sonnet-throughput ==="
RUN_ID="${RUN_PREFIX}-sonnet-throughput-${HUST_COMMIT:0:9}-${TODAY}"
RESULT_DIR="$BENCH_REPO/.benchmarks/runs/$RUN_ID"
mkdir -p "$RESULT_DIR"
SONNET="$HUST_REPO/benchmarks/sonnet.txt"

cd "$HUST_REPO"
$PY -m vllm.entrypoints.cli.main bench throughput \
  --model "$MODEL" \
  --dataset-name sonnet \
  --dataset-path "$SONNET" \
  --num-prompts 200 \
  --output-json "$RESULT_DIR/throughput.json" 2>&1 | tee "$RESULT_DIR/bench.log"

submit_artifact sonnet-throughput "$RESULT_DIR/throughput.json" "$RUN_ID"

# ── 4. 聚合 ──────────────────────────────────────────────────────────────
echo ""
echo "=== Aggregating to leaderboard-data/snapshots/ ==="
cd "$BENCH_REPO"
$PY -m vllm_hust_benchmark.cli publish-website \
  --source-dir submissions \
  --output-dir leaderboard-data/snapshots \
  --execute

echo ""
echo "=== Validating snapshots ==="
$PY scripts/validate_public_leaderboard_snapshots.py

echo ""
echo "========================================================"
echo " DONE!  3 scenarios submitted for ${HUST_COMMIT:0:12}"
echo ""
echo " Next steps to push to remote:"
echo "   cd $BENCH_REPO"
echo "   git add submissions/ leaderboard-data/snapshots/"
echo "   git commit -m 'feat(leaderboard): add single-GPU data for ${HUST_COMMIT:0:12}'"
echo "   git push origin HEAD"
echo "========================================================"
