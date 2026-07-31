#!/bin/bash
set -euo pipefail

# Measure SimLLM warm-cache throughput under a saturated online workload.
#
# Unlike the official online spec (request_rate=1), the measured pass uses
# request_rate=inf and a bounded concurrency. This removes the client-side
# arrival-rate ceiling while avoiding an unbounded number of in-flight requests.
#
# The SimLLM case keeps the existing warm-cache semantics:
#   1. Start one SimLLM-enabled server.
#   2. Send the same deterministic prompts once at a low request rate.
#   3. Without restarting the server, send those prompts at request_rate=inf.
#
# The baseline uses the same measured workload, seed, and concurrency with
# SimLLM disabled. At the end, a compact throughput comparison is written to
# throughput_comparison.json and throughput_comparison.md.
#
# Example:
#   cd /workspace/vllm-hust-benchmark
#   ASCEND_RT_VISIBLE_DEVICES=6 \
#   CURRENT_MODEL_PATH=/data/shared_models/Qwen2.5-14B-Instruct \
#   bash scripts/run_simllm_saturated_throughput_warm_cache.sh
#
# Useful overrides:
#   SIMLLM_THROUGHPUT_MAX_CONCURRENCY=32
#   SIMLLM_THROUGHPUT_NUM_PROMPTS=128
#   SIMLLM_THROUGHPUT_INPUT_LEN=4096
#   SIMLLM_THROUGHPUT_OUTPUT_LEN=32

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=${VLLM_HUST_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}
WARM_CACHE_RUNNER=${WARM_CACHE_RUNNER:-"$SCRIPT_DIR/run_simllm_random_online_warm_cache.sh"}
DEFAULT_BASE_SPEC_FILE="$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json"

BASE_SPEC_FILE=${1:-${BASE_SPEC_FILE:-$DEFAULT_BASE_SPEC_FILE}}
RESULT_DIR=${RESULT_DIR:-"$REPO_ROOT/.benchmarks/simllm-saturated-throughput-warm-cache"}
CURRENT_DATA_SOURCE=${CURRENT_DATA_SOURCE:-"vllm-ascend-hust-ci-simllm-saturated-throughput"}
CURRENT_VLLM_ASCEND_HUST_REPO=${CURRENT_VLLM_ASCEND_HUST_REPO:-"$WORKSPACE_ROOT/vllm-ascend-hust"}

SIMLLM_THROUGHPUT_NUM_PROMPTS=${SIMLLM_THROUGHPUT_NUM_PROMPTS:-32}
SIMLLM_THROUGHPUT_INPUT_LEN=${SIMLLM_THROUGHPUT_INPUT_LEN:-4096}
SIMLLM_THROUGHPUT_OUTPUT_LEN=${SIMLLM_THROUGHPUT_OUTPUT_LEN:-32}
SIMLLM_THROUGHPUT_MAX_CONCURRENCY=${SIMLLM_THROUGHPUT_MAX_CONCURRENCY:-16}
SIMLLM_THROUGHPUT_MAX_NUM_BATCHED_TOKENS=${SIMLLM_THROUGHPUT_MAX_NUM_BATCHED_TOKENS:-$SIMLLM_THROUGHPUT_INPUT_LEN}
SIMLLM_THROUGHPUT_SCENARIO=${SIMLLM_THROUGHPUT_SCENARIO:-random-online}
SIMLLM_THROUGHPUT_ENDPOINT=${SIMLLM_THROUGHPUT_ENDPOINT:-/v1/completions}
SIMLLM_THROUGHPUT_DATASET_NAME=${SIMLLM_THROUGHPUT_DATASET_NAME:-random}
SIMLLM_THROUGHPUT_BACKEND=${SIMLLM_THROUGHPUT_BACKEND:-vllm}
SIMLLM_SATURATED_DRY_RUN=${SIMLLM_SATURATED_DRY_RUN:-0}

# Warm slowly enough that entries are committed before the measured burst.
# The measurement seed defaults to the warmup seed in the underlying runner.
SIMLLM_WARMCACHE_REQUEST_RATE=${SIMLLM_WARMCACHE_REQUEST_RATE:-1}
SIMLLM_WARMCACHE_NUM_PROMPTS=${SIMLLM_WARMCACHE_NUM_PROMPTS:-$SIMLLM_THROUGHPUT_NUM_PROMPTS}
SIMLLM_WARMCACHE_SEED=${SIMLLM_WARMCACHE_SEED:-0}
SIMLLM_MEASURE_SEED=${SIMLLM_MEASURE_SEED:-$SIMLLM_WARMCACHE_SEED}
SIMLLM_KV_CACHE_SIZE=${VLLM_ASCEND_SIMLLM_KV_CACHE_SIZE:-${SIMLLM_KV_CACHE_SIZE:-$SIMLLM_THROUGHPUT_NUM_PROMPTS}}

require_positive_integer() {
  local name=$1
  local value=$2

  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer; got: $value" >&2
    exit 2
  fi
}

if [[ ! -f "$BASE_SPEC_FILE" ]]; then
  echo "Base spec file not found: $BASE_SPEC_FILE" >&2
  exit 2
fi
if [[ ! -x "$WARM_CACHE_RUNNER" ]]; then
  echo "Warm-cache runner not found or not executable: $WARM_CACHE_RUNNER" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to build the saturated workload and comparison." >&2
  exit 2
fi

require_positive_integer SIMLLM_THROUGHPUT_NUM_PROMPTS "$SIMLLM_THROUGHPUT_NUM_PROMPTS"
require_positive_integer SIMLLM_THROUGHPUT_INPUT_LEN "$SIMLLM_THROUGHPUT_INPUT_LEN"
require_positive_integer SIMLLM_THROUGHPUT_OUTPUT_LEN "$SIMLLM_THROUGHPUT_OUTPUT_LEN"
require_positive_integer SIMLLM_THROUGHPUT_MAX_CONCURRENCY "$SIMLLM_THROUGHPUT_MAX_CONCURRENCY"
require_positive_integer SIMLLM_THROUGHPUT_MAX_NUM_BATCHED_TOKENS "$SIMLLM_THROUGHPUT_MAX_NUM_BATCHED_TOKENS"
require_positive_integer SIMLLM_KV_CACHE_SIZE "$SIMLLM_KV_CACHE_SIZE"

if (( SIMLLM_THROUGHPUT_MAX_NUM_BATCHED_TOKENS < SIMLLM_THROUGHPUT_INPUT_LEN )); then
  echo "SIMLLM_THROUGHPUT_MAX_NUM_BATCHED_TOKENS must be at least SIMLLM_THROUGHPUT_INPUT_LEN" >&2
  echo "A smaller token budget would split warmup prompts and prevent exact cache reuse." >&2
  exit 2
fi

BASE_SPEC_FILE=$(cd "$(dirname "$BASE_SPEC_FILE")" && pwd)/$(basename "$BASE_SPEC_FILE")
mkdir -p "$RESULT_DIR"

SATURATED_SPEC_FILE="$RESULT_DIR/saturated-same-spec.json"

jq \
  --arg scenario "$SIMLLM_THROUGHPUT_SCENARIO" \
  --arg dataset_name "$SIMLLM_THROUGHPUT_DATASET_NAME" \
  --arg endpoint "$SIMLLM_THROUGHPUT_ENDPOINT" \
  --arg backend "$SIMLLM_THROUGHPUT_BACKEND" \
  --argjson num_prompts "$SIMLLM_THROUGHPUT_NUM_PROMPTS" \
  --argjson input_len "$SIMLLM_THROUGHPUT_INPUT_LEN" \
  --argjson output_len "$SIMLLM_THROUGHPUT_OUTPUT_LEN" \
  --argjson max_concurrency "$SIMLLM_THROUGHPUT_MAX_CONCURRENCY" \
  --argjson max_num_batched_tokens "$SIMLLM_THROUGHPUT_MAX_NUM_BATCHED_TOKENS" '
    .scenario = $scenario
    | .id = ("simllm-" + $scenario + "-saturated-throughput-warm-cache")
    | .label = "SimLLM saturated throughput warm-cache"
    | .client_parameters = ((.client_parameters // {})
        + {
            backend: $backend,
            endpoint: $endpoint,
            dataset_name: $dataset_name,
            num_prompts: $num_prompts,
            input_len: $input_len,
            output_len: $output_len,
            temperature: 0,
            request_rate: "inf",
            max_concurrency: $max_concurrency
          }
      )
    | .client_parameters |= del(.dataset_path, .num_warmups)
    | .server_parameters = ((.server_parameters // {})
        + {max_num_batched_tokens: $max_num_batched_tokens})
  ' "$BASE_SPEC_FILE" > "$SATURATED_SPEC_FILE"

echo "[simllm-throughput] base spec: $BASE_SPEC_FILE"
echo "[simllm-throughput] saturated spec: $SATURATED_SPEC_FILE"
echo "[simllm-throughput] result dir: $RESULT_DIR"
echo "[simllm-throughput] warmup: rate=$SIMLLM_WARMCACHE_REQUEST_RATE seed=$SIMLLM_WARMCACHE_SEED"
echo "[simllm-throughput] measure: rate=inf concurrency=$SIMLLM_THROUGHPUT_MAX_CONCURRENCY seed=$SIMLLM_MEASURE_SEED"
echo "[simllm-throughput] workload: prompts=$SIMLLM_THROUGHPUT_NUM_PROMPTS input=$SIMLLM_THROUGHPUT_INPUT_LEN output=$SIMLLM_THROUGHPUT_OUTPUT_LEN"
echo "[simllm-throughput] server token budget: $SIMLLM_THROUGHPUT_MAX_NUM_BATCHED_TOKENS"
echo "[simllm-throughput] SimLLM cache entries: $SIMLLM_KV_CACHE_SIZE"

if [[ "$SIMLLM_SATURATED_DRY_RUN" == "1" ]]; then
  echo "[simllm-throughput] dry run: generated spec only"
  exit 0
fi

export RESULT_DIR
export CURRENT_DATA_SOURCE
export CURRENT_VLLM_ASCEND_HUST_REPO
export SIMLLM_WARMCACHE_REQUEST_RATE
export SIMLLM_WARMCACHE_NUM_PROMPTS
export SIMLLM_WARMCACHE_SEED
export SIMLLM_MEASURE_SEED
export SIMLLM_KV_CACHE_SIZE

"$WARM_CACHE_RUNNER" "$SATURATED_SPEC_FILE"

BASELINE_RESULT="$RESULT_DIR/baseline-disabled/raw_benchmark_result.json"
SIMLLM_RESULT="$RESULT_DIR/enabled-warm-cache/raw_benchmark_result.json"
COMPARISON_JSON="$RESULT_DIR/throughput_comparison.json"
COMPARISON_MD="$RESULT_DIR/throughput_comparison.md"

if [[ ! -f "$BASELINE_RESULT" || ! -f "$SIMLLM_RESULT" ]]; then
  echo "[simllm-throughput] raw result missing; skipping throughput summary" >&2
  exit 1
fi

for result_file in "$BASELINE_RESULT" "$SIMLLM_RESULT"; do
  completed=$(jq -r '.completed // 0' "$result_file")
  if [[ "$completed" != "$SIMLLM_THROUGHPUT_NUM_PROMPTS" ]]; then
    echo "[simllm-throughput] invalid result: $result_file completed $completed/$SIMLLM_THROUGHPUT_NUM_PROMPTS requests" >&2
    exit 1
  fi
done

jq -n \
  --slurpfile baseline "$BASELINE_RESULT" \
  --slurpfile simllm "$SIMLLM_RESULT" '
    def pct($before; $after):
      if $before == null or $after == null or $before == 0 then null
      else (($after - $before) / $before * 100)
      end;
    {
      workload: {
        request_rate: $simllm[0].request_rate,
        max_concurrency: $simllm[0].max_concurrency,
        completed: $simllm[0].completed
      },
      baseline: {
        request_throughput: $baseline[0].request_throughput,
        output_token_throughput: $baseline[0].output_throughput,
        total_token_throughput: $baseline[0].total_token_throughput
      },
      simllm_warm_cache: {
        request_throughput: $simllm[0].request_throughput,
        output_token_throughput: $simllm[0].output_throughput,
        total_token_throughput: $simllm[0].total_token_throughput
      },
      improvement_percent: {
        request_throughput: pct($baseline[0].request_throughput; $simllm[0].request_throughput),
        output_token_throughput: pct($baseline[0].output_throughput; $simllm[0].output_throughput),
        total_token_throughput: pct($baseline[0].total_token_throughput; $simllm[0].total_token_throughput)
      }
    }
  ' > "$COMPARISON_JSON"

jq -r '
  def value($number): if $number == null then "n/a" else ($number | tostring) end;
  def percent($number):
    if $number == null then "n/a" else (($number * 100 | round) / 100 | tostring) + "%" end;
  "# SimLLM saturated warm-cache throughput\n\n" +
  "| Metric | Baseline | SimLLM warm cache | Improvement |\n" +
  "| --- | ---: | ---: | ---: |\n" +
  "| Requests/s | \(value(.baseline.request_throughput)) | \(value(.simllm_warm_cache.request_throughput)) | \(percent(.improvement_percent.request_throughput)) |\n" +
  "| Output tokens/s | \(value(.baseline.output_token_throughput)) | \(value(.simllm_warm_cache.output_token_throughput)) | \(percent(.improvement_percent.output_token_throughput)) |\n" +
  "| Total tokens/s | \(value(.baseline.total_token_throughput)) | \(value(.simllm_warm_cache.total_token_throughput)) | \(percent(.improvement_percent.total_token_throughput)) |\n\n" +
  "Measured with request_rate=\(.workload.request_rate), max_concurrency=\(.workload.max_concurrency), completed=\(.workload.completed)."
  ' "$COMPARISON_JSON" > "$COMPARISON_MD"

echo ""
cat "$COMPARISON_MD"
echo ""
echo "[simllm-throughput] comparison JSON: $COMPARISON_JSON"
echo "[simllm-throughput] comparison report: $COMPARISON_MD"
