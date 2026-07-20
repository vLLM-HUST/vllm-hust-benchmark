#!/bin/bash
set -euo pipefail

# SimLLM warm-cache runner with a prefill-heavy measurement workload.
#
# This is a thin wrapper around the existing warm-cache flow:
#   - Keep the same warmup / server lifecycle logic
#   - Rewrite the measurement workload into a temporary same-spec file
#   - Use a longer prefill, shorter decode, and higher request rate so that
#     SimLLM's prefill rewrite is more likely to show up as throughput gain
#
# Default workload:
#   scenario      = random-online
#   num_prompts   = 200
#   input_len     = 4096
#   output_len    = 32
#   request_rate  = 4
#
# Example:
#   bash scripts/run_simllm_random_online_prefill_heavy_warm_cache.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
BASE_WRAPPER=${BASE_WRAPPER:-"$SCRIPT_DIR/run_simllm_warmcache.sh"}
DEFAULT_BASE_SPEC_FILE="$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json"

BASE_SPEC_FILE=${1:-${BASE_SPEC_FILE:-$DEFAULT_BASE_SPEC_FILE}}
RESULT_DIR=${RESULT_DIR:-"$REPO_ROOT/.benchmarks/simllm-random-online-prefill-heavy-warm-cache"}
CURRENT_DATA_SOURCE=${CURRENT_DATA_SOURCE:-"vllm-ascend-hust-ci-simllm-prefill-heavy-random-online"}

SIMLLM_PREFILL_HEAVY_NUM_PROMPTS=${SIMLLM_PREFILL_HEAVY_NUM_PROMPTS:-200}
SIMLLM_PREFILL_HEAVY_INPUT_LEN=${SIMLLM_PREFILL_HEAVY_INPUT_LEN:-4096}
SIMLLM_PREFILL_HEAVY_OUTPUT_LEN=${SIMLLM_PREFILL_HEAVY_OUTPUT_LEN:-32}
SIMLLM_PREFILL_HEAVY_REQUEST_RATE=${SIMLLM_PREFILL_HEAVY_REQUEST_RATE:-4}
SIMLLM_PREFILL_HEAVY_SCENARIO=${SIMLLM_PREFILL_HEAVY_SCENARIO:-random-online}
SIMLLM_PREFILL_HEAVY_ENDPOINT=${SIMLLM_PREFILL_HEAVY_ENDPOINT:-/v1/completions}
SIMLLM_PREFILL_HEAVY_DATASET_NAME=${SIMLLM_PREFILL_HEAVY_DATASET_NAME:-random}
SIMLLM_PREFILL_HEAVY_BACKEND=${SIMLLM_PREFILL_HEAVY_BACKEND:-vllm}

if [[ ! -f "$BASE_SPEC_FILE" ]]; then
  echo "Base spec file not found: $BASE_SPEC_FILE" >&2
  exit 2
fi

if [[ ! -x "$BASE_WRAPPER" ]]; then
  echo "Warm-cache wrapper not found or not executable: $BASE_WRAPPER" >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to rewrite the temporary measurement spec." >&2
  exit 2
fi

BASE_SPEC_FILE=$(cd "$(dirname "$BASE_SPEC_FILE")" && pwd)/$(basename "$BASE_SPEC_FILE")
mkdir -p "$RESULT_DIR"

TEMP_SPEC_FILE="$RESULT_DIR/prefill-heavy-same-spec.json"

jq \
  --arg scenario "$SIMLLM_PREFILL_HEAVY_SCENARIO" \
  --arg dataset_name "$SIMLLM_PREFILL_HEAVY_DATASET_NAME" \
  --arg endpoint "$SIMLLM_PREFILL_HEAVY_ENDPOINT" \
  --arg backend "$SIMLLM_PREFILL_HEAVY_BACKEND" \
  --argjson num_prompts "$SIMLLM_PREFILL_HEAVY_NUM_PROMPTS" \
  --argjson input_len "$SIMLLM_PREFILL_HEAVY_INPUT_LEN" \
  --argjson output_len "$SIMLLM_PREFILL_HEAVY_OUTPUT_LEN" \
  --argjson request_rate "$SIMLLM_PREFILL_HEAVY_REQUEST_RATE" '
    .scenario = $scenario
    | .id = ("simllm-" + $scenario + "-prefill-heavy")
    | .label = ("SimLLM " + $scenario + " prefill-heavy warm-cache")
    | .client_parameters = ((.client_parameters // {})
        + {
            backend: $backend,
            endpoint: $endpoint,
            dataset_name: $dataset_name,
            num_prompts: $num_prompts,
            input_len: $input_len,
            output_len: $output_len,
            request_rate: $request_rate
          }
      )
    | .client_parameters |= del(.dataset_path, .num_warmups)
  ' "$BASE_SPEC_FILE" > "$TEMP_SPEC_FILE"

echo "[simllm-prefill-heavy] base spec: $BASE_SPEC_FILE"
echo "[simllm-prefill-heavy] temp spec: $TEMP_SPEC_FILE"
echo "[simllm-prefill-heavy] result dir: $RESULT_DIR"
echo "[simllm-prefill-heavy] target workload: input=${SIMLLM_PREFILL_HEAVY_INPUT_LEN} output=${SIMLLM_PREFILL_HEAVY_OUTPUT_LEN} request_rate=${SIMLLM_PREFILL_HEAVY_REQUEST_RATE} num_prompts=${SIMLLM_PREFILL_HEAVY_NUM_PROMPTS}"

export RESULT_DIR
export CURRENT_DATA_SOURCE

exec "$BASE_WRAPPER" "$TEMP_SPEC_FILE"
