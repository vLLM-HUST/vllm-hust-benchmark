#!/bin/bash
set -euo pipefail

# SimLLM warm-cache runner for ShareGPT-style serve benchmarks.
#
# This is a thin wrapper around the existing warm-cache A/B flow:
#   - Baseline: SimLLM disabled
#   - SimLLM:   warm up KVManager first, then run the measured benchmark
#
# Default spec:
#   docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-throughput-qwen25-14b-910b2.json
#
# To run the online variant instead, pass the online spec explicitly:
#   bash scripts/run_simllm_warmcache.sh \
#     docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-910b2.json

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DEFAULT_SPEC_FILE="$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-throughput-qwen25-14b-910b2.json"
RUNNER=${RUNNER:-"$SCRIPT_DIR/run_simllm_random_online_warm_cache.sh"}

SPEC_FILE=${1:-${SPEC_FILE:-$DEFAULT_SPEC_FILE}}

if [[ ! -x "$RUNNER" ]]; then
  echo "Warm-cache runner not found or not executable: $RUNNER" >&2
  exit 2
fi

if [[ ! -f "$SPEC_FILE" ]]; then
  echo "Spec file not found: $SPEC_FILE" >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to derive the scenario-specific result directory." >&2
  exit 2
fi

SPEC_FILE=$(cd "$(dirname "$SPEC_FILE")" && pwd)/$(basename "$SPEC_FILE")
SPEC_SCENARIO=$(jq -r '.scenario' "$SPEC_FILE")
SPEC_TAG=$(printf '%s' "$SPEC_SCENARIO" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')

RESULT_DIR=${RESULT_DIR:-"$REPO_ROOT/.benchmarks/simllm-${SPEC_TAG}-warm-cache"}
CURRENT_DATA_SOURCE=${CURRENT_DATA_SOURCE:-"vllm-ascend-hust-ci-simllm-warmcache-${SPEC_TAG}"}

mkdir -p "$RESULT_DIR"

echo "[simllm-warmcache] spec: $SPEC_FILE"
echo "[simllm-warmcache] scenario: $SPEC_SCENARIO"
echo "[simllm-warmcache] result dir: $RESULT_DIR"

export RESULT_DIR
export CURRENT_DATA_SOURCE

exec "$RUNNER" "$SPEC_FILE"
