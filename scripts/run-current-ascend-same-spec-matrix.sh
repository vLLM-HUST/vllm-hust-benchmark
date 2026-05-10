#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=${VLLM_HUST_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}
SINGLE_RUNNER=${SINGLE_RUNNER:-"$SCRIPT_DIR/run-current-ascend-same-spec.sh"}
MATRIX_RUN_ID=${MATRIX_RUN_ID:-"current-ascend-same-spec-matrix-$(date -u +%Y%m%dT%H%M%SZ)"}
MATRIX_RESULT_ROOT=${MATRIX_RESULT_ROOT:-"$REPO_ROOT/.benchmarks/$MATRIX_RUN_ID"}
PUBLISH_WEBSITE=${PUBLISH_WEBSITE:-0}
WEBSITE_OUTPUT_DIR=${WEBSITE_OUTPUT_DIR:-"$WORKSPACE_ROOT/vllm-hust-website/data"}
CURRENT_ENV_PREFIX=${CURRENT_ENV_PREFIX:-"/root/miniconda3/envs/vllm-hust-dev"}
CURRENT_RUNTIME_PYTHON=${CURRENT_RUNTIME_PYTHON:-"$CURRENT_ENV_PREFIX/bin/python"}
CURRENT_RUNTIME_PYTHONPATH=${CURRENT_RUNTIME_PYTHONPATH:-}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-current-ascend-same-spec-matrix.sh <spec-file-or-dir> [more spec files or dirs...]

Behavior:
  - Each JSON spec is executed through scripts/run-current-ascend-same-spec.sh
  - Each run gets its own RESULT_DIR under .benchmarks/<matrix-run-id>/
  - Set PUBLISH_WEBSITE=1 to aggregate all generated submissions into vllm-hust-website/data

Examples:
  bash scripts/run-current-ascend-same-spec-matrix.sh docs/spec-matrix/

  PUBLISH_WEBSITE=1 bash scripts/run-current-ascend-same-spec-matrix.sh \
    docs/spec-matrix/random-online-tp1.json \
    docs/spec-matrix/random-online-tp2.json
EOF
}

slugify() {
  local value=$1
  value=$(basename "$value")
  value=${value%.json}
  value=$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')
  value=$(printf '%s' "$value" | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')
  printf '%s\n' "${value:-spec}"
}

collect_specs() {
  local input_path=$1
  if [[ -d "$input_path" ]]; then
    find "$(realpath "$input_path")" -maxdepth 1 -type f -name '*.json' \
      ! -name '*.stub.json' \
      ! -name '*constraints*' | sort
    return 0
  fi

  if [[ -f "$input_path" ]]; then
    realpath "$input_path"
    return 0
  fi

  echo "Spec path not found: $input_path" >&2
  return 1
}

if [[ ! -x "$SINGLE_RUNNER" ]]; then
  echo "Single-run same-spec runner is not executable: $SINGLE_RUNNER" >&2
  exit 2
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

if [[ ! -x "$CURRENT_RUNTIME_PYTHON" ]]; then
  echo "CURRENT_RUNTIME_PYTHON is not executable: $CURRENT_RUNTIME_PYTHON" >&2
  exit 2
fi

mkdir -p "$MATRIX_RESULT_ROOT"

mapfile -t SPEC_FILES < <(
  for path in "$@"; do
    collect_specs "$path"
  done | awk 'NF' | sort -u
)

if [[ ${#SPEC_FILES[@]} -eq 0 ]]; then
  echo "No spec files resolved from input arguments." >&2
  exit 2
fi

echo "[same-spec-matrix] result root: $MATRIX_RESULT_ROOT"
echo "[same-spec-matrix] resolved ${#SPEC_FILES[@]} spec file(s)"

for spec_file in "${SPEC_FILES[@]}"; do
  spec_slug=$(slugify "$spec_file")
  result_dir="$MATRIX_RESULT_ROOT/$spec_slug"
  run_id="$MATRIX_RUN_ID-$spec_slug"

  echo
  echo "[same-spec-matrix] running spec: $spec_file"
  echo "[same-spec-matrix] result dir: $result_dir"

  RESULT_DIR="$result_dir" \
  RUN_ID="$run_id" \
  CURRENT_RUNTIME_PYTHON="$CURRENT_RUNTIME_PYTHON" \
  CURRENT_RUNTIME_PYTHONPATH="$CURRENT_RUNTIME_PYTHONPATH" \
  bash "$SINGLE_RUNNER" "$spec_file"
done

if [[ "$PUBLISH_WEBSITE" == "1" ]]; then
  echo
  echo "[same-spec-matrix] publishing aggregated website data from: $MATRIX_RESULT_ROOT"
  PYTHONPATH="$REPO_ROOT/src${CURRENT_RUNTIME_PYTHONPATH:+:$CURRENT_RUNTIME_PYTHONPATH}" \
    "$CURRENT_RUNTIME_PYTHON" -m vllm_hust_benchmark.cli publish-website \
      --source-dir "$MATRIX_RESULT_ROOT" \
      --output-dir "$WEBSITE_OUTPUT_DIR" \
      --execute
  echo "[same-spec-matrix] website data refreshed at: $WEBSITE_OUTPUT_DIR"
fi

echo
echo "[same-spec-matrix] completed ${#SPEC_FILES[@]} run(s)"
echo "[same-spec-matrix] submissions root: $MATRIX_RESULT_ROOT"