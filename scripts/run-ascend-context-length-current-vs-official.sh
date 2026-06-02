#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=${VLLM_HUST_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}
PREPARE_SCRIPT=${PREPARE_SCRIPT:-"$SCRIPT_DIR/prepare-official-ascend-baseline-env.sh"}
OFFICIAL_RUNNER=${OFFICIAL_RUNNER:-"$SCRIPT_DIR/run-official-ascend-goal-baseline.sh"}
CURRENT_RUNNER=${CURRENT_RUNNER:-"$SCRIPT_DIR/run-current-ascend-same-spec.sh"}
DEFAULT_SPEC_DIR="$REPO_ROOT/docs/official-baselines/context-length-sweep-random-online-qwen25-14b-910b3"
MATRIX_RUN_ID=${MATRIX_RUN_ID:-"ascend-context-length-sweep-$(date -u +%Y%m%dT%H%M%SZ)"}
MATRIX_RESULT_ROOT=${MATRIX_RESULT_ROOT:-"$REPO_ROOT/.benchmarks/$MATRIX_RUN_ID"}
SUMMARY_FILE=${MATRIX_SUMMARY_FILE:-"$MATRIX_RESULT_ROOT/summary.md"}
PUBLISH_WEBSITE=${PUBLISH_WEBSITE:-1}
WEBSITE_OUTPUT_DIR=${WEBSITE_OUTPUT_DIR:-"$WORKSPACE_ROOT/vllm-hust-website/data"}
PREPARE_OFFICIAL_ENV=${PREPARE_OFFICIAL_ENV:-1}
FORCE_REPAIR_OFFICIAL_ENV=${FORCE_REPAIR_OFFICIAL_ENV:-0}
GOAL_BASELINE_ENV_PREFIX=${GOAL_BASELINE_ENV_PREFIX:-}
CURRENT_ENV_PREFIX=${CURRENT_ENV_PREFIX:-"/root/miniconda3/envs/vllm-hust-dev"}
CURRENT_RUNTIME_PYTHON=${CURRENT_RUNTIME_PYTHON:-"$CURRENT_ENV_PREFIX/bin/python"}
CURRENT_VLLM_HUST_REPO=${CURRENT_VLLM_HUST_REPO:-"$WORKSPACE_ROOT/vllm-hust"}
CURRENT_VLLM_ASCEND_HUST_REPO=${CURRENT_VLLM_ASCEND_HUST_REPO:-"$WORKSPACE_ROOT/vllm-ascend-hust"}
CURRENT_GITHUB_REF=${CURRENT_GITHUB_REF:-$(git -C "$CURRENT_VLLM_HUST_REPO" branch --show-current 2>/dev/null || echo main)}
CURRENT_PLUGIN_GITHUB_REF=${CURRENT_PLUGIN_GITHUB_REF:-$(git -C "$CURRENT_VLLM_ASCEND_HUST_REPO" branch --show-current 2>/dev/null || echo main)}
HOST_PYTHON_BIN=${HOST_PYTHON_BIN:-$(command -v python3 || command -v python || true)}
OFFICIAL_SERVER_PORT_BASE=${OFFICIAL_SERVER_PORT_BASE:-8000}
CURRENT_SERVER_PORT_BASE=${CURRENT_SERVER_PORT_BASE:-8100}
ASCEND_DEVICE_PREFERENCE_FILE=${ASCEND_DEVICE_PREFERENCE_FILE:-"$MATRIX_RESULT_ROOT/.runtime-state/preferred-ascend-device.txt"}
CONTEXT_SWEEP_ALLOW_LONG_MAX_MODEL_LEN=${CONTEXT_SWEEP_ALLOW_LONG_MAX_MODEL_LEN:-1}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-900}
READY_STATUS_INTERVAL_SECONDS=${READY_STATUS_INTERVAL_SECONDS:-30}
CLIENT_READY_CHECK_TIMEOUT_SECONDS=${CLIENT_READY_CHECK_TIMEOUT_SECONDS:-900}
DEVICE_SELECTION_RETRIES=${DEVICE_SELECTION_RETRIES:-20}
DEVICE_SELECTION_RETRY_DELAY_SECONDS=${DEVICE_SELECTION_RETRY_DELAY_SECONDS:-30}
SPEC_TIMEOUT_SECONDS=${SPEC_TIMEOUT_SECONDS:-3600}
JOB_START_EPOCH=$(date +%s)
JOB_TIMEOUT_SECONDS=${JOB_TIMEOUT_SECONDS:-19800}  # 5.5h safety margin within 6h job timeout
RESUME_CHECKPOINT_ROOT=${RESUME_CHECKPOINT_ROOT:-}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-1}
CHECKPOINT_MAX_AGE_HOURS=${CHECKPOINT_MAX_AGE_HOURS:-48}
CURRENT_GIT_COMMIT=${CURRENT_GIT_COMMIT:-$(git -C "$CURRENT_VLLM_HUST_REPO" rev-parse HEAD 2>/dev/null || true)}
CURRENT_PLUGIN_GIT_COMMIT=${CURRENT_PLUGIN_GIT_COMMIT:-$(git -C "$CURRENT_VLLM_ASCEND_HUST_REPO" rev-parse HEAD 2>/dev/null || true)}
OFFICIAL_GIT_COMMIT=${OFFICIAL_GIT_COMMIT:-$(git -C "$WORKSPACE_ROOT/reference-repos/vllm" rev-parse HEAD 2>/dev/null || true)}
OFFICIAL_PLUGIN_GIT_COMMIT=${OFFICIAL_PLUGIN_GIT_COMMIT:-$(git -C "$WORKSPACE_ROOT/reference-repos/vllm-ascend" rev-parse HEAD 2>/dev/null || true)}

PREPARED_ENV=0
FAILED_COUNT=0
FAILED_ITEMS=()
RUN_COUNT=0
RESUMED_COUNT=0
RESUMED_ITEMS=()

# Issue 2: trap to write summary on cancellation so partial results are captured
trap_write_partial_summary() {
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]] && [[ -f "$SUMMARY_FILE" ]]; then
    echo "" >> "$GITHUB_STEP_SUMMARY"
    echo "**Job interrupted (signal received)**" >> "$GITHUB_STEP_SUMMARY"
    cat "$SUMMARY_FILE" >> "$GITHUB_STEP_SUMMARY" 2>/dev/null || true
  fi
}
trap trap_write_partial_summary SIGTERM SIGINT

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-ascend-context-length-current-vs-official.sh [<spec-file-or-dir> ...]

Behavior:
  - Defaults to docs/official-baselines/context-length-sweep-random-online-qwen25-14b-910b3
  - Runs the official v0.11.0 baseline and current vllm-hust/vllm-ascend-hust current benchmark once per spec
  - Aggregates all generated submissions into website snapshots when PUBLISH_WEBSITE=1

Environment:
  GOAL_BASELINE_ENV_PREFIX   Required official baseline conda env prefix
  CURRENT_ENV_PREFIX         Current benchmark conda env prefix (default: /root/miniconda3/envs/vllm-hust-dev)
  CURRENT_RUNTIME_PYTHON     Optional explicit current runtime Python path
  PUBLISH_WEBSITE            1 to generate leaderboard_single/multi/compare snapshots (default: 1)
  WEBSITE_OUTPUT_DIR         Output dir for aggregated website snapshots
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

append_summary() {
  printf '%s\n' "$1" >> "$SUMMARY_FILE"
}

record_failure() {
  local item=$1
  local reason=$2
  FAILED_ITEMS+=("${item}: ${reason}")
  ((FAILED_COUNT+=1))
  append_summary "- Failure: ${item} (${reason})"
}

record_resume() {
  local item=$1
  local reason=$2
  RESUMED_ITEMS+=("${item}: ${reason}")
  ((RESUMED_COUNT+=1))
  append_summary "- Resumed: ${item} (${reason})"
}

resolve_spec_hash() {
  local spec_file=$1
  local tmp_dir tmp_file resolved_hash

  tmp_dir=$(mktemp -d)
  tmp_file="$tmp_dir/resolved_same_spec.json"

  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$HOST_PYTHON_BIN" -m vllm_hust_benchmark.same_spec resolve \
      --spec-file "$spec_file" \
      --output-file "$tmp_file" >/dev/null

  resolved_hash=$(jq -r '.resolved_spec_hash // empty' "$tmp_file")
  rm -rf "$tmp_dir"

  if [[ -z "$resolved_hash" ]]; then
    return 1
  fi

  printf '%s\n' "$resolved_hash"
}

checkpoint_benchmark_root() {
  local checkpoint_root=${1:-}
  local benchmark_root

  [[ -n "$checkpoint_root" ]] || return 1
  [[ -d "$checkpoint_root/.benchmarks" ]] || return 1

  benchmark_root=$(find "$checkpoint_root/.benchmarks" -maxdepth 1 -type d -name 'context-length-sweep-*' | sort | tail -n 1)
  [[ -n "$benchmark_root" ]] || return 1

  printf '%s\n' "$benchmark_root"
}

result_dir_has_submission() {
  local result_dir=$1

  [[ -f "$result_dir/raw_benchmark_result.json" ]] \
    && [[ -f "$result_dir/resolved_same_spec.json" ]] \
    && [[ -f "$result_dir/submission/leaderboard_manifest.json" ]] \
    && [[ -f "$result_dir/submission/run_leaderboard.json" ]]
}

restore_result_dir_from_checkpoint() {
  local side=$1
  local spec_file=$2
  local spec_slug=$3
  local current_spec_hash=$4
  local checkpoint_root
  local source_dir
  local target_dir="$MATRIX_RESULT_ROOT/$side/$spec_slug"
  local source_spec_id source_spec_hash
  local current_spec_id

  [[ "$RESUME_FROM_CHECKPOINT" == "1" ]] || return 1
  [[ -n "$RESUME_CHECKPOINT_ROOT" ]] || return 1

  checkpoint_root=$(checkpoint_benchmark_root "$RESUME_CHECKPOINT_ROOT") || return 1
  source_dir="$checkpoint_root/$side/$spec_slug"
  [[ -d "$source_dir" ]] || return 1
  result_dir_has_submission "$source_dir" || return 1

  # Issue 3: checkpoint staleness check
  if [[ "$CHECKPOINT_MAX_AGE_HOURS" -gt 0 ]]; then
    local checkpoint_mtime
    checkpoint_mtime=$(stat -c '%Y' "$source_dir/raw_benchmark_result.json" 2>/dev/null || echo 0)
    local now_epoch
    now_epoch=$(date +%s)
    local max_age_seconds=$((CHECKPOINT_MAX_AGE_HOURS * 3600))
    if (( (now_epoch - checkpoint_mtime) > max_age_seconds )); then
      echo "[context-sweep] checkpoint for ${side}:${spec_slug} is older than ${CHECKPOINT_MAX_AGE_HOURS}h; skipping restore" >&2
      return 1
    fi
  fi

  current_spec_id=$(jq -r '.id // empty' "$spec_file")
  source_spec_id=$(jq -r '.same_spec.spec_id // .spec_id // empty' "$source_dir/resolved_same_spec.json")
  source_spec_hash=$(jq -r '.same_spec.resolved_spec_hash // empty' "$source_dir/submission/run_leaderboard.json")

  if [[ -z "$source_spec_hash" ]]; then
    source_spec_hash=$(jq -r '.resolved_spec_hash // empty' "$source_dir/resolved_same_spec.json")
  fi

  [[ -n "$current_spec_id" ]] || return 1
  [[ "$source_spec_id" == "$current_spec_id" ]] || return 1
  [[ "$source_spec_hash" == "$current_spec_hash" ]] || return 1

  case "$side" in
    current)
      [[ $(jq -r '.metadata.submitter // empty' "$source_dir/submission/run_leaderboard.json") == "same-spec-current" ]] || return 1
      [[ $(jq -r '.metadata.runtime_provenance.engine.commit // empty' "$source_dir/submission/run_leaderboard.json") == "$CURRENT_GIT_COMMIT" ]] || return 1
      [[ $(jq -r '.metadata.runtime_provenance.plugin.commit // empty' "$source_dir/submission/run_leaderboard.json") == "$CURRENT_PLUGIN_GIT_COMMIT" ]] || return 1
      ;;
    official)
      [[ $(jq -r '.metadata.submitter // empty' "$source_dir/submission/run_leaderboard.json") == "official-ascend-baseline" ]] || return 1
      [[ $(jq -r '.metadata.github_repository // empty' "$source_dir/submission/run_leaderboard.json") == "vllm-project/vllm-ascend" ]] || return 1
      [[ $(jq -r '.metadata.github_ref // empty' "$source_dir/submission/run_leaderboard.json") == "v0.11.0" ]] || return 1
      [[ $(jq -r '.metadata.runtime_provenance.plugin.commit // empty' "$source_dir/submission/run_leaderboard.json") == "$OFFICIAL_PLUGIN_GIT_COMMIT" ]] || return 1
      ;;
    *)
      return 1
      ;;
  esac

  mkdir -p "$(dirname "$target_dir")"
  rm -rf "$target_dir"
  cp -a "$source_dir" "$target_dir"
  record_resume "${side}:${spec_slug}" "restored from checkpoint"
  return 0
}

prepare_official_env_once() {
  if [[ "$PREPARED_ENV" == "1" ]] || [[ "$PREPARE_OFFICIAL_ENV" != "1" ]]; then
    return 0
  fi

  echo "[context-sweep] preparing official baseline environment: $GOAL_BASELINE_ENV_PREFIX"
  ENV_PREFIX="$GOAL_BASELINE_ENV_PREFIX" \
  FORCE_REPAIR_OFFICIAL_ENV="$FORCE_REPAIR_OFFICIAL_ENV" \
  VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
  bash "$PREPARE_SCRIPT"
  PREPARED_ENV=1
}

run_official_for_spec() {
  local spec_file=$1
  local spec_slug=$2
  local spec_index=$3
  local result_dir="$MATRIX_RESULT_ROOT/official/$spec_slug"
  local run_id="$MATRIX_RUN_ID-official-$spec_slug"
  local runner_log="$result_dir/runner.log"
  local port=$((OFFICIAL_SERVER_PORT_BASE + spec_index * 10))

  mkdir -p "$result_dir"

  set +e
  RESULT_DIR="$result_dir" \
  RUN_ID="$run_id" \
  GOAL_BASELINE_ENV_PREFIX="$GOAL_BASELINE_ENV_PREFIX" \
  GOAL_BASELINE_DEVICE_PREFERENCE_FILE="$ASCEND_DEVICE_PREFERENCE_FILE" \
  VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
  HOST_PYTHON_BIN="$HOST_PYTHON_BIN" \
  VLLM_ALLOW_LONG_MAX_MODEL_LEN="$CONTEXT_SWEEP_ALLOW_LONG_MAX_MODEL_LEN" \
  OFFICIAL_SERVER_PORT="$port" \
  OFFICIAL_CLIENT_PORT="$port" \
  READY_TIMEOUT_SECONDS="$READY_TIMEOUT_SECONDS" \
  READY_STATUS_INTERVAL_SECONDS="$READY_STATUS_INTERVAL_SECONDS" \
  CLIENT_READY_CHECK_TIMEOUT_SECONDS="$CLIENT_READY_CHECK_TIMEOUT_SECONDS" \
  DEVICE_SELECTION_RETRIES="$DEVICE_SELECTION_RETRIES" \
  DEVICE_SELECTION_RETRY_DELAY_SECONDS="$DEVICE_SELECTION_RETRY_DELAY_SECONDS" \
  bash "$OFFICIAL_RUNNER" "$spec_file" 2>&1 | tee "$runner_log"
  local runner_status=${PIPESTATUS[0]}
  set -e

  if [[ "$runner_status" -ne 0 ]]; then
    record_failure "official:${spec_slug}" "exit=${runner_status}"
    return "$runner_status"
  fi

  ((RUN_COUNT+=1))
  append_summary "  - official run: ${result_dir}"
  return 0
}

run_current_for_spec() {
  local spec_file=$1
  local spec_slug=$2
  local spec_index=$3
  local result_dir="$MATRIX_RESULT_ROOT/current/$spec_slug"
  local run_id="$MATRIX_RUN_ID-current-$spec_slug"
  local runner_log="$result_dir/runner.log"
  local port=$((CURRENT_SERVER_PORT_BASE + spec_index * 10))

  mkdir -p "$result_dir"

  set +e
  RESULT_DIR="$result_dir" \
  RUN_ID="$run_id" \
  VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
  CURRENT_ENV_PREFIX="$CURRENT_ENV_PREFIX" \
  CURRENT_RUNTIME_PYTHON="$CURRENT_RUNTIME_PYTHON" \
  CURRENT_VLLM_HUST_REPO="$CURRENT_VLLM_HUST_REPO" \
  CURRENT_VLLM_ASCEND_HUST_REPO="$CURRENT_VLLM_ASCEND_HUST_REPO" \
  CURRENT_DEVICE_PREFERENCE_FILE="$ASCEND_DEVICE_PREFERENCE_FILE" \
  CURRENT_GITHUB_REF="$CURRENT_GITHUB_REF" \
  CURRENT_PLUGIN_GITHUB_REF="$CURRENT_PLUGIN_GITHUB_REF" \
  VLLM_ALLOW_LONG_MAX_MODEL_LEN="$CONTEXT_SWEEP_ALLOW_LONG_MAX_MODEL_LEN" \
  CURRENT_SERVER_PORT="$port" \
  CURRENT_CLIENT_PORT="$port" \
  READY_TIMEOUT_SECONDS="$READY_TIMEOUT_SECONDS" \
  READY_STATUS_INTERVAL_SECONDS="$READY_STATUS_INTERVAL_SECONDS" \
  CLIENT_READY_CHECK_TIMEOUT_SECONDS="$CLIENT_READY_CHECK_TIMEOUT_SECONDS" \
  DEVICE_SELECTION_RETRIES="$DEVICE_SELECTION_RETRIES" \
  DEVICE_SELECTION_RETRY_DELAY_SECONDS="$DEVICE_SELECTION_RETRY_DELAY_SECONDS" \
  bash "$CURRENT_RUNNER" "$spec_file" 2>&1 | tee "$runner_log"
  local runner_status=${PIPESTATUS[0]}
  set -e

  if [[ "$runner_status" -ne 0 ]]; then
    record_failure "current:${spec_slug}" "exit=${runner_status}"
    return "$runner_status"
  fi

  ((RUN_COUNT+=1))
  append_summary "  - current run: ${result_dir}"
  return 0
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$PREPARE_SCRIPT" ]]; then
  echo "Prepare script not found: $PREPARE_SCRIPT" >&2
  exit 2
fi

if [[ ! -f "$OFFICIAL_RUNNER" ]]; then
  echo "Official runner not found: $OFFICIAL_RUNNER" >&2
  exit 2
fi

if [[ ! -f "$CURRENT_RUNNER" ]]; then
  echo "Current runner not found: $CURRENT_RUNNER" >&2
  exit 2
fi

if [[ -z "$GOAL_BASELINE_ENV_PREFIX" ]]; then
  echo "GOAL_BASELINE_ENV_PREFIX is required" >&2
  exit 2
fi

if [[ -z "$HOST_PYTHON_BIN" ]] || [[ ! -x "$HOST_PYTHON_BIN" ]]; then
  echo "HOST_PYTHON_BIN is not executable: $HOST_PYTHON_BIN" >&2
  exit 2
fi

if [[ ! -x "$CURRENT_RUNTIME_PYTHON" ]]; then
  echo "CURRENT_RUNTIME_PYTHON is not executable: $CURRENT_RUNTIME_PYTHON" >&2
  exit 2
fi

if [[ ! -d "$CURRENT_VLLM_HUST_REPO" ]]; then
  echo "CURRENT_VLLM_HUST_REPO not found: $CURRENT_VLLM_HUST_REPO" >&2
  exit 2
fi

if [[ ! -d "$CURRENT_VLLM_ASCEND_HUST_REPO" ]]; then
  echo "CURRENT_VLLM_ASCEND_HUST_REPO not found: $CURRENT_VLLM_ASCEND_HUST_REPO" >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 2
fi

if [[ $# -lt 1 ]]; then
  set -- "$DEFAULT_SPEC_DIR"
fi

mkdir -p "$MATRIX_RESULT_ROOT"
mkdir -p "$(dirname "$SUMMARY_FILE")"
: > "$SUMMARY_FILE"

mapfile -t SPEC_FILES < <(
  for path in "$@"; do
    collect_specs "$path"
  done | awk 'NF' | sort -u
)

if [[ ${#SPEC_FILES[@]} -eq 0 ]]; then
  echo "No spec files resolved from input arguments." >&2
  exit 2
fi

append_summary "## Ascend Context Length Sweep"
append_summary "- Result root: $MATRIX_RESULT_ROOT"
append_summary "- Website output dir: $WEBSITE_OUTPUT_DIR"
append_summary "- Spec count: ${#SPEC_FILES[@]}"
append_summary "- Current runtime repo: $CURRENT_VLLM_HUST_REPO"
append_summary "- Current plugin repo: $CURRENT_VLLM_ASCEND_HUST_REPO"
append_summary "- Official env prefix: $GOAL_BASELINE_ENV_PREFIX"
append_summary "- Current env prefix: $CURRENT_ENV_PREFIX"
if [[ -n "$RESUME_CHECKPOINT_ROOT" ]]; then
  append_summary "- Resume checkpoint root: $RESUME_CHECKPOINT_ROOT"
fi

echo "[context-sweep] result root: $MATRIX_RESULT_ROOT"
echo "[context-sweep] resolved ${#SPEC_FILES[@]} spec file(s)"

for index in "${!SPEC_FILES[@]}"; do
  spec_file="${SPEC_FILES[$index]}"
  spec_slug=$(slugify "$spec_file")
  spec_id=$(jq -r '.id' "$spec_file")
  input_len=$(jq -r '.client_parameters.input_len' "$spec_file")
  output_len=$(jq -r '.client_parameters.output_len' "$spec_file")
  spec_hash=$(resolve_spec_hash "$spec_file")

  # Issue 1: check remaining time budget before starting a new spec pair
  elapsed_seconds=$(( $(date +%s) - JOB_START_EPOCH ))
  remaining_seconds=$(( JOB_TIMEOUT_SECONDS - elapsed_seconds ))
  if (( remaining_seconds < SPEC_TIMEOUT_SECONDS )); then
    echo
    echo "[context-sweep] WARNING: only ${remaining_seconds}s remaining (need ${SPEC_TIMEOUT_SECONDS}s per spec); skipping remaining specs" >&2
    append_summary "### SKIPPED (time budget)"
    append_summary "- Specs skipped from index ${index}: insufficient time budget (${remaining_seconds}s < ${SPEC_TIMEOUT_SECONDS}s)"
    for skip_idx in $(seq "$index" "$((${#SPEC_FILES[@]} - 1))"); do
      skip_file="${SPEC_FILES[$skip_idx]}"
      skip_id=$(jq -r '.id' "$skip_file")
      record_failure "${skip_id}" "time-budget-exceeded"
    done
    break
  fi

  echo
  echo "[context-sweep] spec: $spec_file"
  echo "[context-sweep] spec id: $spec_id"
  echo "[context-sweep] workload: input_len=${input_len}, output_len=${output_len}"
  echo "[context-sweep] time budget: ${remaining_seconds}s remaining"
  append_summary "### ${spec_id}"
  append_summary "- Spec file: ${spec_file}"
  append_summary "- Workload: input_len=${input_len}, output_len=${output_len}"

  if ! restore_result_dir_from_checkpoint "official" "$spec_file" "$spec_slug" "$spec_hash"; then
    prepare_official_env_once
    if ! run_official_for_spec "$spec_file" "$spec_slug" "$index"; then
      append_summary "  - official log: $MATRIX_RESULT_ROOT/official/$spec_slug/runner.log"
    fi
    # Issue 8: brief port-release wait after runner completes
    sleep 3
  fi

  if ! restore_result_dir_from_checkpoint "current" "$spec_file" "$spec_slug" "$spec_hash"; then
    echo "[context-sweep] waiting 10s for NPU resource release before current run"
    sleep 10
    if ! run_current_for_spec "$spec_file" "$spec_slug" "$index"; then
      append_summary "  - current log: $MATRIX_RESULT_ROOT/current/$spec_slug/runner.log"
    fi
    # Issue 8: brief port-release wait after runner completes
    sleep 3
  fi

  # Issue 6: write intermediate progress to GITHUB_STEP_SUMMARY
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    echo "**Progress: spec $((index + 1))/${#SPEC_FILES[@]} complete** (${spec_id})" >> "$GITHUB_STEP_SUMMARY"
    echo "" >> "$GITHUB_STEP_SUMMARY"
  fi

  # Issue 2: write intermediate results marker for artifact upload
  echo "$((index + 1))/${#SPEC_FILES[@]} ${spec_id}" > "$MATRIX_RESULT_ROOT/.last-completed-spec"
done

if [[ "$PUBLISH_WEBSITE" == "1" ]]; then
  manifest_count=$(find "$MATRIX_RESULT_ROOT" -type f -name 'leaderboard_manifest.json' | wc -l | tr -d '[:space:]')
  if [[ "$manifest_count" == "0" ]]; then
    echo
    echo "[context-sweep] skipping website aggregation because no leaderboard manifests were generated"
    append_summary "- Website aggregation skipped: no leaderboard_manifest.json files were generated"
  else
    echo
    echo "[context-sweep] aggregating website snapshots from $MATRIX_RESULT_ROOT"
    mkdir -p "$WEBSITE_OUTPUT_DIR"
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$HOST_PYTHON_BIN" -m vllm_hust_benchmark.cli publish-website \
        --source-dir "$MATRIX_RESULT_ROOT" \
        --output-dir "$WEBSITE_OUTPUT_DIR" \
        --execute

    compare_group_count=$(PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" WEBSITE_OUTPUT_DIR="$WEBSITE_OUTPUT_DIR" "$HOST_PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

compare_path = Path(os.environ["WEBSITE_OUTPUT_DIR"]) / "leaderboard_compare.json"
if not compare_path.is_file():
    print(0)
else:
    payload = json.loads(compare_path.read_text(encoding="utf-8"))
    print(int(payload.get("group_count") or 0))
PY
)
    append_summary "- Website snapshots rebuilt at: $WEBSITE_OUTPUT_DIR"
    append_summary "- Compare group count: ${compare_group_count}"
  fi
fi

append_summary "- Successful runs: $((RUN_COUNT + RESUMED_COUNT))"
append_summary "- Executed runs: $RUN_COUNT"
append_summary "- Resumed runs: $RESUMED_COUNT"
append_summary "- Failed runs: $FAILED_COUNT"
if (( FAILED_COUNT > 0 )); then
  for failed_item in "${FAILED_ITEMS[@]}"; do
    append_summary "  - $failed_item"
  done
fi
if (( RESUMED_COUNT > 0 )); then
  for resumed_item in "${RESUMED_ITEMS[@]}"; do
    append_summary "  - $resumed_item"
  done
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  cat "$SUMMARY_FILE" >> "$GITHUB_STEP_SUMMARY"
fi

echo
echo "[context-sweep] successful runs: $((RUN_COUNT + RESUMED_COUNT))"
echo "[context-sweep] executed runs: $RUN_COUNT"
echo "[context-sweep] resumed runs: $RESUMED_COUNT"
echo "[context-sweep] failed runs: $FAILED_COUNT"
echo "[context-sweep] summary: $SUMMARY_FILE"

if (( FAILED_COUNT > 0 )); then
  exit 1
fi