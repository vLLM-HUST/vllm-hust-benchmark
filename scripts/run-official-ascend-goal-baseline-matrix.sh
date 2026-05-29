#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=${VLLM_HUST_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}
PREPARE_SCRIPT=${PREPARE_SCRIPT:-"$SCRIPT_DIR/prepare-official-ascend-baseline-env.sh"}
SINGLE_RUNNER=${SINGLE_RUNNER:-"$SCRIPT_DIR/run-official-ascend-goal-baseline.sh"}
MATRIX_RUN_ID=${MATRIX_RUN_ID:-"official-ascend-goal-baseline-matrix-$(date -u +%Y%m%dT%H%M%SZ)"}
MATRIX_RESULT_ROOT=${MATRIX_RESULT_ROOT:-"$REPO_ROOT/.benchmarks/$MATRIX_RUN_ID"}
CANONICAL_SUBMISSIONS_ROOT=${CANONICAL_SUBMISSIONS_ROOT:-"$REPO_ROOT/submissions"}
EXISTING_CANONICAL_SUBMISSIONS_ROOT=${EXISTING_CANONICAL_SUBMISSIONS_ROOT:-"$CANONICAL_SUBMISSIONS_ROOT"}
PUBLISH_WEBSITE=${PUBLISH_WEBSITE:-0}
PUBLISH_RESULTS=${PUBLISH_RESULTS:-0}
WEBSITE_OUTPUT_DIR=${WEBSITE_OUTPUT_DIR:-"$WORKSPACE_ROOT/vllm-hust-website/data"}
FORCE_RUN_EXISTING=${FORCE_RUN_EXISTING:-0}
REPEAT_COUNT=${REPEAT_COUNT:-3}
MIN_SUCCESSFUL_REPEATS=${MIN_SUCCESSFUL_REPEATS:-}
MAX_REPEAT_ATTEMPTS=${MAX_REPEAT_ATTEMPTS:-}
PREPARE_OFFICIAL_ENV=${PREPARE_OFFICIAL_ENV:-1}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-900}
SUMMARY_FILE=${MATRIX_SUMMARY_FILE:-"$MATRIX_RESULT_ROOT/summary.md"}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || true)}
PUBLICATION_SYNC_HELPER=${PUBLICATION_SYNC_HELPER:-}
PUBLICATION_COMMIT_MESSAGE_PREFIX=${PUBLICATION_COMMIT_MESSAGE_PREFIX:-"chore(data): publish official ascend baseline"}
PREPARED_ENV=0
SKIPPED_COUNT=0
RUN_COUNT=0
PROMOTED_COUNT=0
REVIEW_COUNT=0
FAILED_COUNT=0
FAILED_SPECS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-official-ascend-goal-baseline-matrix.sh [<spec-file-or-dir> ...]

Behavior:
  - Defaults to docs/official-baselines when no path is provided
  - Skips specs that already have canonical submissions under submissions/<spec-id>/
  - Runs specs without canonical submissions through scripts/run-official-ascend-goal-baseline.sh
  - Repeats missing-canonical specs REPEAT_COUNT times (default: 3)
  - Allows bounded repeat failures and still selects a canonical candidate when enough successful repeats exist
  - Chooses the repeat whose primary metric is closest to the median run and promotes only that candidate into canonical submissions/<spec-id>/
  - Set FORCE_RUN_EXISTING=1 to run specs even when a canonical submission already exists
    Existing canonical submissions are preserved and the new run is left in .benchmarks/ for manual review
  - Set PUBLISH_WEBSITE=1 to rebuild website data from the full submissions/ tree after the batch
  - Set PUBLISH_RESULTS=1 together with publication helper env vars to publish each promoted spec immediately

Examples:
  export GOAL_BASELINE_ENV_PREFIX=/root/miniconda3/envs/vllm-ascend-official-v0110
  bash scripts/run-official-ascend-goal-baseline-matrix.sh

  REPEAT_COUNT=5 bash scripts/run-official-ascend-goal-baseline-matrix.sh \
    docs/official-baselines/official-ascend-jan-2026-v0110-sharegpt-online-qwen25-14b-910b3.json

  FORCE_RUN_EXISTING=1 bash scripts/run-official-ascend-goal-baseline-matrix.sh \
    docs/official-baselines/official-ascend-jan-2026-v0110-random-online-qwen25-14b-910b3.json
EOF
}

if [[ ! -x "$SINGLE_RUNNER" ]]; then
  echo "Single official baseline runner is not executable: $SINGLE_RUNNER" >&2
  exit 2
fi

if [[ ! -f "$PREPARE_SCRIPT" ]]; then
  echo "Prepare script not found: $PREPARE_SCRIPT" >&2
  exit 2
fi

if [[ -z "$PYTHON_BIN" ]] || [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python3 is required for official baseline matrix orchestration" >&2
  exit 2
fi

if [[ -z "${GOAL_BASELINE_ENV_PREFIX:-}" ]]; then
  echo "GOAL_BASELINE_ENV_PREFIX is required" >&2
  exit 2
fi

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

resolve_spec_id() {
  local spec_file=$1
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY' "$spec_file"
from pathlib import Path
import sys

from vllm_hust_benchmark.official_baselines import get_official_baseline_spec_id
from vllm_hust_benchmark.official_baselines import load_official_baseline_spec

print(get_official_baseline_spec_id(load_official_baseline_spec(Path(sys.argv[1]))))
PY
}

resolve_canonical_dir() {
  local spec_file=$1
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY' "$spec_file" "$CANONICAL_SUBMISSIONS_ROOT"
from pathlib import Path
import sys

from vllm_hust_benchmark.official_baselines import get_canonical_submission_dir
from vllm_hust_benchmark.official_baselines import load_official_baseline_spec

spec = load_official_baseline_spec(Path(sys.argv[1]))
print(get_canonical_submission_dir(spec, submissions_root=Path(sys.argv[2])).resolve())
PY
}

resolve_existing_canonical_dir() {
  local spec_file=$1
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY' "$spec_file" "$EXISTING_CANONICAL_SUBMISSIONS_ROOT"
from pathlib import Path
import sys

from vllm_hust_benchmark.official_baselines import get_canonical_submission_dir
from vllm_hust_benchmark.official_baselines import load_official_baseline_spec

spec = load_official_baseline_spec(Path(sys.argv[1]))
print(get_canonical_submission_dir(spec, submissions_root=Path(sys.argv[2])).resolve())
PY
}

has_canonical_run() {
  local spec_file=$1
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY' "$spec_file" "$EXISTING_CANONICAL_SUBMISSIONS_ROOT"
from pathlib import Path
import sys

from vllm_hust_benchmark.official_baselines import has_canonical_run
from vllm_hust_benchmark.official_baselines import load_official_baseline_spec

spec = load_official_baseline_spec(Path(sys.argv[1]))
raise SystemExit(0 if has_canonical_run(spec, submissions_root=Path(sys.argv[2])) else 1)
PY
}

prepare_official_env_once() {
  if [[ "$PREPARED_ENV" == "1" ]] || [[ "$PREPARE_OFFICIAL_ENV" != "1" ]]; then
    return 0
  fi

  echo "[official-baseline-matrix] preparing official baseline environment: $GOAL_BASELINE_ENV_PREFIX"
  ENV_PREFIX="$GOAL_BASELINE_ENV_PREFIX" \
  VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
  bash "$PREPARE_SCRIPT"
  PREPARED_ENV=1
}

promote_submission_to_canonical() {
  local result_dir=$1
  local canonical_dir=$2
  local spec_id=$3
  local submission_dir="$result_dir/submission"
  local backup_dir

  if [[ ! -f "$submission_dir/run_leaderboard.json" ]] || [[ ! -f "$submission_dir/leaderboard_manifest.json" ]]; then
    echo "Canonical promotion source is incomplete for $spec_id: $submission_dir" >&2
    return 1
  fi

  if [[ -e "$canonical_dir" ]]; then
    backup_dir="${canonical_dir}.backup-$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$canonical_dir" "$backup_dir"
    echo "[official-baseline-matrix] moved pre-existing non-canonical path aside: $backup_dir"
  fi

  mkdir -p "$(dirname "$canonical_dir")"
  cp -a "$submission_dir" "$canonical_dir"
  echo "[official-baseline-matrix] promoted canonical submission: $spec_id -> $canonical_dir"
}

append_summary() {
  printf '%s\n' "$1" >> "$SUMMARY_FILE"
}

record_failed_spec() {
  local spec_id=$1
  local reason=$2

  FAILED_SPECS+=("${spec_id}: ${reason}")
  ((FAILED_COUNT+=1))
  append_summary "- Failed spec: ${spec_id} (${reason})"
}

extract_repeat_failure_reason() {
  local log_file=$1
  local line=""

  [[ -f "$log_file" ]] || return 1

  line=$(grep -m1 "Timed out waiting for official baseline server at " "$log_file" || true)
  if [[ -n "$line" ]]; then
    printf 'server readiness timeout after %ss: %s\n' "${READY_TIMEOUT_SECONDS:-unknown}" "$line"
    return 0
  fi

  line=$(grep -m1 "Ascend runtime did not become ready after " "$log_file" || true)
  if [[ -n "$line" ]]; then
    printf '%s\n' "$line"
    return 0
  fi

  line=$(grep -m1 "All detected Ascend devices are busy" "$log_file" || true)
  if [[ -n "$line" ]]; then
    printf '%s\n' "$line"
    return 0
  fi

  line=$(grep -m1 "No idle Ascend device is currently available" "$log_file" || true)
  if [[ -n "$line" ]]; then
    printf '%s\n' "$line"
    return 0
  fi

  line=$(tail -n 1 "$log_file" | tr -d '\r')
  if [[ -n "$line" ]]; then
    printf '%s\n' "$line"
    return 0
  fi

  return 1
}

append_repeat_failure_summary() {
  local spec_id=$1
  local repeat_label=$2
  local result_dir=$3
  local exit_status=$4
  local runner_log="$result_dir/runner.log"
  local server_log="$result_dir/server.stdout.log"
  local reason=""
  local model_phase=""

  append_summary "  - ${repeat_label} failed for ${spec_id} (exit=${exit_status})"

  reason=$(extract_repeat_failure_reason "$runner_log" || true)
  if [[ -n "$reason" ]]; then
    append_summary "    - ${reason}"
  fi

  if [[ -f "$server_log" ]]; then
    model_phase=$(grep -m1 "Starting to load model " "$server_log" || true)
    if [[ -n "$model_phase" ]]; then
      append_summary "    - last observed server phase: ${model_phase}"
    fi
  fi

  append_summary "    - runner log: ${result_dir}/runner.log"
}

append_repeat_result_diagnostic() {
  local repeat_label=$1
  local result_dir=$2
  local artifact_file="$result_dir/submission/run_leaderboard.json"
  local reason=""

  if [[ -f "$artifact_file" ]]; then
    append_summary "    - ${repeat_label}: leaderboard artifact produced"
    return 0
  fi

  append_summary "    - ${repeat_label}: no leaderboard artifact produced"
  reason=$(extract_repeat_failure_reason "$result_dir/runner.log" || true)
  if [[ -n "$reason" ]]; then
    append_summary "      reason: ${reason}"
  fi
}

append_selection_failure_summary() {
  local spec_id=$1
  local error_log=$2
  shift 2
  local selection_reason=""
  local result_dir=""

  append_summary "  - Candidate selection failed for ${spec_id}"

  if [[ -f "$error_log" ]]; then
    selection_reason=$(grep -E "ValueError:|RuntimeError:|Error:" "$error_log" | tail -n 1 | tr -d '\r' || true)
    if [[ -n "$selection_reason" ]]; then
      append_summary "    - ${selection_reason}"
    fi
  fi

  for result_dir in "$@"; do
    append_repeat_result_diagnostic "$(basename "$result_dir")" "$result_dir"
  done
}

publish_promoted_spec() {
  local spec_id=$1
  local snapshot_output_dir=${LOCAL_SNAPSHOT_OUTPUT_DIR:-}
  local spec_snapshot_output_dir=""

  if [[ "$PUBLISH_RESULTS" != "1" ]]; then
    return 0
  fi

  if [[ -z "$PUBLICATION_SYNC_HELPER" ]] || [[ ! -f "$PUBLICATION_SYNC_HELPER" ]]; then
    echo "Publication sync helper is required when PUBLISH_RESULTS=1: $PUBLICATION_SYNC_HELPER" >&2
    return 2
  fi

  if [[ -n "$snapshot_output_dir" ]]; then
    spec_snapshot_output_dir="$snapshot_output_dir/$spec_id"
  fi

  SOURCE_BENCHMARK_REPO_DIR="$REPO_ROOT" \
  SNAPSHOT_SOURCE_PATTERN="$spec_id" \
  SNAPSHOT_COMMIT_MESSAGE="${PUBLICATION_COMMIT_MESSAGE_PREFIX} ${spec_id} from ${MATRIX_RUN_ID}" \
  LOCAL_SNAPSHOT_OUTPUT_DIR="$spec_snapshot_output_dir" \
  bash "$PUBLICATION_SYNC_HELPER"
}

resolve_benchmark_type() {
  local spec_file=$1
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY' "$spec_file"
from pathlib import Path
import sys

from vllm_hust_benchmark.official_baselines import get_official_baseline_benchmark_type
from vllm_hust_benchmark.official_baselines import load_official_baseline_spec

spec = load_official_baseline_spec(Path(sys.argv[1]))
print(get_official_baseline_benchmark_type(spec))
PY
}

select_canonical_candidate() {
  local benchmark_type=$1
  shift
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY' "$benchmark_type" "$@"
from pathlib import Path
import json
import sys

from vllm_hust_benchmark.official_baselines import select_canonical_candidate

benchmark_type = sys.argv[1]
result_dirs = [Path(item) for item in sys.argv[2:]]
payload = select_canonical_candidate(result_dirs, benchmark_type=benchmark_type)
print(json.dumps(payload))
PY
}

if [[ $# -lt 1 ]]; then
  set -- "$REPO_ROOT/docs/official-baselines"
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
  echo "No official baseline spec files resolved from input arguments." >&2
  exit 2
fi

echo "[official-baseline-matrix] result root: $MATRIX_RESULT_ROOT"
echo "[official-baseline-matrix] canonical submissions root: $CANONICAL_SUBMISSIONS_ROOT"
echo "[official-baseline-matrix] resolved ${#SPEC_FILES[@]} spec file(s)"
append_summary "## Official Baseline Matrix"
append_summary "- Result root: $MATRIX_RESULT_ROOT"
append_summary "- Canonical submissions root: $CANONICAL_SUBMISSIONS_ROOT"
append_summary "- Existing canonical submissions root: $EXISTING_CANONICAL_SUBMISSIONS_ROOT"
append_summary "- Force rerun existing canonical: $FORCE_RUN_EXISTING"
append_summary "- Repeat count for missing canonical specs: $REPEAT_COUNT"

for spec_file in "${SPEC_FILES[@]}"; do
  spec_slug=$(slugify "$spec_file")
  spec_id=$(resolve_spec_id "$spec_file")
  canonical_dir=$(resolve_canonical_dir "$spec_file")
  existing_canonical_dir=$(resolve_existing_canonical_dir "$spec_file")
  benchmark_type=$(resolve_benchmark_type "$spec_file")

  echo
  echo "[official-baseline-matrix] spec: $spec_file"
  echo "[official-baseline-matrix] spec id: $spec_id"
  echo "[official-baseline-matrix] benchmark type: $benchmark_type"

  if has_canonical_run "$spec_file"; then
    if [[ "$FORCE_RUN_EXISTING" != "1" ]]; then
      echo "[official-baseline-matrix] skip: canonical run already exists at $existing_canonical_dir"
      append_summary "- Skip existing canonical: $spec_id -> $existing_canonical_dir"
      ((SKIPPED_COUNT+=1))
      continue
    fi

    echo "[official-baseline-matrix] canonical run already exists; re-running for manual review only"
    append_summary "- Re-run for review only: $spec_id (canonical preserved at $existing_canonical_dir)"
    should_promote=0
  else
    echo "[official-baseline-matrix] no canonical run found; executing and promoting on success"
    append_summary "- Missing canonical, will run: $spec_id"
    should_promote=1
  fi

  prepare_official_env_once

  target_successful_repeats=1
  min_successful_repeats=1
  max_repeat_attempts=1
  if [[ "$should_promote" == "1" ]]; then
    target_successful_repeats="$REPEAT_COUNT"
    if [[ -n "$MIN_SUCCESSFUL_REPEATS" ]]; then
      min_successful_repeats="$MIN_SUCCESSFUL_REPEATS"
    elif (( REPEAT_COUNT < 2 )); then
      min_successful_repeats="$REPEAT_COUNT"
    else
      min_successful_repeats=2
    fi

    if [[ -n "$MAX_REPEAT_ATTEMPTS" ]]; then
      max_repeat_attempts="$MAX_REPEAT_ATTEMPTS"
    else
      max_repeat_attempts=$((REPEAT_COUNT + 1))
    fi
  fi

  if (( target_successful_repeats < 1 )); then
    target_successful_repeats=1
  fi
  if (( min_successful_repeats < 1 )); then
    min_successful_repeats=1
  fi
  if (( min_successful_repeats > target_successful_repeats )); then
    min_successful_repeats=$target_successful_repeats
  fi
  if (( max_repeat_attempts < target_successful_repeats )); then
    max_repeat_attempts=$target_successful_repeats
  fi

  append_summary "  - Repeat policy for $spec_id: target_successes=${target_successful_repeats}, min_successes=${min_successful_repeats}, max_attempts=${max_repeat_attempts}"

  repeat_result_dirs=()
  successful_repeat_count=0
  repeat_failure_count=0
  for repeat_index in $(seq 1 "$max_repeat_attempts"); do
    if (( successful_repeat_count >= target_successful_repeats )); then
      break
    fi

    repeat_label=$(printf 'repeat-%02d' "$repeat_index")
    result_dir="$MATRIX_RESULT_ROOT/$spec_slug/$repeat_label"
    run_id="$MATRIX_RUN_ID-$spec_slug-$repeat_label"
    runner_log="$result_dir/runner.log"

    echo "[official-baseline-matrix] running $repeat_label for $spec_id"
    append_summary "  - Executing $repeat_label for $spec_id -> $result_dir"

    mkdir -p "$result_dir"

    set +e
    RESULT_DIR="$result_dir" \
    RUN_ID="$run_id" \
    GOAL_BASELINE_ENV_PREFIX="$GOAL_BASELINE_ENV_PREFIX" \
    VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
    bash "$SINGLE_RUNNER" "$spec_file" 2>&1 | tee "$runner_log"
    runner_status=${PIPESTATUS[0]}
    set -e

    if [[ "$runner_status" -ne 0 ]]; then
      append_repeat_failure_summary "$spec_id" "$repeat_label" "$result_dir" "$runner_status"
      ((repeat_failure_count+=1))
      remaining_attempts=$((max_repeat_attempts - repeat_index))
      if (( successful_repeat_count + remaining_attempts < min_successful_repeats )); then
        append_summary "    - stopping early: cannot reach minimum successful repeats for $spec_id after ${repeat_label}"
        break
      fi
      continue
    fi

    repeat_result_dirs+=("$result_dir")
    ((successful_repeat_count+=1))
    ((RUN_COUNT+=1))
  done

  if (( successful_repeat_count < min_successful_repeats )); then
    record_failed_spec "$spec_id" "only ${successful_repeat_count}/${min_successful_repeats} successful repeat(s) across ${max_repeat_attempts} attempt(s)"
    continue
  fi

  if (( successful_repeat_count < target_successful_repeats )); then
    append_summary "  - Proceeding with degraded sample count for $spec_id: ${successful_repeat_count}/${target_successful_repeats} successful repeat(s), failures=${repeat_failure_count}"
  fi

  selection_error_log="$MATRIX_RESULT_ROOT/$spec_slug/candidate-selection.stderr.log"
  set +e
  selection_json=$(select_canonical_candidate "$benchmark_type" "${repeat_result_dirs[@]}" 2>"$selection_error_log")
  selection_status=$?
  set -e
  if [[ "$selection_status" -ne 0 ]]; then
    append_selection_failure_summary "$spec_id" "$selection_error_log" "${repeat_result_dirs[@]}"
    record_failed_spec "$spec_id" "candidate selection failed"
    continue
  fi
  selected_result_dir=$(printf '%s' "$selection_json" | jq -r '.selected_result_dir')
  primary_metric_name=$(printf '%s' "$selection_json" | jq -r '.primary_metric_name')
  median_value=$(printf '%s' "$selection_json" | jq -r '.median_value')
  append_summary "  - Candidate selection for $spec_id uses $primary_metric_name median=$median_value"
  while IFS= read -r candidate_line; do
    [[ -z "$candidate_line" ]] && continue
    append_summary "    - $candidate_line"
  done < <(printf '%s' "$selection_json" | jq -r '.candidates[] | (.result_dir + " | metric=" + (.primary_metric_value|tostring) + " | error_rate=" + (.error_rate|tostring) + " | distance_to_median=" + (.distance_to_median|tostring))')

  if [[ "$should_promote" == "1" ]]; then
    promote_submission_to_canonical "$selected_result_dir" "$canonical_dir" "$spec_id"
    append_summary "  - Selected canonical candidate: $selected_result_dir"
    ((PROMOTED_COUNT+=1))

    if [[ "$PUBLISH_RESULTS" == "1" ]]; then
      if publish_promoted_spec "$spec_id"; then
        append_summary "  - Published canonical submission for $spec_id"
      else
        publish_status=$?
        record_failed_spec "$spec_id" "publication failed with exit=${publish_status}"
        continue
      fi
    fi
  else
    echo "[official-baseline-matrix] review hint: selected rerun candidate is $selected_result_dir; canonical remains $canonical_dir"
    append_summary "  - Review hint: selected rerun candidate is $selected_result_dir; canonical remains $canonical_dir"
    ((REVIEW_COUNT+=1))
  fi
done

if [[ "$PUBLISH_WEBSITE" == "1" ]]; then
  echo
  echo "[official-baseline-matrix] rebuilding website data from full submissions tree"
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m vllm_hust_benchmark.cli publish-website \
    --source-dir "$CANONICAL_SUBMISSIONS_ROOT" \
    --output-dir "$WEBSITE_OUTPUT_DIR" \
    --execute
  append_summary "- Website data rebuilt at: $WEBSITE_OUTPUT_DIR"
fi

append_summary "- Skipped existing canonical runs: $SKIPPED_COUNT"
append_summary "- Executed runs: $RUN_COUNT"
append_summary "- Promoted new canonical runs: $PROMOTED_COUNT"
append_summary "- Review-only reruns: $REVIEW_COUNT"
append_summary "- Failed specs: $FAILED_COUNT"
if (( FAILED_COUNT > 0 )); then
  for failed_spec in "${FAILED_SPECS[@]}"; do
    append_summary "  - $failed_spec"
  done
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  cat "$SUMMARY_FILE" >> "$GITHUB_STEP_SUMMARY"
fi

echo
echo "[official-baseline-matrix] skipped existing canonical runs: $SKIPPED_COUNT"
echo "[official-baseline-matrix] executed runs: $RUN_COUNT"
echo "[official-baseline-matrix] promoted new canonical runs: $PROMOTED_COUNT"
echo "[official-baseline-matrix] review-only reruns: $REVIEW_COUNT"
echo "[official-baseline-matrix] failed specs: $FAILED_COUNT"
echo "[official-baseline-matrix] summary: $SUMMARY_FILE"

if (( FAILED_COUNT > 0 )); then
  exit 1
fi