#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SPEC_FILE=${1:-"$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0110-random-online-qwen25-14b-910b3.json"}
CONFIG_FILE=${CONFIG_FILE:-"$SCRIPT_DIR/run-current-ascend-same-spec-msprof.env"}

if [[ -f "$CONFIG_FILE" ]]; then
  # The default config uses shell defaults, so exported environment variables
  # can still override it without editing the file.
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
fi

SPEC_FILE=$(realpath -m "$SPEC_FILE")

RUNNER_SCRIPT=${RUNNER_SCRIPT:-"$SCRIPT_DIR/run-current-ascend-same-spec.sh"}
MSPROF_EXECUTABLE=${MSPROF_EXECUTABLE:-msprof}
MSPROF_FLAGS=${MSPROF_FLAGS:-"--ascendcl=on --runtime-api=on --task-time=l1 --hccl=on --type=text"}
DRY_RUN=${DRY_RUN:-0}

PROFILE_RUN_ID=${PROFILE_RUN_ID:-${RUN_ID:-"current-ascend-msprof-$(date -u +%Y%m%dT%H%M%SZ)"}}
PROFILE_RUN_ROOT=${PROFILE_RUN_ROOT:-"$REPO_ROOT/.benchmarks/current-ascend-msprof"}
PROFILE_RUN_DIR=${PROFILE_RUN_DIR:-"$PROFILE_RUN_ROOT/$PROFILE_RUN_ID"}
MSPROF_RAW_DIR=${MSPROF_RAW_DIR:-"$PROFILE_RUN_DIR/msprof_raw"}
BENCHMARK_RESULT_DIR=${BENCHMARK_RESULT_DIR:-"$PROFILE_RUN_DIR/benchmark"}
MSPROF_LOG_FILE=${MSPROF_LOG_FILE:-"$PROFILE_RUN_DIR/msprof.log"}
WORKLOAD_WRAPPER=${WORKLOAD_WRAPPER:-"$PROFILE_RUN_DIR/run_benchmark_under_msprof.sh"}
RUN_META_FILE=${RUN_META_FILE:-"$PROFILE_RUN_DIR/run_meta.env"}

bool_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

write_shell_var() {
  local name=$1
  local value=$2
  printf '%s=' "$name"
  printf '%q\n' "$value"
}

if [[ ! -f "$SPEC_FILE" ]]; then
  echo "Spec file not found: $SPEC_FILE" >&2
  exit 2
fi

if [[ ! -x "$RUNNER_SCRIPT" ]]; then
  echo "Runner script is not executable: $RUNNER_SCRIPT" >&2
  exit 2
fi

if ! bool_true "$DRY_RUN" && ! command -v "$MSPROF_EXECUTABLE" >/dev/null 2>&1; then
  echo "$MSPROF_EXECUTABLE not found; activate the Ascend profiler environment first." >&2
  exit 2
fi

mkdir -p "$PROFILE_RUN_DIR" "$MSPROF_RAW_DIR" "$BENCHMARK_RESULT_DIR"

cat > "$WORKLOAD_WRAPPER" <<EOF
#!/bin/bash
set -euo pipefail
export RUN_ID=$(printf '%q' "$PROFILE_RUN_ID")
export RESULT_DIR=\${RESULT_DIR:-$(printf '%q' "$BENCHMARK_RESULT_DIR")}
exec $(printf '%q' "$RUNNER_SCRIPT") $(printf '%q' "$SPEC_FILE")
EOF
chmod +x "$WORKLOAD_WRAPPER"

{
  write_shell_var profile_run_id "$PROFILE_RUN_ID"
  write_shell_var profile_run_dir "$PROFILE_RUN_DIR"
  write_shell_var spec_file "$SPEC_FILE"
  write_shell_var config_file "$CONFIG_FILE"
  write_shell_var runner_script "$RUNNER_SCRIPT"
  write_shell_var benchmark_result_dir "$BENCHMARK_RESULT_DIR"
  write_shell_var msprof_executable "$MSPROF_EXECUTABLE"
  write_shell_var msprof_flags "$MSPROF_FLAGS"
  write_shell_var msprof_raw_dir "$MSPROF_RAW_DIR"
  write_shell_var msprof_log_file "$MSPROF_LOG_FILE"
  write_shell_var workload_wrapper "$WORKLOAD_WRAPPER"
  write_shell_var created_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RUN_META_FILE"

read -r -a msprof_args <<< "$MSPROF_FLAGS"
msprof_command=(
  "$MSPROF_EXECUTABLE"
  "--output=$MSPROF_RAW_DIR"
  "--application=$WORKLOAD_WRAPPER"
  "${msprof_args[@]}"
)

echo "profile_run_dir: $PROFILE_RUN_DIR"
echo "msprof_raw_dir: $MSPROF_RAW_DIR"
echo "benchmark_result_dir: $BENCHMARK_RESULT_DIR"
echo "+ ${msprof_command[*]} > $MSPROF_LOG_FILE 2>&1"

if ! bool_true "$DRY_RUN"; then
  if "${msprof_command[@]}" >"$MSPROF_LOG_FILE" 2>&1; then
    :
  else
    status=$?
    echo "msprof run failed with exit code $status. Last log lines:" >&2
    tail -n 80 "$MSPROF_LOG_FILE" >&2 || true
    exit "$status"
  fi
fi

echo "msprof log: $MSPROF_LOG_FILE"
echo "msprof raw output: $MSPROF_RAW_DIR"

# Optional offline TraceLoom analysis, run outside this benchmark repository:
#
#   traceloom analysis "$MSPROF_RAW_DIR" --out-dir "$PROFILE_RUN_DIR/analysis"
#
# Add TraceLoom-specific device filters or analysis limits in that command, not
# in this benchmark runner.
