#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SPEC_FILE=${1:-"$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0110-random-online-qwen25-14b-910b3.json"}
CONSTRAINTS_FILE=${CONSTRAINTS_FILE:-"$REPO_ROOT/docs/official-baselines/official-ascend-constraints.stub.json"}
WORKSPACE_ROOT=${VLLM_HUST_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}
CURRENT_RUNTIME_CWD=${CURRENT_RUNTIME_CWD:-"/tmp"}
CURRENT_VLLM_HUST_REPO=${CURRENT_VLLM_HUST_REPO:-"$WORKSPACE_ROOT/vllm-hust"}
CURRENT_VLLM_ASCEND_HUST_REPO=${CURRENT_VLLM_ASCEND_HUST_REPO:-"$WORKSPACE_ROOT/vllm-ascend-hust"}
CURRENT_RUNTIME_PYTHONPATH=${CURRENT_RUNTIME_PYTHONPATH:-}
CURRENT_ENV_PREFIX=${CURRENT_ENV_PREFIX:-"/root/miniconda3/envs/vllm-hust-dev"}
CURRENT_RUNTIME_PYTHON=${CURRENT_RUNTIME_PYTHON:-"$CURRENT_ENV_PREFIX/bin/python"}
CURRENT_VLLM_CACHE_ROOT=${CURRENT_VLLM_CACHE_ROOT:-"$REPO_ROOT/.cache/current-ascend-same-spec"}
CURRENT_MODEL_PATH=${CURRENT_MODEL_PATH:-}
CURRENT_SERVER_HOST=${CURRENT_SERVER_HOST:-}
CURRENT_SERVER_PORT=${CURRENT_SERVER_PORT:-"8001"}
CURRENT_CLIENT_HOST=${CURRENT_CLIENT_HOST:-}
CURRENT_CLIENT_PORT=${CURRENT_CLIENT_PORT:-$CURRENT_SERVER_PORT}
CURRENT_ENGINE=${CURRENT_ENGINE:-"vllm-hust"}
CURRENT_ENGINE_VERSION=${CURRENT_ENGINE_VERSION:-}
CURRENT_SUBMITTER=${CURRENT_SUBMITTER:-"same-spec-current"}
CURRENT_BASELINE_ENGINE=${CURRENT_BASELINE_ENGINE:-"vllm"}
CURRENT_DATA_SOURCE=${CURRENT_DATA_SOURCE:-"vllm-hust-benchmark"}
CURRENT_GITHUB_REPOSITORY=${CURRENT_GITHUB_REPOSITORY:-"vLLM-HUST/vllm-hust"}
CURRENT_GITHUB_REF=${CURRENT_GITHUB_REF:-$(git -C "$CURRENT_VLLM_HUST_REPO" branch --show-current 2>/dev/null || echo main)}
CURRENT_GIT_COMMIT=${CURRENT_GIT_COMMIT:-$(git -C "$CURRENT_VLLM_HUST_REPO" rev-parse HEAD 2>/dev/null || true)}
CURRENT_PLUGIN_ENGINE=${CURRENT_PLUGIN_ENGINE:-"vllm-ascend-hust"}
CURRENT_PLUGIN_GITHUB_REPOSITORY=${CURRENT_PLUGIN_GITHUB_REPOSITORY:-"vLLM-HUST/vllm-ascend-hust"}
CURRENT_PLUGIN_GITHUB_REF=${CURRENT_PLUGIN_GITHUB_REF:-$(git -C "$CURRENT_VLLM_ASCEND_HUST_REPO" branch --show-current 2>/dev/null || echo main)}
CURRENT_PLUGIN_GIT_COMMIT=${CURRENT_PLUGIN_GIT_COMMIT:-$(git -C "$CURRENT_VLLM_ASCEND_HUST_REPO" rev-parse HEAD 2>/dev/null || true)}
ASCEND_TOOLKIT_SET_ENV=${ASCEND_TOOLKIT_SET_ENV:-"/usr/local/Ascend/ascend-toolkit/set_env.sh"}
ASCEND_ATB_SET_ENV=${ASCEND_ATB_SET_ENV:-"/usr/local/Ascend/nnal/atb/set_env.sh"}
ASCEND_ATB_CXX_ABI=${ASCEND_ATB_CXX_ABI:-"1"}
RESULT_DIR=${RESULT_DIR:-"$REPO_ROOT/.benchmarks/current-ascend-same-spec"}
RUN_ID=${RUN_ID:-"current-ascend-same-spec-$(date -u +%Y%m%dT%H%M%SZ)"}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-600}
READY_STATUS_INTERVAL_SECONDS=${READY_STATUS_INTERVAL_SECONDS:-30}
CLIENT_READY_CHECK_TIMEOUT_SECONDS=${CLIENT_READY_CHECK_TIMEOUT_SECONDS:-$READY_TIMEOUT_SECONDS}
SERVER_PID=""
RUNNER_LOCK_FD=""

CURRENT_RUNTIME_SOURCE_PYTHONPATH="$CURRENT_VLLM_ASCEND_HUST_REPO:$CURRENT_VLLM_HUST_REPO"
if [[ -n "$CURRENT_RUNTIME_PYTHONPATH" ]]; then
  CURRENT_RUNTIME_PYTHONPATH="$CURRENT_RUNTIME_SOURCE_PYTHONPATH:$CURRENT_RUNTIME_PYTHONPATH"
else
  CURRENT_RUNTIME_PYTHONPATH="$CURRENT_RUNTIME_SOURCE_PYTHONPATH"
fi

if [[ ! -x "$CURRENT_RUNTIME_PYTHON" ]]; then
  echo "CURRENT_RUNTIME_PYTHON is not executable: $CURRENT_RUNTIME_PYTHON" >&2
  exit 2
fi

if [[ ! -f "$SPEC_FILE" ]]; then
  echo "Spec file not found: $SPEC_FILE" >&2
  exit 2
fi

if [[ ! -f "$CONSTRAINTS_FILE" ]]; then
  echo "Constraints stub not found: $CONSTRAINTS_FILE" >&2
  exit 2
fi

SCRIPT_BASENAME=$(basename "$0")

is_valid_engine_version() {
  local value=${1//$'\r'/}
  value=$(printf '%s' "$value" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')
  [[ -n "$value" ]] || return 1
  [[ "$value" =~ [0-9] ]] || return 1
  [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]
}

resolve_current_engine_version_from_git() {
  local described=""

  if [[ -n "$CURRENT_GIT_COMMIT" ]]; then
    described=$(git -C "$CURRENT_VLLM_HUST_REPO" describe --tags --always "$CURRENT_GIT_COMMIT" 2>/dev/null || true)
    if [[ -n "$described" ]]; then
      printf '%s' "$described"
      return 0
    fi
  fi

  described=$(git -C "$CURRENT_VLLM_HUST_REPO" describe --tags --always HEAD 2>/dev/null || true)
  if [[ -n "$described" ]]; then
    printf '%s' "$described"
    return 0
  fi

  git -C "$CURRENT_VLLM_HUST_REPO" rev-parse --short HEAD 2>/dev/null || true
}

detect_current_engine_version() {
  local raw_output=""
  local detected=""
  local fallback=""

  raw_output=$(run_in_current_runtime "$CURRENT_RUNTIME_PYTHONPATH" "$CURRENT_RUNTIME_PYTHON" - <<'PY'
from importlib import metadata

version = None
try:
    import vllm
    version = getattr(vllm, '__version__', None)
except Exception:
    version = None

if not version:
    try:
        version = metadata.version('vllm')
    except Exception:
        version = None

print(f"__VLLM_HUST_ENGINE_VERSION__={version or ''}")
PY
)

  detected=$(printf '%s\n' "$raw_output" | sed -n 's/^__VLLM_HUST_ENGINE_VERSION__=//p' | tail -n 1)
  detected=$(printf '%s' "$detected" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')

  if is_valid_engine_version "$detected"; then
    printf '%s' "$detected"
    return 0
  fi

  fallback=$(resolve_current_engine_version_from_git)
  if is_valid_engine_version "$fallback"; then
    printf '%s' "$fallback"
    return 0
  fi

  printf '%s' "unknown"
}

list_port_listener_pids() {
  local port=$1

  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -E ":${port}[[:space:]]" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true
    return 0
  fi

  if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null | sort -u || true
    return 0
  fi

  if command -v fuser >/dev/null 2>&1; then
    fuser "${port}/tcp" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort -u || true
  fi
}

list_child_pids() {
  local parent_pid=$1
  ps -eo pid=,ppid= | awk -v target="$parent_pid" '$2 == target {print $1}'
}

collect_process_tree_pids() {
  local root_pid=$1
  local child_pid

  if ! kill -0 "$root_pid" 2>/dev/null; then
    return 0
  fi

  echo "$root_pid"
  while IFS= read -r child_pid; do
    [[ -z "$child_pid" ]] && continue
    collect_process_tree_pids "$child_pid"
  done < <(list_child_pids "$root_pid")
}

process_args() {
  local pid=$1
  ps -p "$pid" -o args= 2>/dev/null || true
}

is_managed_wrapper_process() {
  local pid=$1
  local args
  args=$(process_args "$pid")
  [[ -n "$args" ]] && [[ "$args" == *"$SCRIPT_BASENAME"* ]]
}

is_server_process_for_port() {
  local pid=$1
  local port=$2
  local args
  args=$(process_args "$pid")
  [[ -n "$args" ]] && [[ "$args" == *"vllm.entrypoints.openai.api_server"* ]] && [[ "$args" == *"--port $port"* ]]
}

terminate_pid_tree() {
  local pid=$1
  local description=$2
  local tree_pids
  local tree_list
  local still_running
  local tree_pid
  local attempt

  tree_pids=$(collect_process_tree_pids "$pid" | sort -u)
  [[ -z "$tree_pids" ]] && return 0
  tree_list=$(echo "$tree_pids" | tr '\n' ' ')

  echo "[same-spec-current] stopping ${description}: ${tree_list}"
  kill $tree_list 2>/dev/null || true

  for attempt in $(seq 1 10); do
    still_running=0
    while IFS= read -r tree_pid; do
      [[ -z "$tree_pid" ]] && continue
      if kill -0 "$tree_pid" 2>/dev/null; then
        still_running=1
        break
      fi
    done <<< "$tree_pids"

    if [[ "$still_running" == "0" ]]; then
      return 0
    fi
    sleep 1
  done

  kill -9 $tree_list 2>/dev/null || true
}

acquire_runner_lock() {
  mkdir -p "$RUNNER_STATE_DIR"
  exec {RUNNER_LOCK_FD}>"$RUNNER_LOCK_FILE"
  if ! flock -n "$RUNNER_LOCK_FD"; then
    echo "Another current same-spec benchmark run is already active for $RESULT_DIR" >&2
    exit 1
  fi
}

persist_managed_server_state() {
  printf '%s\n' "$SERVER_PORT" > "$MANAGED_SERVER_PORT_FILE"
  if [[ -n "$SERVER_PID" ]]; then
    printf '%s\n' "$SERVER_PID" > "$MANAGED_SERVER_WRAPPER_PID_FILE"
  fi

  list_port_listener_pids "$SERVER_PORT" > "$MANAGED_SERVER_LISTENER_PIDS_FILE" || true
}

clear_managed_server_state() {
  rm -f "$MANAGED_SERVER_PORT_FILE" "$MANAGED_SERVER_WRAPPER_PID_FILE" "$MANAGED_SERVER_LISTENER_PIDS_FILE"
}

cleanup_managed_server() {
  local managed_port=""
  local candidate_pids=""
  local pid

  if [[ -f "$MANAGED_SERVER_PORT_FILE" ]]; then
    managed_port=$(tr -d '[:space:]' < "$MANAGED_SERVER_PORT_FILE")
  fi

  if [[ -f "$MANAGED_SERVER_WRAPPER_PID_FILE" ]]; then
    candidate_pids+=$(cat "$MANAGED_SERVER_WRAPPER_PID_FILE")$'\n'
  fi
  if [[ -f "$MANAGED_SERVER_LISTENER_PIDS_FILE" ]]; then
    candidate_pids+=$(cat "$MANAGED_SERVER_LISTENER_PIDS_FILE")$'\n'
  fi
  if [[ -n "$managed_port" ]]; then
    candidate_pids+=$(list_port_listener_pids "$managed_port")$'\n'
  fi

  candidate_pids=$(printf '%s' "$candidate_pids" | sed '/^$/d' | sort -u)

  if [[ -n "$candidate_pids" ]]; then
    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      if [[ -n "$managed_port" ]] && is_server_process_for_port "$pid" "$managed_port"; then
        terminate_pid_tree "$pid" "managed current same-spec server"
      elif is_managed_wrapper_process "$pid"; then
        terminate_pid_tree "$pid" "managed current same-spec wrapper"
      fi
    done <<< "$candidate_pids"
  fi

  clear_managed_server_state

  if [[ -n "$managed_port" ]]; then
    local remaining_pids
    remaining_pids=$(list_port_listener_pids "$managed_port")
    if [[ -n "$remaining_pids" ]]; then
      echo "Managed current same-spec port ${managed_port} is still occupied after cleanup: $remaining_pids" >&2
      return 1
    fi
  fi
}

run_in_current_runtime() {
  local pythonpath_prefix=$1
  shift
  (
    cd "$CURRENT_RUNTIME_CWD"
    export ZSH_VERSION=""
    if [[ -f "$ASCEND_TOOLKIT_SET_ENV" ]]; then
      # shellcheck disable=SC1090
      source "$ASCEND_TOOLKIT_SET_ENV"
    fi
    if [[ -f "$ASCEND_ATB_SET_ENV" ]]; then
      set +u
      # shellcheck disable=SC1090
      source "$ASCEND_ATB_SET_ENV" --cxx_abi="$ASCEND_ATB_CXX_ABI"
      set -u
    fi
    export VLLM_CACHE_ROOT="$CURRENT_VLLM_CACHE_ROOT"
    PYTHONPATH="$pythonpath_prefix${PYTHONPATH:+:$PYTHONPATH}" \
      "$@"
  )
}

run_server_command() {
  (
    cd "$CURRENT_RUNTIME_CWD"
    export ZSH_VERSION=""
    if [[ -f "$ASCEND_TOOLKIT_SET_ENV" ]]; then
      # shellcheck disable=SC1090
      source "$ASCEND_TOOLKIT_SET_ENV"
    fi
    if [[ -f "$ASCEND_ATB_SET_ENV" ]]; then
      set +u
      # shellcheck disable=SC1090
      source "$ASCEND_ATB_SET_ENV" --cxx_abi="$ASCEND_ATB_CXX_ABI"
      set -u
    fi
    export VLLM_CACHE_ROOT="$CURRENT_VLLM_CACHE_ROOT"
    PYTHONUNBUFFERED=1 \
      PYTHONPATH="$CURRENT_RUNTIME_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
      "$CURRENT_RUNTIME_PYTHON" -u -m vllm.entrypoints.openai.api_server $SERVER_ARGS
  )
}

run_client_command() {
  run_in_current_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
    "$CURRENT_RUNTIME_PYTHON" -m vllm.entrypoints.cli.main bench serve \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$(basename "$RAW_RESULT_FILE")" \
    $CLIENT_ARGS
}

json2args() {
  local json_string=$1
  echo "$json_string" | jq -r '
    to_entries |
    map(if (.value | tostring) == "" then "--" + (.key | gsub("_"; "-")) else "--" + (.key | gsub("_"; "-")) + " " + (.value | tostring) end) |
    join(" ")
  '
}

resolve_runtime_model() {
  if [[ -n "$CURRENT_MODEL_PATH" ]]; then
    echo "$CURRENT_MODEL_PATH"
    return 0
  fi

  run_in_current_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
    env MODEL_ID="$MODEL" \
    "$CURRENT_RUNTIME_PYTHON" -c "import os; from huggingface_hub import snapshot_download; print(snapshot_download(os.environ['MODEL_ID'], local_files_only=True))" \
    2>/dev/null || return 1
}

local_runtime_model_has_required_artifacts() {
  local runtime_model_candidate=$1

  run_in_current_runtime "$REPO_ROOT/src${CURRENT_RUNTIME_PYTHONPATH:+:$CURRENT_RUNTIME_PYTHONPATH}" \
    env RUNTIME_MODEL_CANDIDATE="$runtime_model_candidate" \
    "$CURRENT_RUNTIME_PYTHON" - <<'PY' >/dev/null
import os

from vllm_hust_benchmark.same_spec import runtime_model_path_has_required_artifacts

raise SystemExit(
    0 if runtime_model_path_has_required_artifacts(os.environ["RUNTIME_MODEL_CANDIDATE"]) else 1
)
PY
}

resolve_same_spec() {
  local resolve_args=(
    "$CURRENT_RUNTIME_PYTHON" -m vllm_hust_benchmark.same_spec resolve
    --spec-file "$SPEC_FILE"
    --output-file "$SAME_SPEC_FILE"
    --runtime-model "$RUNTIME_MODEL"
    --server-port "$CURRENT_SERVER_PORT"
    --client-port "$CURRENT_CLIENT_PORT"
  )

  if [[ -n "$CURRENT_SERVER_HOST" ]]; then
    resolve_args+=(--server-host "$CURRENT_SERVER_HOST")
  fi
  if [[ -n "$CURRENT_CLIENT_HOST" ]]; then
    resolve_args+=(--client-host "$CURRENT_CLIENT_HOST")
  fi

  run_in_current_runtime "$REPO_ROOT/src${CURRENT_RUNTIME_PYTHONPATH:+:$CURRENT_RUNTIME_PYTHONPATH}" "${resolve_args[@]}"
}

port_has_listener() {
  local port=$1

  if command -v ss >/dev/null 2>&1; then
    ss -ltnH "( sport = :$port )" 2>/dev/null | grep -q .
    return $?
  fi

  if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:"$port" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi

  if command -v fuser >/dev/null 2>&1; then
    fuser "${port}/tcp" >/dev/null 2>&1
    return $?
  fi

  return 1
}

assert_target_port_available() {
  local label=$1
  local host=$2
  local port=$3

  if curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; then
    echo "${label} target ${host}:${port} is already serving /health; refusing to reuse a stale service." >&2
    return 1
  fi

  if port_has_listener "$port"; then
    echo "${label} target port ${port} already has a listening process; choose another port or stop the stale service." >&2
    return 1
  fi
}

probe_server_ready() {
  local host=$1
  local port=$2
  local ready_path
  local ready_paths=(
    "/health"
    "/v1/models"
  )

  for ready_path in "${ready_paths[@]}"; do
    if curl -fsS "http://${host}:${port}${ready_path}" >/dev/null 2>&1; then
      return 0
    fi
  done

  return 1
}

wait_for_server() {
  local host=$1
  local port=$2
  local waited=0
  local timeout_sec=${READY_TIMEOUT_SECONDS}
  local status_interval_sec=${READY_STATUS_INTERVAL_SECONDS}
  local next_status_at=0

  if (( status_interval_sec <= 0 )); then
    status_interval_sec=30
  fi

  while (( waited < timeout_sec )); do
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Current same-spec server exited before becoming ready" >&2
      if [[ -n "${SERVER_STDOUT_LOG:-}" && -f "$SERVER_STDOUT_LOG" ]]; then
        tail -n 40 "$SERVER_STDOUT_LOG" >&2
      fi
      return 1
    fi

    if probe_server_ready "$host" "$port"; then
      if (( waited > 0 )); then
        echo "[same-spec-current] current same-spec server became ready after ${waited}s"
      fi
      return 0
    fi

    if (( waited >= next_status_at )); then
      echo "[same-spec-current] waiting for current same-spec server at ${host}:${port} (${waited}s/${timeout_sec}s)" >&2
      next_status_at=$((waited + status_interval_sec))
    fi

    sleep 1
    ((waited += 1))
  done

  echo "Timed out waiting for current same-spec server at ${host}:${port}" >&2
  if [[ -n "${SERVER_STDOUT_LOG:-}" && -f "$SERVER_STDOUT_LOG" ]]; then
    tail -n 40 "$SERVER_STDOUT_LOG" >&2
  fi
  return 1
}

kill_server() {
  cleanup_managed_server || true
}

trap kill_server EXIT

mkdir -p "$RESULT_DIR"
mkdir -p "$CURRENT_VLLM_CACHE_ROOT"
RUNNER_STATE_DIR="$RESULT_DIR/.runtime-state"
RUNNER_LOCK_FILE="$RUNNER_STATE_DIR/runner.lock"
MANAGED_SERVER_PORT_FILE="$RUNNER_STATE_DIR/server.port"
MANAGED_SERVER_WRAPPER_PID_FILE="$RUNNER_STATE_DIR/server.wrapper.pid"
MANAGED_SERVER_LISTENER_PIDS_FILE="$RUNNER_STATE_DIR/server.listener.pids"

acquire_runner_lock
cleanup_managed_server

MODEL=$(jq -r '.model' "$SPEC_FILE")
MODEL_PARAMETERS=$(jq -r '.model_parameters' "$SPEC_FILE")
MODEL_PRECISION=$(jq -r '.model_precision' "$SPEC_FILE")
HARDWARE_VENDOR=$(jq -r '.hardware_vendor' "$SPEC_FILE")
HARDWARE_CHIP_MODEL=$(jq -r '.hardware_chip_model' "$SPEC_FILE")
CHIP_COUNT=$(jq -r '.chip_count' "$SPEC_FILE")
NODE_COUNT=$(jq -r '.node_count' "$SPEC_FILE")
SCENARIO=$(jq -r '.scenario' "$SPEC_FILE")
INPUT_LEN=$(jq -r '.client_parameters.input_len' "$SPEC_FILE")
OUTPUT_LEN=$(jq -r '.client_parameters.output_len' "$SPEC_FILE")

if [[ -z "$CURRENT_ENGINE_VERSION" ]]; then
  CURRENT_ENGINE_VERSION=$(detect_current_engine_version)
fi

RUNTIME_MODEL="$MODEL"
if cached_model_path=$(resolve_runtime_model); then
  if [[ -n "$CURRENT_MODEL_PATH" ]] || local_runtime_model_has_required_artifacts "$cached_model_path"; then
    RUNTIME_MODEL="$cached_model_path"
  else
    echo "[same-spec-current] cached local snapshot is missing tokenizer or weight artifacts; falling back to model ID ${MODEL}" >&2
  fi
fi

SAME_SPEC_FILE="$RESULT_DIR/resolved_same_spec.json"
resolve_same_spec

SERVER_HOST=$(jq -r '.resolved_server_parameters.host' "$SAME_SPEC_FILE")
SERVER_PORT=$(jq -r '.resolved_server_parameters.port' "$SAME_SPEC_FILE")
CLIENT_HOST=$(jq -r '.resolved_client_parameters.host' "$SAME_SPEC_FILE")
CLIENT_PORT=$(jq -r '.resolved_client_parameters.port' "$SAME_SPEC_FILE")

assert_target_port_available "Current same-spec benchmark" "$CLIENT_HOST" "$CLIENT_PORT"

SERVER_ARGS=$(json2args "$(jq -c '.resolved_server_parameters | del(.disable_log_requests)' "$SAME_SPEC_FILE")")
CLIENT_ARGS=$(json2args "$({
  jq -c \
    --argjson ready_timeout "$CLIENT_READY_CHECK_TIMEOUT_SECONDS" \
    '.resolved_client_parameters
      | .ready_check_timeout_sec = (
          if (.ready_check_timeout_sec // 0) > 0
          then .ready_check_timeout_sec
          else $ready_timeout
          end
        )' "$SAME_SPEC_FILE"
})")

RAW_RESULT_FILE="$RESULT_DIR/raw_benchmark_result.json"
ARTIFACT_DIR="$RESULT_DIR/submission"
SERVER_STDOUT_LOG="$RESULT_DIR/server.stdout.log"

echo "[same-spec-current] neutral cwd: $CURRENT_RUNTIME_CWD"
echo "[same-spec-current] vllm cache root: $CURRENT_VLLM_CACHE_ROOT"
echo "[same-spec-current] runtime model source: $RUNTIME_MODEL"
echo "[same-spec-current] resolved spec file: $SAME_SPEC_FILE"
echo "[same-spec-current] benchmark endpoint: ${CLIENT_HOST}:${CLIENT_PORT}"
run_server_command >"$SERVER_STDOUT_LOG" 2>&1 &
SERVER_PID=$!
persist_managed_server_state

wait_for_server "$CLIENT_HOST" "$CLIENT_PORT"
persist_managed_server_state
run_client_command

run_in_current_runtime "$REPO_ROOT/src${CURRENT_RUNTIME_PYTHONPATH:+:$CURRENT_RUNTIME_PYTHONPATH}" \
"$CURRENT_RUNTIME_PYTHON" -m vllm_hust_benchmark.cli export-leaderboard-artifact \
  "$SCENARIO" \
  --benchmark-result-file "$RAW_RESULT_FILE" \
  --constraints-file "$CONSTRAINTS_FILE" \
  --same-spec-file "$SAME_SPEC_FILE" \
  --output-dir "$ARTIFACT_DIR" \
  --run-id "$RUN_ID" \
  --engine "$CURRENT_ENGINE" \
  --engine-version "$CURRENT_ENGINE_VERSION" \
  --model-name "$MODEL" \
  --model-parameters "$MODEL_PARAMETERS" \
  --model-precision "$MODEL_PRECISION" \
  --hardware-vendor "$HARDWARE_VENDOR" \
  --hardware-chip-model "$HARDWARE_CHIP_MODEL" \
  --chip-count "$CHIP_COUNT" \
  --node-count "$NODE_COUNT" \
  --submitter "$CURRENT_SUBMITTER" \
  --baseline-engine "$CURRENT_BASELINE_ENGINE" \
  --data-source "$CURRENT_DATA_SOURCE" \
  --input-length "$INPUT_LEN" \
  --output-length "$OUTPUT_LEN" \
  --git-commit "$CURRENT_GIT_COMMIT" \
  --github-repository "$CURRENT_GITHUB_REPOSITORY" \
  --github-ref "$CURRENT_GITHUB_REF" \
  --runtime-python "$CURRENT_RUNTIME_PYTHON" \
  --engine-source-repository "$CURRENT_GITHUB_REPOSITORY" \
  --engine-source-ref "$CURRENT_GITHUB_REF" \
  --engine-source-commit "$CURRENT_GIT_COMMIT" \
  --plugin-source-engine "$CURRENT_PLUGIN_ENGINE" \
  --plugin-source-repository "$CURRENT_PLUGIN_GITHUB_REPOSITORY" \
  --plugin-source-ref "$CURRENT_PLUGIN_GITHUB_REF" \
  --plugin-source-commit "$CURRENT_PLUGIN_GIT_COMMIT"

echo "[same-spec-current] exported leaderboard artifact to $ARTIFACT_DIR"