#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PREPARE_SCRIPT=${PREPARE_SCRIPT:-"$REPO_ROOT/scripts/prepare-official-ascend-baseline-env.sh"}
SPEC_FILE=${1:-"$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0110-random-online-qwen25-14b-910b3.json"}
CONSTRAINTS_FILE=${CONSTRAINTS_FILE:-"$REPO_ROOT/docs/official-baselines/official-ascend-constraints.stub.json"}
WORKSPACE_ROOT=${VLLM_HUST_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}
OFFICIAL_VLLM_REPO=${OFFICIAL_VLLM_REPO:-"$WORKSPACE_ROOT/reference-repos/vllm"}
OFFICIAL_VLLM_ASCEND_REPO=${OFFICIAL_VLLM_ASCEND_REPO:-"$WORKSPACE_ROOT/reference-repos/vllm-ascend"}
OFFICIAL_VLLM_WORKTREE=${OFFICIAL_VLLM_WORKTREE:-"/tmp/vllm-v0110"}
OFFICIAL_VLLM_ASCEND_WORKTREE=${OFFICIAL_VLLM_ASCEND_WORKTREE:-"/tmp/vllm-ascend-v0110"}
OFFICIAL_RUNTIME_CWD=${OFFICIAL_RUNTIME_CWD:-"/tmp"}
OFFICIAL_VLLM_CACHE_ROOT=${OFFICIAL_VLLM_CACHE_ROOT:-"$REPO_ROOT/.cache/official-ascend-goal-baseline"}
OFFICIAL_MODEL_PATH=${OFFICIAL_MODEL_PATH:-}
OFFICIAL_SERVER_HOST=${OFFICIAL_SERVER_HOST:-}
OFFICIAL_SERVER_PORT=${OFFICIAL_SERVER_PORT:-"8000"}
OFFICIAL_CLIENT_HOST=${OFFICIAL_CLIENT_HOST:-}
OFFICIAL_CLIENT_PORT=${OFFICIAL_CLIENT_PORT:-$OFFICIAL_SERVER_PORT}
OFFICIAL_CORE_VERSION=${OFFICIAL_CORE_VERSION:-}
OFFICIAL_BACKEND_VERSION=${OFFICIAL_BACKEND_VERSION:-}
ASCEND_TOOLKIT_SET_ENV=${ASCEND_TOOLKIT_SET_ENV:-"/usr/local/Ascend/ascend-toolkit/set_env.sh"}
ASCEND_ATB_SET_ENV=${ASCEND_ATB_SET_ENV:-"/usr/local/Ascend/nnal/atb/set_env.sh"}
ASCEND_ATB_CXX_ABI=${ASCEND_ATB_CXX_ABI:-"1"}
GOAL_BASELINE_ENV_PREFIX=${GOAL_BASELINE_ENV_PREFIX:-}
RESULT_DIR=${RESULT_DIR:-"$REPO_ROOT/.benchmarks/official-ascend-goal-baseline"}
RUN_ID=${RUN_ID:-"official-ascend-jan-2026-$(date -u +%Y%m%dT%H%M%SZ)"}
SERVER_PID=""
RUNNER_LOCK_FD=""

if [[ -z "$GOAL_BASELINE_ENV_PREFIX" ]]; then
  echo "GOAL_BASELINE_ENV_PREFIX is required" >&2
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

if [[ ! -d "$REPO_ROOT/src" ]]; then
  echo "Benchmark repo not found: $REPO_ROOT" >&2
  exit 2
fi

if [[ ! -f "$PREPARE_SCRIPT" ]]; then
  echo "Prepare script not found: $PREPARE_SCRIPT" >&2
  exit 2
fi

SCRIPT_BASENAME=$(basename "$0")

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

  echo "[goal-baseline] stopping ${description}: ${tree_list}"
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
    echo "Another official baseline run is already active for $RESULT_DIR" >&2
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
        terminate_pid_tree "$pid" "managed official baseline server"
      elif is_managed_wrapper_process "$pid"; then
        terminate_pid_tree "$pid" "managed official baseline wrapper"
      fi
    done <<< "$candidate_pids"
  fi

  clear_managed_server_state

  if [[ -n "$managed_port" ]]; then
    local remaining_pids
    remaining_pids=$(list_port_listener_pids "$managed_port")
    if [[ -n "$remaining_pids" ]]; then
      echo "Managed official baseline port ${managed_port} is still occupied after cleanup: $remaining_pids" >&2
      return 1
    fi
  fi
}

run_in_official_runtime() {
  local pythonpath_prefix=$1
  shift
  (
    cd "$OFFICIAL_RUNTIME_CWD"
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
    export VLLM_CACHE_ROOT="$OFFICIAL_VLLM_CACHE_ROOT"
    PYTHONPATH="$pythonpath_prefix${PYTHONPATH:+:$PYTHONPATH}" \
      conda run -p "$GOAL_BASELINE_ENV_PREFIX" "$@"
  )
}

run_server_command() {
  (
    cd "$OFFICIAL_RUNTIME_CWD"
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
    export VLLM_CACHE_ROOT="$OFFICIAL_VLLM_CACHE_ROOT"
    PYTHONUNBUFFERED=1 \
      PYTHONPATH="$OFFICIAL_RUNTIME_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
      conda run --no-capture-output -p "$GOAL_BASELINE_ENV_PREFIX" \
      python -u -m vllm.entrypoints.openai.api_server $SERVER_ARGS
  )
}

run_client_command() {
  run_in_official_runtime "$OFFICIAL_RUNTIME_PYTHONPATH" \
    python -m vllm.entrypoints.cli.main bench serve \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$(basename "$RAW_RESULT_FILE")" \
    $CLIENT_ARGS
}

resolve_same_spec() {
  local resolve_args=(
    python -m vllm_hust_benchmark.same_spec resolve
    --spec-file "$SPEC_FILE"
    --output-file "$SAME_SPEC_FILE"
    --runtime-model "$RUNTIME_MODEL"
    --server-port "$OFFICIAL_SERVER_PORT"
    --client-port "$OFFICIAL_CLIENT_PORT"
  )

  if [[ -n "$OFFICIAL_SERVER_HOST" ]]; then
    resolve_args+=(--server-host "$OFFICIAL_SERVER_HOST")
  fi
  if [[ -n "$OFFICIAL_CLIENT_HOST" ]]; then
    resolve_args+=(--client-host "$OFFICIAL_CLIENT_HOST")
  fi

  run_in_official_runtime "$REPO_ROOT/src:$OFFICIAL_RUNTIME_PYTHONPATH" "${resolve_args[@]}"
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

ensure_worktree() {
  local source_repo=$1
  local target_dir=$2
  local ref_name=$3
  if [[ -f "$target_dir/pyproject.toml" ]]; then
    return 0
  fi
  git -C "$source_repo" worktree add --detach "$target_dir" "$ref_name"
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
  if [[ -n "$OFFICIAL_MODEL_PATH" ]]; then
    echo "$OFFICIAL_MODEL_PATH"
    return 0
  fi

  run_in_official_runtime "$OFFICIAL_RUNTIME_PYTHONPATH" \
    env MODEL_ID="$MODEL" \
    python -c "import os; from huggingface_hub import snapshot_download; print(snapshot_download(os.environ['MODEL_ID'], local_files_only=True))" \
    2>/dev/null || return 1
}

local_runtime_model_has_required_artifacts() {
  local runtime_model_candidate=$1

  run_in_official_runtime "$REPO_ROOT/src${OFFICIAL_RUNTIME_PYTHONPATH:+:$OFFICIAL_RUNTIME_PYTHONPATH}" \
    env RUNTIME_MODEL_CANDIDATE="$runtime_model_candidate" \
    python - <<'PY' >/dev/null
import os

from vllm_hust_benchmark.same_spec import runtime_model_path_has_required_artifacts

raise SystemExit(
    0 if runtime_model_path_has_required_artifacts(os.environ["RUNTIME_MODEL_CANDIDATE"]) else 1
)
PY

is_valid_engine_version() {
  local version=${1:-}
  [[ -n "$version" ]] && [[ "$version" != "unknown" ]] && [[ "$version" != "not-installed" ]] && [[ "$version" != "N/A" ]]
}

detect_official_core_version() {
  local raw_output=""
  local detected=""
  local fallback=""

  raw_output=$(run_in_official_runtime "$OFFICIAL_RUNTIME_PYTHONPATH" python - <<'PY'
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

print(f"__VLLM_HUST_CORE_VERSION__={version or ''}")
PY
)

  detected=$(printf '%s\n' "$raw_output" | sed -n 's/^__VLLM_HUST_CORE_VERSION__=//p' | tail -n 1)
  detected=$(printf '%s' "$detected" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')

  if is_valid_engine_version "$detected"; then
    printf '%s' "$detected"
    return 0
  fi

  fallback=$(git -C "$OFFICIAL_VLLM_WORKTREE" describe --tags --always HEAD 2>/dev/null || true)
  if is_valid_engine_version "$fallback"; then
    printf '%s' "$fallback"
    return 0
  fi

  printf '%s' "unknown"
}

detect_official_backend_version() {
  local raw_output=""
  local detected=""
  local fallback=""

  raw_output=$(run_in_official_runtime "$OFFICIAL_RUNTIME_PYTHONPATH" python - <<'PY'
from importlib import metadata

version = None
try:
    import vllm_ascend
    version = getattr(vllm_ascend, '__version__', None)
except Exception:
    version = None

if not version:
    for dist_name in ('vllm-ascend', 'vllm_ascend'):
        try:
            version = metadata.version(dist_name)
            break
        except Exception:
            continue

print(f"__VLLM_HUST_BACKEND_VERSION__={version or ''}")
PY
)

  detected=$(printf '%s\n' "$raw_output" | sed -n 's/^__VLLM_HUST_BACKEND_VERSION__=//p' | tail -n 1)
  detected=$(printf '%s' "$detected" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')

  if is_valid_engine_version "$detected"; then
    printf '%s' "$detected"
    return 0
  fi

  fallback=$(git -C "$OFFICIAL_VLLM_ASCEND_WORKTREE" describe --tags --always HEAD 2>/dev/null || true)
  if is_valid_engine_version "$fallback"; then
    printf '%s' "$fallback"
    return 0
  fi

  printf '%s' "unknown"
}
}

wait_for_server() {
  local host=$1
  local port=$2
  local waited=0
  local timeout_sec=${READY_TIMEOUT_SECONDS:-300}

  while (( waited < timeout_sec )); do
    if curl -fsS "http://${host}:${port}/health" >/dev/null; then
      return 0
    fi
    sleep 1
    ((waited += 1))
  done

  echo "Timed out waiting for official baseline server at ${host}:${port}" >&2
  return 1
}

kill_server() {
  cleanup_managed_server || true
}

trap kill_server EXIT

ensure_worktree "$OFFICIAL_VLLM_REPO" "$OFFICIAL_VLLM_WORKTREE" "v0.11.0"
ensure_worktree "$OFFICIAL_VLLM_ASCEND_REPO" "$OFFICIAL_VLLM_ASCEND_WORKTREE" "v0.11.0"

OFFICIAL_RUNTIME_PYTHONPATH="$OFFICIAL_VLLM_ASCEND_WORKTREE:$OFFICIAL_VLLM_WORKTREE"

mkdir -p "$RESULT_DIR"
mkdir -p "$OFFICIAL_VLLM_CACHE_ROOT"
RUNNER_STATE_DIR="$RESULT_DIR/.runtime-state"
RUNNER_LOCK_FILE="$RUNNER_STATE_DIR/runner.lock"
MANAGED_SERVER_PORT_FILE="$RUNNER_STATE_DIR/server.port"
MANAGED_SERVER_WRAPPER_PID_FILE="$RUNNER_STATE_DIR/server.wrapper.pid"
MANAGED_SERVER_LISTENER_PIDS_FILE="$RUNNER_STATE_DIR/server.listener.pids"

acquire_runner_lock
cleanup_managed_server

SCENARIO=$(jq -r '.scenario' "$SPEC_FILE")
MODEL=$(jq -r '.model' "$SPEC_FILE")
MODEL_PARAMETERS=$(jq -r '.model_parameters' "$SPEC_FILE")
MODEL_PRECISION=$(jq -r '.model_precision' "$SPEC_FILE")
HARDWARE_VENDOR=$(jq -r '.hardware_vendor' "$SPEC_FILE")
HARDWARE_CHIP_MODEL=$(jq -r '.hardware_chip_model' "$SPEC_FILE")
CHIP_COUNT=$(jq -r '.chip_count' "$SPEC_FILE")
NODE_COUNT=$(jq -r '.node_count' "$SPEC_FILE")
ENGINE=$(jq -r '.export.engine' "$SPEC_FILE")
ENGINE_VERSION=$(jq -r '.export.engine_version' "$SPEC_FILE")
SUBMITTER=$(jq -r '.export.submitter' "$SPEC_FILE")
BASELINE_ENGINE=$(jq -r '.export.baseline_engine' "$SPEC_FILE")
GITHUB_REPOSITORY=$(jq -r '.export.github_repository' "$SPEC_FILE")
GITHUB_REF=$(jq -r '.export.github_ref' "$SPEC_FILE")
GIT_COMMIT=$(jq -r '.export.git_commit' "$SPEC_FILE")
DATA_SOURCE=$(jq -r '.export.data_source' "$SPEC_FILE")
INPUT_LEN=$(jq -r '.client_parameters.input_len' "$SPEC_FILE")
OUTPUT_LEN=$(jq -r '.client_parameters.output_len' "$SPEC_FILE")

if [[ -z "$OFFICIAL_CORE_VERSION" ]]; then
  OFFICIAL_CORE_VERSION=$(detect_official_core_version)
fi
if ! is_valid_engine_version "$OFFICIAL_CORE_VERSION"; then
  OFFICIAL_CORE_VERSION="$ENGINE_VERSION"
fi

if [[ -z "$OFFICIAL_BACKEND_VERSION" ]]; then
  OFFICIAL_BACKEND_VERSION=$(detect_official_backend_version)
fi
if ! is_valid_engine_version "$OFFICIAL_BACKEND_VERSION"; then
  OFFICIAL_BACKEND_VERSION="$ENGINE_VERSION"
fi

RUNTIME_MODEL="$MODEL"
if cached_model_path=$(resolve_runtime_model); then
  if [[ -n "$OFFICIAL_MODEL_PATH" ]] || local_runtime_model_has_required_artifacts "$cached_model_path"; then
    RUNTIME_MODEL="$cached_model_path"
  else
    echo "[goal-baseline] cached local snapshot is missing tokenizer or weight artifacts; falling back to model ID ${MODEL}" >&2
  fi
fi

SAME_SPEC_FILE="$RESULT_DIR/resolved_same_spec.json"
resolve_same_spec

SERVER_HOST=$(jq -r '.resolved_server_parameters.host' "$SAME_SPEC_FILE")
SERVER_PORT=$(jq -r '.resolved_server_parameters.port' "$SAME_SPEC_FILE")
CLIENT_HOST=$(jq -r '.resolved_client_parameters.host' "$SAME_SPEC_FILE")
CLIENT_PORT=$(jq -r '.resolved_client_parameters.port' "$SAME_SPEC_FILE")

BENCHMARK_SERVER_PORT="$SERVER_PORT" \
PREPARE_BENCHMARK_ADMISSION_ONLY=1 \
ENV_PREFIX="$GOAL_BASELINE_ENV_PREFIX" \
VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
bash "$PREPARE_SCRIPT"

assert_target_port_available "Official baseline" "$CLIENT_HOST" "$CLIENT_PORT"

SERVER_ARGS=$(json2args "$(jq -c '.resolved_server_parameters' "$SAME_SPEC_FILE")")
CLIENT_ARGS=$(json2args "$(jq -c '.resolved_client_parameters' "$SAME_SPEC_FILE")")

RAW_RESULT_FILE="$RESULT_DIR/raw_benchmark_result.json"
ARTIFACT_DIR="$RESULT_DIR/submission"

SERVER_COMMAND="PYTHONUNBUFFERED=1 PYTHONPATH=$OFFICIAL_RUNTIME_PYTHONPATH\${PYTHONPATH:+:\$PYTHONPATH} conda run --no-capture-output -p $GOAL_BASELINE_ENV_PREFIX python -u -m vllm.entrypoints.openai.api_server $SERVER_ARGS"
CLIENT_COMMAND="PYTHONPATH=$OFFICIAL_RUNTIME_PYTHONPATH\${PYTHONPATH:+:\$PYTHONPATH} conda run -p $GOAL_BASELINE_ENV_PREFIX python -m vllm.entrypoints.cli.main bench serve --save-result --result-dir $RESULT_DIR --result-filename $(basename "$RAW_RESULT_FILE") $CLIENT_ARGS"

echo "[goal-baseline] using worktrees: $OFFICIAL_VLLM_WORKTREE and $OFFICIAL_VLLM_ASCEND_WORKTREE"
echo "[goal-baseline] neutral cwd: $OFFICIAL_RUNTIME_CWD"
echo "[goal-baseline] vllm cache root: $OFFICIAL_VLLM_CACHE_ROOT"
echo "[goal-baseline] export model id: $MODEL"
echo "[goal-baseline] runtime model source: $RUNTIME_MODEL"
echo "[goal-baseline] benchmark endpoint: ${CLIENT_HOST}:${CLIENT_PORT}"
run_in_official_runtime "$OFFICIAL_RUNTIME_PYTHONPATH" python - <<'PY'
from importlib import metadata

import vllm
import vllm_ascend


def dist_version(*names: str) -> str:
  for name in names:
    try:
      return metadata.version(name)
    except metadata.PackageNotFoundError:
      continue
  return "not-installed"


print(f"[goal-baseline] vllm module: {vllm.__file__}")
print(f"[goal-baseline] vllm version: {getattr(vllm, '__version__', 'unknown')} (dist={dist_version('vllm')})")
print(f"[goal-baseline] vllm_ascend module: {vllm_ascend.__file__}")
print(
  "[goal-baseline] vllm_ascend version: "
  f"{getattr(vllm_ascend, '__version__', 'unknown')} "
  f"(dist={dist_version('vllm-ascend', 'vllm_ascend')})"
)
PY
echo "[goal-baseline] server command: $SERVER_COMMAND"
run_server_command &
SERVER_PID=$!
persist_managed_server_state

wait_for_server "$CLIENT_HOST" "$CLIENT_PORT"
persist_managed_server_state

echo "[goal-baseline] client command: $CLIENT_COMMAND"
run_client_command

run_in_official_runtime "$REPO_ROOT/src:$OFFICIAL_RUNTIME_PYTHONPATH" \
python -m vllm_hust_benchmark.cli export-leaderboard-artifact \
  "$SCENARIO" \
  --benchmark-result-file "$RAW_RESULT_FILE" \
  --constraints-file "$CONSTRAINTS_FILE" \
  --same-spec-file "$SAME_SPEC_FILE" \
  --output-dir "$ARTIFACT_DIR" \
  --run-id "$RUN_ID" \
  --engine "$ENGINE" \
  --engine-version "$ENGINE_VERSION" \
  --core-version "$OFFICIAL_CORE_VERSION" \
  --backend-version "$OFFICIAL_BACKEND_VERSION" \
  --model-name "$MODEL" \
  --model-parameters "$MODEL_PARAMETERS" \
  --model-precision "$MODEL_PRECISION" \
  --hardware-vendor "$HARDWARE_VENDOR" \
  --hardware-chip-model "$HARDWARE_CHIP_MODEL" \
  --chip-count "$CHIP_COUNT" \
  --node-count "$NODE_COUNT" \
  --submitter "$SUBMITTER" \
  --baseline-engine "$BASELINE_ENGINE" \
  --data-source "$DATA_SOURCE" \
  --input-length "$INPUT_LEN" \
  --output-length "$OUTPUT_LEN" \
  --git-commit "$GIT_COMMIT" \
  --github-repository "$GITHUB_REPOSITORY" \
  --github-ref "$GITHUB_REF"

echo "[goal-baseline] exported leaderboard artifact to $ARTIFACT_DIR"