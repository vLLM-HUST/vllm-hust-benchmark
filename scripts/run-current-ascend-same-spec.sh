#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VLLM_CLI_COMPAT=${VLLM_CLI_COMPAT:-"$REPO_ROOT/scripts/run_vllm_cli_compat.py"}
SPEC_FILE=${1:-"$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json"}
CONSTRAINTS_FILE=${CONSTRAINTS_FILE:-"$REPO_ROOT/docs/official-baselines/official-ascend-constraints.stub.json"}
WORKSPACE_ROOT=${VLLM_HUST_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}
CURRENT_RUNTIME_CWD=${CURRENT_RUNTIME_CWD:-"/tmp"}
CURRENT_VLLM_HUST_REPO=${CURRENT_VLLM_HUST_REPO:-"$WORKSPACE_ROOT/vllm-hust"}
CURRENT_VLLM_ASCEND_HUST_REPO=${CURRENT_VLLM_ASCEND_HUST_REPO:-"$WORKSPACE_ROOT/vllm-ascend-hust"}
CURRENT_RUNTIME_PYTHONPATH=${CURRENT_RUNTIME_PYTHONPATH:-}
CURRENT_ENV_PREFIX=${CURRENT_ENV_PREFIX:-"/root/miniconda3/envs/vllm-hust-dev"}
CURRENT_RUNTIME_PYTHON=${CURRENT_RUNTIME_PYTHON:-"$CURRENT_ENV_PREFIX/bin/python"}
CURRENT_VLLM_CACHE_ROOT=${CURRENT_VLLM_CACHE_ROOT:-"/data/shared_datasets/vllm-hust-benchmark/current-ascend-same-spec-cache"}
CURRENT_BENCHMARK_DATASET_ROOT=${CURRENT_BENCHMARK_DATASET_ROOT:-"/data/shared_datasets/vllm-hust-benchmark/current-benchmark-datasets"}
CURRENT_SHAREGPT_DATASET_URL=${CURRENT_SHAREGPT_DATASET_URL:-"https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"}
HF_HOME=${HF_HOME:-"/data/shared_datasets/vllm-hust-benchmark/huggingface"}
HF_HUB_CACHE=${HF_HUB_CACHE:-"$HF_HOME/hub"}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$HF_HOME/transformers"}
export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE
CURRENT_MODEL_PATH=${CURRENT_MODEL_PATH:-}
CURRENT_SERVER_HOST=${CURRENT_SERVER_HOST:-}
CURRENT_SERVER_PORT=${CURRENT_SERVER_PORT:-"8001"}
CURRENT_CLIENT_HOST=${CURRENT_CLIENT_HOST:-}
CURRENT_CLIENT_PORT=${CURRENT_CLIENT_PORT:-$CURRENT_SERVER_PORT}
CURRENT_USE_MANAGED_SERVER=${CURRENT_USE_MANAGED_SERVER:-0}
CURRENT_ENGINE=${CURRENT_ENGINE:-"vllm-hust"}
CURRENT_ENGINE_VERSION=${CURRENT_ENGINE_VERSION:-}
CURRENT_CORE_VERSION=${CURRENT_CORE_VERSION:-}
CURRENT_BACKEND_VERSION=${CURRENT_BACKEND_VERSION:-}
CURRENT_SUBMITTER=${CURRENT_SUBMITTER:-"same-spec-current"}
CURRENT_BASELINE_ENGINE=${CURRENT_BASELINE_ENGINE:-"vllm"}
CURRENT_DATA_SOURCE=${CURRENT_DATA_SOURCE:-"vllm-hust-benchmark"}
CURRENT_DTYPE=${CURRENT_DTYPE:-}
CURRENT_MODEL_NAME=${CURRENT_MODEL_NAME:-}
CURRENT_MODEL_PARAMETERS=${CURRENT_MODEL_PARAMETERS:-}
CURRENT_MODEL_PRECISION=${CURRENT_MODEL_PRECISION:-}
CURRENT_CLIENT_MODEL_NAME=${CURRENT_CLIENT_MODEL_NAME:-}
CURRENT_CLIENT_TOKENIZER=${CURRENT_CLIENT_TOKENIZER:-}
CURRENT_CLIENT_TEMPERATURE=${CURRENT_CLIENT_TEMPERATURE:-}
CURRENT_HARDWARE_CHIP_MODEL=${CURRENT_HARDWARE_CHIP_MODEL:-}
CURRENT_GITHUB_REPOSITORY=${CURRENT_GITHUB_REPOSITORY:-"vLLM-HUST/vllm-hust"}
CURRENT_GITHUB_REF=${CURRENT_GITHUB_REF:-$(git -C "$CURRENT_VLLM_HUST_REPO" branch --show-current 2>/dev/null || echo main)}
CURRENT_GIT_COMMIT=${CURRENT_GIT_COMMIT:-$(git -C "$CURRENT_VLLM_HUST_REPO" rev-parse HEAD 2>/dev/null || true)}
CURRENT_PLUGIN_ENGINE=${CURRENT_PLUGIN_ENGINE:-"vllm-ascend-hust"}
CURRENT_PLUGIN_GITHUB_REPOSITORY=${CURRENT_PLUGIN_GITHUB_REPOSITORY:-"vLLM-HUST/vllm-ascend-hust"}
CURRENT_PLUGIN_GITHUB_REF=${CURRENT_PLUGIN_GITHUB_REF:-$(git -C "$CURRENT_VLLM_ASCEND_HUST_REPO" branch --show-current 2>/dev/null || echo main)}
CURRENT_PLUGIN_GIT_COMMIT=${CURRENT_PLUGIN_GIT_COMMIT:-$(git -C "$CURRENT_VLLM_ASCEND_HUST_REPO" rev-parse HEAD 2>/dev/null || true)}
ASCEND_TOOLKIT_SET_ENV=${ASCEND_TOOLKIT_SET_ENV:-}
ASCEND_ATB_SET_ENV=${ASCEND_ATB_SET_ENV:-}
ASCEND_ATB_CXX_ABI=${ASCEND_ATB_CXX_ABI:-"1"}
RESULT_DIR=${RESULT_DIR:-"$REPO_ROOT/.benchmarks/current-ascend-same-spec"}
RUN_ID=${RUN_ID:-"current-ascend-same-spec-$(date -u +%Y%m%dT%H%M%SZ)"}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-600}
READY_STATUS_INTERVAL_SECONDS=${READY_STATUS_INTERVAL_SECONDS:-30}
CLIENT_READY_CHECK_TIMEOUT_SECONDS=${CLIENT_READY_CHECK_TIMEOUT_SECONDS:-$READY_TIMEOUT_SECONDS}
SERVER_START_RETRIES=${SERVER_START_RETRIES:-3}
SERVER_START_RETRY_DELAY_SECONDS=${SERVER_START_RETRY_DELAY_SECONDS:-10}
SERVER_PID=""
RUNNER_LOCK_FD=""

if [[ "$CURRENT_USE_MANAGED_SERVER" == "1" ]]; then
  CURRENT_RUNTIME_SOURCE_PYTHONPATH="$CURRENT_VLLM_HUST_REPO"
else
  CURRENT_RUNTIME_SOURCE_PYTHONPATH="$CURRENT_VLLM_ASCEND_HUST_REPO:$CURRENT_VLLM_HUST_REPO"
fi
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

if [[ ! -f "$VLLM_CLI_COMPAT" ]]; then
  echo "CLI compatibility wrapper not found: $VLLM_CLI_COMPAT" >&2
  exit 2
fi

detect_first_existing_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ -z "$ASCEND_TOOLKIT_SET_ENV" ]]; then
  ASCEND_TOOLKIT_SET_ENV=$(detect_first_existing_file \
    "/opt/hust-ascend-cann/Ascend/cann-8.5.0/set_env.sh" \
    "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
    "/usr/local/Ascend/latest/set_env.sh" \
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" \
    || true)
fi

if [[ -z "$ASCEND_ATB_SET_ENV" ]]; then
  ASCEND_ATB_SET_ENV=$(detect_first_existing_file \
    "/usr/local/Ascend/nnal/atb/set_env.sh" \
    "/opt/hust-ascend-cann/Ascend/nnal/atb/set_env.sh" \
    || true)
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

resolve_current_backend_version_from_git() {
  local described=""

  if [[ -n "$CURRENT_PLUGIN_GIT_COMMIT" ]]; then
    described=$(git -C "$CURRENT_VLLM_ASCEND_HUST_REPO" describe --tags --always "$CURRENT_PLUGIN_GIT_COMMIT" 2>/dev/null || true)
    if [[ -n "$described" ]]; then
      printf '%s' "$described"
      return 0
    fi
  fi

  described=$(git -C "$CURRENT_VLLM_ASCEND_HUST_REPO" describe --tags --always HEAD 2>/dev/null || true)
  if [[ -n "$described" ]]; then
    printf '%s' "$described"
    return 0
  fi

  git -C "$CURRENT_VLLM_ASCEND_HUST_REPO" rev-parse --short HEAD 2>/dev/null || true
}

detect_current_backend_version() {
  local raw_output=""
  local detected=""
  local fallback=""

  raw_output=$(run_in_current_runtime "$CURRENT_RUNTIME_PYTHONPATH" "$CURRENT_RUNTIME_PYTHON" - <<'PY'
from importlib import metadata

version = None
try:
    import vllm_ascend
    version = getattr(vllm_ascend, '__version__', None)
except Exception:
    version = None

if not version:
    for dist_name in ('vllm-ascend-hust', 'vllm-ascend'):
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

  fallback=$(resolve_current_backend_version_from_git)
  if is_valid_engine_version "$fallback"; then
    printf '%s' "$fallback"
    return 0
  fi

  printf '%s' "unknown"
}

append_export_arg_if_present() {
  local flag=$1
  local value=$2

  if [[ -n "$value" ]] && [[ "$value" != "null" ]]; then
    EXPORT_ARGS+=("$flag" "$value")
  fi
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
      set +u
      # shellcheck disable=SC1090
      source "$ASCEND_TOOLKIT_SET_ENV"
      set -u
    fi
    if [[ -f "$ASCEND_ATB_SET_ENV" ]]; then
      set +u
      # shellcheck disable=SC1090
      source "$ASCEND_ATB_SET_ENV" --cxx_abi="$ASCEND_ATB_CXX_ABI"
      set -u
    fi
    if [[ -d "$CURRENT_ENV_PREFIX/lib" ]]; then
      export LD_LIBRARY_PATH="$CURRENT_ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    fi
    if [[ "$CURRENT_USE_MANAGED_SERVER" == "1" ]]; then
      export VLLM_PLUGINS=""
      export TORCH_DEVICE_BACKEND_AUTOLOAD=0
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
      set +u
      # shellcheck disable=SC1090
      source "$ASCEND_TOOLKIT_SET_ENV"
      set -u
    fi
    if [[ -f "$ASCEND_ATB_SET_ENV" ]]; then
      set +u
      # shellcheck disable=SC1090
      source "$ASCEND_ATB_SET_ENV" --cxx_abi="$ASCEND_ATB_CXX_ABI"
      set -u
    fi
    if [[ -d "$CURRENT_ENV_PREFIX/lib" ]]; then
      export LD_LIBRARY_PATH="$CURRENT_ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    fi
    export VLLM_CACHE_ROOT="$CURRENT_VLLM_CACHE_ROOT"
    PYTHONUNBUFFERED=1 \
      PYTHONPATH="$CURRENT_RUNTIME_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
      "$CURRENT_RUNTIME_PYTHON" -u -m vllm.entrypoints.openai.api_server $SERVER_ARGS
  )
}

run_client_command() {
  case "$BENCHMARK_TYPE" in
    serve)
      run_in_current_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
        "$CURRENT_RUNTIME_PYTHON" "$VLLM_CLI_COMPAT" bench serve \
        --save-result \
        --result-dir "$RESULT_DIR" \
        --result-filename "$(basename "$RAW_RESULT_FILE")" \
        $CLIENT_ARGS
      ;;
    throughput|latency)
      run_in_current_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
        "$CURRENT_RUNTIME_PYTHON" "$VLLM_CLI_COMPAT" bench "$BENCHMARK_TYPE" \
        --output-json "$RAW_RESULT_FILE" \
        $CLIENT_ARGS
      ;;
    *)
      echo "Unsupported benchmark type for current same-spec runner: $BENCHMARK_TYPE" >&2
      return 2
      ;;
  esac
}

json2args() {
  local json_string=$1
  echo "$json_string" | jq -r '
    to_entries |
    map(if (.value | tostring) == "" then "--" + (.key | gsub("_"; "-")) else "--" + (.key | gsub("_"; "-")) + " " + (.value | tostring) end) |
    join(" ")
  '
}

download_file() {
  local url=$1
  local target_file=$2

  mkdir -p "$(dirname "$target_file")"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$target_file" "$url"
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --output "$target_file" "$url"
    return 0
  fi

  echo "wget or curl is required to download benchmark datasets" >&2
  return 2
}

download_json_file_atomic() {
  local url=$1
  local target_file=$2
  local lock_file="$target_file.lock"
  local tmp_file="$target_file.tmp.$$"

  if [[ "$url" != https://hf-mirror.com/* ]]; then
    echo "benchmark dataset downloads must use hf-mirror.com: $url" >&2
    return 2
  fi

  mkdir -p "$(dirname "$target_file")"
  (
    flock -x 9
    if [[ -f "$target_file" ]] && "$CURRENT_RUNTIME_PYTHON" -m json.tool "$target_file" >/dev/null 2>&1; then
      return 0
    fi

    rm -f "$tmp_file"
    download_file "$url" "$tmp_file"
    "$CURRENT_RUNTIME_PYTHON" -m json.tool "$tmp_file" >/dev/null
    mv -f "$tmp_file" "$target_file"
  ) 9>"$lock_file"
  local status=$?
  rm -f "$tmp_file"
  return "$status"
}

ensure_runtime_dataset_available() {
  local dataset_path=${1:-}
  local sharegpt_target

  [[ -z "$dataset_path" ]] && return 0

  case "$dataset_path" in
    /*)
      if [[ ! -f "$dataset_path" ]]; then
        echo "runtime dataset path not found: $dataset_path" >&2
        return 2
      fi
      return 0
      ;;
    ShareGPT_V3_unfiltered_cleaned_split.json)
      sharegpt_target="$CURRENT_BENCHMARK_DATASET_ROOT/$dataset_path"
      if [[ -f "$sharegpt_target" ]] && "$CURRENT_RUNTIME_PYTHON" -m json.tool "$sharegpt_target" >/dev/null 2>&1; then
        return 0
      fi
      echo "[same-spec-current] downloading ShareGPT benchmark dataset to $sharegpt_target"
      download_json_file_atomic "$CURRENT_SHAREGPT_DATASET_URL" "$sharegpt_target"
      ;;
    benchmarks/*)
      if [[ ! -f "$CURRENT_VLLM_HUST_REPO/$dataset_path" ]]; then
        echo "benchmark dataset path not found in current vllm worktree: $CURRENT_VLLM_HUST_REPO/$dataset_path" >&2
        return 2
      fi
      ;;
  esac
}

normalized_client_parameters_json() {
  run_in_current_runtime "$REPO_ROOT/src${CURRENT_RUNTIME_PYTHONPATH:+:$CURRENT_RUNTIME_PYTHONPATH}" \
    env SAME_SPEC_FILE="$SAME_SPEC_FILE" \
    BENCHMARK_TYPE="$BENCHMARK_TYPE" \
    CLIENT_READY_CHECK_TIMEOUT_SECONDS="$CLIENT_READY_CHECK_TIMEOUT_SECONDS" \
    CURRENT_VLLM_WORKTREE="$CURRENT_VLLM_HUST_REPO" \
    BENCHMARK_REPO_ROOT="$REPO_ROOT" \
    CURRENT_BENCHMARK_DATASET_ROOT="$CURRENT_BENCHMARK_DATASET_ROOT" \
    "$CURRENT_RUNTIME_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

from vllm_hust_benchmark.official_runtime_inputs import normalize_client_parameters

payload = json.loads(Path(os.environ["SAME_SPEC_FILE"]).read_text(encoding="utf-8"))
ready_timeout = int(os.environ.get("CLIENT_READY_CHECK_TIMEOUT_SECONDS") or 0)
parameters = normalize_client_parameters(
    payload["resolved_client_parameters"],
    benchmark_type=os.environ["BENCHMARK_TYPE"],
    ready_check_timeout_sec=ready_timeout,
    vllm_worktree=os.environ.get("CURRENT_VLLM_WORKTREE"),
    benchmark_repo=os.environ.get("BENCHMARK_REPO_ROOT"),
    dataset_cache_root=os.environ.get("CURRENT_BENCHMARK_DATASET_ROOT"),
)
client_model_name = os.environ.get("CURRENT_CLIENT_MODEL_NAME", "").strip()
if os.environ["BENCHMARK_TYPE"] == "serve" and client_model_name:
    parameters["model"] = client_model_name
client_tokenizer = os.environ.get("CURRENT_CLIENT_TOKENIZER", "").strip()
if client_tokenizer:
    parameters["tokenizer"] = client_tokenizer
client_temperature = os.environ.get("CURRENT_CLIENT_TEMPERATURE", "").strip()
if os.environ["BENCHMARK_TYPE"] == "serve" and client_temperature:
    parameters["temperature"] = client_temperature
print(
    json.dumps(
        parameters,
        separators=(",", ":"),
        ensure_ascii=True,
    )
)
PY
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
  )

  if [[ -n "$CURRENT_DTYPE" ]]; then
    resolve_args+=(--dtype "$CURRENT_DTYPE")
  fi
  if [[ -n "$CURRENT_MODEL_NAME" ]]; then
    resolve_args+=(--model "$CURRENT_MODEL_NAME")
  fi
  if [[ -n "$CURRENT_MODEL_PARAMETERS" ]]; then
    resolve_args+=(--model-parameters "$CURRENT_MODEL_PARAMETERS")
  fi
  if [[ -n "$CURRENT_MODEL_PRECISION" ]]; then
    resolve_args+=(--model-precision "$CURRENT_MODEL_PRECISION")
  fi
  if [[ -n "$MODEL_QUANTIZATION" ]]; then
    resolve_args+=(--model-quantization "$MODEL_QUANTIZATION")
  fi
  if [[ -n "$CURRENT_HARDWARE_CHIP_MODEL" ]]; then
    resolve_args+=(--hardware-chip-model "$CURRENT_HARDWARE_CHIP_MODEL")
  fi

  if [[ "$BENCHMARK_TYPE" == "serve" ]]; then
    resolve_args+=(
      --server-port "$CURRENT_SERVER_PORT"
      --client-port "$CURRENT_CLIENT_PORT"
    )

    if [[ -n "$CURRENT_SERVER_HOST" ]]; then
      resolve_args+=(--server-host "$CURRENT_SERVER_HOST")
    fi
    if [[ -n "$CURRENT_CLIENT_HOST" ]]; then
      resolve_args+=(--client-host "$CURRENT_CLIENT_HOST")
    fi
  fi

  run_in_current_runtime "$REPO_ROOT/src${CURRENT_RUNTIME_PYTHONPATH:+:$CURRENT_RUNTIME_PYTHONPATH}" "${resolve_args[@]}"
}

resolve_scenario_benchmark_type() {
  run_in_current_runtime "$REPO_ROOT/src${CURRENT_RUNTIME_PYTHONPATH:+:$CURRENT_RUNTIME_PYTHONPATH}" \
    env SCENARIO_NAME="$SCENARIO" \
    "$CURRENT_RUNTIME_PYTHON" -c "import os; from vllm_hust_benchmark.registry import get_scenario; print(get_scenario(os.environ['SCENARIO_NAME']).benchmark_type)"
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

server_log_indicates_node_env_failure() {
  local log_file=$1

  [[ -f "$log_file" ]] || return 1

  grep -Eq "DrvMngGetConsoleLogLevel failed|dcmi model initialized failed|ret is -8020|drvRet=87|drvRetCode=87|ErrCode=507899|error code is 507899|rtGetDeviceCount|Can't get ascend_hal device count|driver error:internal error|Resource_Busy\(EL0005\)|The resources are busy|ERR99999 UNKNOWN applicaiton exception|ERR99999 UNKNOWN application exception|Engine core initialization failed" "$log_file"
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
        if server_log_indicates_node_env_failure "$SERVER_STDOUT_LOG"; then
          echo "Detected Ascend node-level runtime failure during current same-spec startup" >&2
          return 86
        fi
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
    if server_log_indicates_node_env_failure "$SERVER_STDOUT_LOG"; then
      echo "Detected Ascend node-level runtime failure while waiting for current same-spec startup" >&2
      return 86
    fi
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
MODEL_QUANTIZATION=$(jq -r '.model_quantization // empty' "$SPEC_FILE")
if [[ -n "$CURRENT_MODEL_NAME" ]]; then
  MODEL="$CURRENT_MODEL_NAME"
fi
if [[ -n "$CURRENT_MODEL_PARAMETERS" ]]; then
  MODEL_PARAMETERS="$CURRENT_MODEL_PARAMETERS"
fi
if [[ -n "$CURRENT_MODEL_PRECISION" ]]; then
  MODEL_PRECISION="$CURRENT_MODEL_PRECISION"
fi
if [[ -n "${CURRENT_MODEL_QUANTIZATION:-}" ]]; then
  MODEL_QUANTIZATION="$CURRENT_MODEL_QUANTIZATION"
fi
HARDWARE_VENDOR=$(jq -r '.hardware_vendor' "$SPEC_FILE")
HARDWARE_CHIP_MODEL=$(jq -r '.hardware_chip_model' "$SPEC_FILE")
if [[ -n "$CURRENT_HARDWARE_CHIP_MODEL" ]]; then
  HARDWARE_CHIP_MODEL="$CURRENT_HARDWARE_CHIP_MODEL"
fi
CHIP_COUNT=$(jq -r '.chip_count' "$SPEC_FILE")
NODE_COUNT=$(jq -r '.node_count' "$SPEC_FILE")
SCENARIO=$(jq -r '.scenario' "$SPEC_FILE")
INPUT_LEN=$(jq -r '.client_parameters.input_len' "$SPEC_FILE")
OUTPUT_LEN=$(jq -r '.client_parameters.output_len' "$SPEC_FILE")
BENCHMARK_TYPE=$(resolve_scenario_benchmark_type)

if [[ -z "$CURRENT_ENGINE_VERSION" ]]; then
  CURRENT_ENGINE_VERSION=$(detect_current_engine_version)
fi

if [[ -z "$CURRENT_CORE_VERSION" ]]; then
  CURRENT_CORE_VERSION="$CURRENT_ENGINE_VERSION"
fi

if [[ -z "$CURRENT_BACKEND_VERSION" ]]; then
  CURRENT_BACKEND_VERSION=$(detect_current_backend_version)
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

resolved_dataset_path=$(jq -r '.resolved_client_parameters.dataset_path // empty' "$SAME_SPEC_FILE")
ensure_runtime_dataset_available "$resolved_dataset_path"

SERVER_HOST=""
SERVER_PORT=""
CLIENT_HOST=""
CLIENT_PORT=""
SERVER_ARGS=""
CLIENT_ARGS=$(json2args "$(normalized_client_parameters_json)")

RAW_RESULT_FILE="$RESULT_DIR/raw_benchmark_result.json"
ARTIFACT_DIR="$RESULT_DIR/submission"
SERVER_STDOUT_LOG="$RESULT_DIR/server.stdout.log"

if [[ "$BENCHMARK_TYPE" == "serve" ]]; then
  SERVER_HOST=$(jq -r '.resolved_server_parameters.host' "$SAME_SPEC_FILE")
  SERVER_PORT=$(jq -r '.resolved_server_parameters.port' "$SAME_SPEC_FILE")
  CLIENT_HOST=$(jq -r '.resolved_client_parameters.host' "$SAME_SPEC_FILE")
  CLIENT_PORT=$(jq -r '.resolved_client_parameters.port' "$SAME_SPEC_FILE")

  if [[ "$CURRENT_USE_MANAGED_SERVER" != "1" ]]; then
    assert_target_port_available "Current same-spec benchmark" "$CLIENT_HOST" "$CLIENT_PORT"
  fi
  SERVER_ARGS=$(json2args "$(jq -c '.resolved_server_parameters | del(.disable_log_requests)' "$SAME_SPEC_FILE")")
fi

echo "[same-spec-current] neutral cwd: $CURRENT_RUNTIME_CWD"
echo "[same-spec-current] vllm cache root: $CURRENT_VLLM_CACHE_ROOT"
echo "[same-spec-current] runtime model source: $RUNTIME_MODEL"
echo "[same-spec-current] resolved spec file: $SAME_SPEC_FILE"
echo "[same-spec-current] benchmark type: $BENCHMARK_TYPE"
if [[ "$BENCHMARK_TYPE" == "serve" ]]; then
  echo "[same-spec-current] benchmark endpoint: ${CLIENT_HOST}:${CLIENT_PORT}"
  if [[ "$CURRENT_USE_MANAGED_SERVER" == "1" ]]; then
    echo "[same-spec-current] using externally managed server"
    wait_for_server "$CLIENT_HOST" "$CLIENT_PORT"
  else
    server_ready=0
    for start_attempt in $(seq 1 "$SERVER_START_RETRIES"); do
      : >"$SERVER_STDOUT_LOG"
      run_server_command >"$SERVER_STDOUT_LOG" 2>&1 &
      SERVER_PID=$!
      persist_managed_server_state

      if wait_for_server "$CLIENT_HOST" "$CLIENT_PORT"; then
        persist_managed_server_state
        server_ready=1
        break
      fi

      server_wait_status=$?
      if [[ "$server_wait_status" -eq 86 && "$start_attempt" -lt "$SERVER_START_RETRIES" ]]; then
        echo "[same-spec-current] detected transient Ascend runtime startup failure; retrying server start in ${SERVER_START_RETRY_DELAY_SECONDS}s (attempt ${start_attempt}/${SERVER_START_RETRIES})" >&2
        cleanup_managed_server || true
        sleep "$SERVER_START_RETRY_DELAY_SECONDS"
        continue
      fi

      exit "$server_wait_status"
    done

    if [[ "$server_ready" != "1" ]]; then
      echo "[same-spec-current] vLLM server did not become ready after ${SERVER_START_RETRIES} start attempt(s)" >&2
      exit 1
    fi
  fi
fi

run_client_command

EXPORT_ARGS=(
  "$SCENARIO"
  --benchmark-result-file "$RAW_RESULT_FILE"
  --constraints-file "$CONSTRAINTS_FILE"
  --same-spec-file "$SAME_SPEC_FILE"
  --output-dir "$ARTIFACT_DIR"
  --run-id "$RUN_ID"
  --engine "$CURRENT_ENGINE"
  --engine-version "$CURRENT_ENGINE_VERSION"
  --core-version "$CURRENT_CORE_VERSION"
  --backend-version "$CURRENT_BACKEND_VERSION"
  --model-name "$MODEL"
  --model-parameters "$MODEL_PARAMETERS"
  --model-precision "$MODEL_PRECISION"
  ${MODEL_QUANTIZATION:+--quantization "$MODEL_QUANTIZATION"}
  --hardware-vendor "$HARDWARE_VENDOR"
  --hardware-chip-model "$HARDWARE_CHIP_MODEL"
  --chip-count "$CHIP_COUNT"
  --node-count "$NODE_COUNT"
  --submitter "$CURRENT_SUBMITTER"
  --baseline-engine "$CURRENT_BASELINE_ENGINE"
  --data-source "$CURRENT_DATA_SOURCE"
  --git-commit "$CURRENT_GIT_COMMIT"
  --github-repository "$CURRENT_GITHUB_REPOSITORY"
  --github-ref "$CURRENT_GITHUB_REF"
  --runtime-python "$CURRENT_RUNTIME_PYTHON"
  --engine-source-repository "$CURRENT_GITHUB_REPOSITORY"
  --engine-source-ref "$CURRENT_GITHUB_REF"
  --engine-source-commit "$CURRENT_GIT_COMMIT"
  --plugin-source-engine "$CURRENT_PLUGIN_ENGINE"
  --plugin-source-repository "$CURRENT_PLUGIN_GITHUB_REPOSITORY"
  --plugin-source-ref "$CURRENT_PLUGIN_GITHUB_REF"
  --plugin-source-commit "$CURRENT_PLUGIN_GIT_COMMIT"
)

append_export_arg_if_present --input-length "$INPUT_LEN"
append_export_arg_if_present --output-length "$OUTPUT_LEN"

run_in_current_runtime "$REPO_ROOT/src${CURRENT_RUNTIME_PYTHONPATH:+:$CURRENT_RUNTIME_PYTHONPATH}" \
"$CURRENT_RUNTIME_PYTHON" -m vllm_hust_benchmark.cli export-leaderboard-artifact \
  "${EXPORT_ARGS[@]}"

echo "[same-spec-current] exported leaderboard artifact to $ARTIFACT_DIR"
