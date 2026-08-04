#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PREPARE_SCRIPT=${PREPARE_SCRIPT:-"$REPO_ROOT/scripts/prepare-official-ascend-baseline-env.sh"}
VLLM_CLI_COMPAT=${VLLM_CLI_COMPAT:-"$REPO_ROOT/scripts/run_vllm_cli_compat.py"}
SPEC_FILE=${1:-"$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json"}
CONSTRAINTS_FILE=${CONSTRAINTS_FILE:-"$REPO_ROOT/docs/official-baselines/official-ascend-constraints.stub.json"}
WORKSPACE_ROOT=${VLLM_HUST_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}
OFFICIAL_VLLM_REPO=${OFFICIAL_VLLM_REPO:-"$WORKSPACE_ROOT/reference-repos/vllm"}
OFFICIAL_VLLM_ASCEND_REPO=${OFFICIAL_VLLM_ASCEND_REPO:-"$WORKSPACE_ROOT/reference-repos/vllm-ascend"}
OFFICIAL_VLLM_WORKTREE=${OFFICIAL_VLLM_WORKTREE:-"/tmp/vllm-v0180"}
OFFICIAL_VLLM_ASCEND_WORKTREE=${OFFICIAL_VLLM_ASCEND_WORKTREE:-"/tmp/vllm-ascend-v0180"}
OFFICIAL_RUNTIME_CWD=${OFFICIAL_RUNTIME_CWD:-"/tmp"}
OFFICIAL_VLLM_CACHE_ROOT=${OFFICIAL_VLLM_CACHE_ROOT:-"/data/shared_datasets/vllm-hust-benchmark/official-ascend-goal-baseline-cache"}
OFFICIAL_BENCHMARK_DATASET_ROOT=${OFFICIAL_BENCHMARK_DATASET_ROOT:-"/data/shared_datasets/vllm-hust-benchmark/official-baseline-datasets"}
OFFICIAL_TRACE_ASSET_ROOT=${OFFICIAL_TRACE_ASSET_ROOT:-"$REPO_ROOT/.benchmarks/traces"}
OFFICIAL_SHAREGPT_DATASET_URL=${OFFICIAL_SHAREGPT_DATASET_URL:-"https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"}
HF_HOME=${HF_HOME:-"/data/shared_datasets/vllm-hust-benchmark/huggingface"}
HF_HUB_CACHE=${HF_HUB_CACHE:-"$HF_HOME/hub"}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$HF_HOME/transformers"}
export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE
OFFICIAL_RUNTIME_PYTHON=${OFFICIAL_RUNTIME_PYTHON:-"$GOAL_BASELINE_ENV_PREFIX/bin/python"}
OFFICIAL_RUNTIME_IMAGE=${OFFICIAL_RUNTIME_IMAGE:-}
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
HOST_PYTHON_BIN=${HOST_PYTHON_BIN:-$(command -v python3 || command -v python || true)}
GOAL_BASELINE_ENV_PREFIX=${GOAL_BASELINE_ENV_PREFIX:-}
RESULT_DIR=${RESULT_DIR:-"$REPO_ROOT/.benchmarks/official-ascend-goal-baseline"}
RUN_ID=${RUN_ID:-"official-ascend-jan-2026-$(date -u +%Y%m%dT%H%M%SZ)"}
SERVER_START_RETRIES=${SERVER_START_RETRIES:-8}
SERVER_START_RETRY_DELAY_SECONDS=${SERVER_START_RETRY_DELAY_SECONDS:-10}
DEVICE_SELECTION_RETRIES=${DEVICE_SELECTION_RETRIES:-20}
DEVICE_SELECTION_RETRY_DELAY_SECONDS=${DEVICE_SELECTION_RETRY_DELAY_SECONDS:-30}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-900}
READY_STATUS_INTERVAL_SECONDS=${READY_STATUS_INTERVAL_SECONDS:-30}
CLIENT_READY_CHECK_TIMEOUT_SECONDS=${CLIENT_READY_CHECK_TIMEOUT_SECONDS:-$READY_TIMEOUT_SECONDS}
ASCEND_RUNTIME_READY_TIMEOUT_SECONDS=${ASCEND_RUNTIME_READY_TIMEOUT_SECONDS:-30}
ASCEND_RUNTIME_READY_POLL_SECONDS=${ASCEND_RUNTIME_READY_POLL_SECONDS:-10}
RESOURCE_BUSY_EXIT_CODE=${RESOURCE_BUSY_EXIT_CODE:-75}
NPU_SMI_TIMEOUT_SECONDS=${NPU_SMI_TIMEOUT_SECONDS:-20}
GOAL_BASELINE_DEVICE_PREFERENCE_FILE=${GOAL_BASELINE_DEVICE_PREFERENCE_FILE:-}
SERVER_PID=""
RUNNER_LOCK_FD=""
PEAK_HBM_SAMPLER_PID=""
PEAK_HBM_EVIDENCE_FILE=""

if [[ -z "$GOAL_BASELINE_ENV_PREFIX" ]]; then
  echo "GOAL_BASELINE_ENV_PREFIX is required" >&2
  exit 2
fi

if [[ ! -x "$OFFICIAL_RUNTIME_PYTHON" ]]; then
  echo "OFFICIAL_RUNTIME_PYTHON is not executable: $OFFICIAL_RUNTIME_PYTHON" >&2
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

if [[ ! -f "$VLLM_CLI_COMPAT" ]]; then
  echo "CLI compatibility wrapper not found: $VLLM_CLI_COMPAT" >&2
  exit 2
fi

if [[ -z "$HOST_PYTHON_BIN" ]] || [[ ! -x "$HOST_PYTHON_BIN" ]]; then
  echo "python3 or python is required for benchmark repo utilities" >&2
  exit 2
fi

SPEC_FILE=$(realpath "$SPEC_FILE")
CONSTRAINTS_FILE=$(realpath "$CONSTRAINTS_FILE")
RESULT_DIR=$(realpath -m "$RESULT_DIR")
OFFICIAL_VLLM_CACHE_ROOT=$(realpath -m "$OFFICIAL_VLLM_CACHE_ROOT")

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

set_ascend_visible_devices_scope() {
  local visible_devices=${1:-}

  if [[ -n "$visible_devices" ]]; then
    export ASCEND_VISIBLE_DEVICES="$visible_devices"
    export ASCEND_RT_VISIBLE_DEVICES="$visible_devices"
    return 0
  fi

  unset ASCEND_VISIBLE_DEVICES
  unset ASCEND_RT_VISIBLE_DEVICES
}

read_preferred_ascend_device() {
  local preference_file=${GOAL_BASELINE_DEVICE_PREFERENCE_FILE:-}
  local preferred_device=""

  [[ -n "$preference_file" ]] || return 1
  [[ -f "$preference_file" ]] || return 1

  preferred_device=$(tr -d '[:space:]' < "$preference_file")
  [[ "$preferred_device" =~ ^[0-9]+$ ]] || return 1

  printf '%s\n' "$preferred_device"
}

persist_preferred_ascend_device() {
  local selected_device=${1:-}
  local preference_file=${GOAL_BASELINE_DEVICE_PREFERENCE_FILE:-}

  [[ -n "$preference_file" ]] || return 0
  [[ "$selected_device" =~ ^[0-9]+$ ]] || return 0

  mkdir -p "$(dirname "$preference_file")"
  printf '%s\n' "$selected_device" > "$preference_file"
}

source_ascend_runtime_env() {
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
}

run_in_official_runtime() {
  local pythonpath_prefix=$1
  shift
  (
    cd "$OFFICIAL_RUNTIME_CWD"
    source_ascend_runtime_env
    export VLLM_CACHE_ROOT="$OFFICIAL_VLLM_CACHE_ROOT"
    export HF_HOME="${HF_HOME:-/data/shared_datasets/vllm-hust-benchmark/huggingface}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
    if [[ -n "${OFFICIAL_CORE_VERSION:-}" ]]; then
      export VLLM_VERSION="$OFFICIAL_CORE_VERSION"
    fi
    if [[ -n "${OFFICIAL_RUNTIME_VLLM_BATCH_INVARIANT:-}" ]]; then
      export VLLM_BATCH_INVARIANT="$OFFICIAL_RUNTIME_VLLM_BATCH_INVARIANT"
    else
      unset VLLM_BATCH_INVARIANT
    fi
    export PATH="$GOAL_BASELINE_ENV_PREFIX/bin:$PATH"
    PYTHONPATH="$pythonpath_prefix${PYTHONPATH:+:$PYTHONPATH}" \
      "$@"
  )
}

run_in_official_runtime_python() {
  local pythonpath_prefix=$1
  shift
  local script_file
  local status=0

  script_file=$(mktemp "${TMPDIR:-/tmp}/official-runtime-python-XXXXXX.py")
  cat > "$script_file"

  if run_in_official_runtime "$pythonpath_prefix" "$@" python "$script_file"; then
    status=0
  else
    status=$?
  fi

  rm -f "$script_file"
  return "$status"
}

capture_initial_ascend_device_scope() {
  if [[ "${GOAL_BASELINE_INITIAL_ASCEND_DEVICE_SCOPE_CAPTURED:-0}" == "1" ]]; then
    return 0
  fi

  if [[ -n "${ASCEND_VISIBLE_DEVICES+x}" ]]; then
    GOAL_BASELINE_INITIAL_ASCEND_VISIBLE_DEVICES_IS_SET=1
    GOAL_BASELINE_INITIAL_ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-}
  else
    GOAL_BASELINE_INITIAL_ASCEND_VISIBLE_DEVICES_IS_SET=0
    unset GOAL_BASELINE_INITIAL_ASCEND_VISIBLE_DEVICES
  fi

  if [[ -n "${ASCEND_RT_VISIBLE_DEVICES+x}" ]]; then
    GOAL_BASELINE_INITIAL_ASCEND_RT_VISIBLE_DEVICES_IS_SET=1
    GOAL_BASELINE_INITIAL_ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}
  else
    GOAL_BASELINE_INITIAL_ASCEND_RT_VISIBLE_DEVICES_IS_SET=0
    unset GOAL_BASELINE_INITIAL_ASCEND_RT_VISIBLE_DEVICES
  fi

  GOAL_BASELINE_INITIAL_ASCEND_DEVICE_SCOPE_CAPTURED=1
}

normalize_visible_devices() {
  local raw_value=${1:-}
  local device
  local -a devices=()
  local normalized_devices

  IFS=',' read -r -a raw_devices <<< "$raw_value"
  for device in "${raw_devices[@]}"; do
    device=${device//[[:space:]]/}
    if [[ -n "$device" ]]; then
      devices+=("$device")
    fi
  done

  if [[ ${#devices[@]} -eq 0 ]]; then
    return 1
  fi

  normalized_devices=$(IFS=','; echo "${devices[*]}")
  printf '%s\n' "$normalized_devices"
}

resolve_npu_smi_bin() {
  local candidate

  if candidate=$(command -v npu-smi 2>/dev/null) && [[ -n "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  for candidate in /usr/local/bin/npu-smi /usr/local/sbin/npu-smi /usr/sbin/npu-smi /usr/bin/npu-smi; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

select_ascend_device() {
  local selection_attempt=${1:-1}
  local npu_smi_bin=${2:-}
  local preferred_device=${3:-}

  ASCEND_DEVICE_SELECTION_ATTEMPT="$selection_attempt" \
    NPU_SMI_BIN="$npu_smi_bin" \
    PREFERRED_ASCEND_DEVICE="$preferred_device" \
    "$HOST_PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import re
import subprocess
import sys


def parse_logical_map(mapping_output: str) -> dict[tuple[str, str], int]:
    logical_map = {}
    for line in mapping_output.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        npu_id, chip_id, logical_id = parts[:3]
        if npu_id.isdigit() and chip_id.isdigit() and logical_id.isdigit():
            logical_map[(npu_id, chip_id)] = int(logical_id)
    return logical_map


def list_logical_devices(mapping_output: str) -> list[int]:
    logical_devices = set(parse_logical_map(mapping_output).values())
    return sorted(logical_devices)


def list_status_devices(info_output: str) -> list[int]:
    status_devices = set()
    for raw_line in info_output.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue

        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 2:
            continue

        left_column = parts[0].split()
        if len(left_column) >= 2 and left_column[0].isdigit() and parts[1] and ":" not in parts[1]:
            status_devices.add(int(left_column[0]))

    return sorted(status_devices)


def list_process_busy_devices(info_output: str) -> set[int]:
    busy_devices = set()
    in_process_section = False

    for raw_line in info_output.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue

        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 3:
            continue

        if parts[1] == "Process id":
            in_process_section = True
            continue

        if not in_process_section:
            continue

        left_column = parts[0].split()
        if len(left_column) >= 2 and left_column[0].isdigit() and parts[1].isdigit():
            busy_devices.add(int(left_column[0]))

    return busy_devices


def list_devnode_devices() -> list[int]:
    devnode_devices = set()
    for device_path in Path("/dev").glob("davinci[0-9]*"):
        suffix = device_path.name.removeprefix("davinci")
        if suffix.isdigit():
            devnode_devices.add(int(suffix))
    return sorted(devnode_devices)


def run_npu_smi(*args: str) -> subprocess.CompletedProcess[str] | None:
  npu_smi_bin = os.environ.get("NPU_SMI_BIN")
  if not npu_smi_bin:
    return None

  try:
    timeout_seconds = float(os.environ.get("NPU_SMI_TIMEOUT_SECONDS", "20"))
  except ValueError:
    timeout_seconds = 20.0

  clean_env = {
    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    "HOME": os.environ.get("HOME", ""),
    "LANG": os.environ.get("LANG", "C.UTF-8"),
    "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
  }

  try:
    return subprocess.run(
      [npu_smi_bin, *args],
      check=False,
      capture_output=True,
      text=True,
      timeout=timeout_seconds,
      env=clean_env,
    )
  except subprocess.TimeoutExpired:
    return None
  except Exception:
    return None


def classify_npu_smi_failure(
  result: subprocess.CompletedProcess[str] | None,
) -> str | None:
  if result is None or result.returncode == 0:
    return None

  details = f"{result.stdout}\n{result.stderr}".lower()
  if "device is used" in details or "-8020" in details:
    return "device-used"
  if "permission" in details or "ret=4" in details:
    return "permission-limited"
  return f"exit-{result.returncode}"


def annotate_fallback_source(base_source: str, failure_reason: str | None) -> str:
  if not failure_reason:
    return base_source
  return f"{base_source}+npu-smi-{failure_reason}"


def parse_preferred_device() -> int | None:
  raw_value = os.environ.get("PREFERRED_ASCEND_DEVICE", "").strip()
  if raw_value.isdigit():
    return int(raw_value)
  return None


def select_best_idle_device(
  info_output: str,
  logical_map: dict[tuple[str, str], int],
  busy_devices: set[int],
  preferred_device: int | None,
) -> tuple[int, str] | None:
  hbm_usage_pattern = re.compile(r"(\d+)\s*/\s*(\d+)\s*$")
  device_stats = []
  current_npu_id = None
  current_health = None

  for raw_line in info_output.splitlines():
    line = raw_line.strip()
    if not line.startswith("|"):
      continue

    parts = [part.strip() for part in line.strip("|").split("|")]
    if len(parts) < 3:
      continue

    left_column = parts[0].split()
    if len(left_column) >= 2 and left_column[0].isdigit() and parts[1] and ":" not in parts[1]:
      current_npu_id = left_column[0]
      current_health = parts[1]
      continue

    if current_npu_id is None or current_health != "OK":
      continue

    if len(left_column) != 1 or not left_column[0].isdigit() or ":" not in parts[1]:
      continue

    chip_id = left_column[0]
    logical_id = logical_map.get((current_npu_id, chip_id))
    device_source = "idle"
    if logical_id is None:
      if chip_id != "0":
        continue
      logical_id = int(current_npu_id)
      device_source = "status-fallback"

    if logical_id in busy_devices:
      continue

    hbm_match = hbm_usage_pattern.search(parts[2])
    if hbm_match is None:
      continue

    used_memory_mb = int(hbm_match.group(1))
    total_memory_mb = int(hbm_match.group(2))
    free_memory_mb = max(0, total_memory_mb - used_memory_mb)
    device_stats.append((logical_id, free_memory_mb, device_source))

  if not device_stats:
    return None

  if preferred_device is not None:
    for logical_id, _, device_source in device_stats:
      if logical_id == preferred_device:
        return logical_id, f"preferred-{device_source}"

  device_stats.sort(key=lambda item: (-item[1], item[0], item[2]))
  selected_device, _, selected_source = device_stats[0]
  return selected_device, selected_source


mapping_result = run_npu_smi("info", "-m")
mapping_failure_reason = classify_npu_smi_failure(mapping_result)
logical_map = {}
logical_devices = []
if mapping_result is not None and mapping_result.returncode == 0:
    logical_map = parse_logical_map(mapping_result.stdout)
    logical_devices = list_logical_devices(mapping_result.stdout)

selection_attempt = max(1, int(os.environ.get("ASCEND_DEVICE_SELECTION_ATTEMPT", "1")))
preferred_device = parse_preferred_device()

info_result = run_npu_smi("info")
info_failure_reason = classify_npu_smi_failure(info_result)
if info_result is not None and info_result.returncode == 0:
    busy_devices = list_process_busy_devices(info_result.stdout)

    selected_device = select_best_idle_device(
        info_result.stdout,
        logical_map,
        busy_devices,
        preferred_device,
    )
    if selected_device is not None:
        device_id, device_source = selected_device
        print(f"{device_id}\t{device_source}")
        sys.exit(0)

    status_devices = list_status_devices(info_result.stdout)
    if busy_devices:
        status_devices = [device for device in status_devices if device not in busy_devices]

    if preferred_device is not None and preferred_device in status_devices:
        print(f"{preferred_device}\tpreferred-status")
        sys.exit(0)

    if status_devices:
        fallback_device = status_devices[(selection_attempt - 1) % len(status_devices)]
        print(f"{fallback_device}\tstatus-round-robin")
        sys.exit(0)

    if busy_devices:
        busy_device_list = sorted(busy_devices)
        print("__ALL_BUSY__\t" + ",".join(str(device) for device in busy_device_list))
        sys.exit(0)

fallback_failure_reason = info_failure_reason or mapping_failure_reason

if preferred_device is not None and preferred_device in logical_devices:
    print(
        f"{preferred_device}\tpreferred-"
        f"{annotate_fallback_source('logical-round-robin', fallback_failure_reason)}"
    )
    sys.exit(0)

if logical_devices:
    fallback_device = logical_devices[(selection_attempt - 1) % len(logical_devices)]
    print(f"{fallback_device}\t{annotate_fallback_source('logical-round-robin', fallback_failure_reason)}")
    sys.exit(0)

devnode_devices = list_devnode_devices()
if preferred_device is not None and preferred_device in devnode_devices:
    print(
        f"{preferred_device}\tpreferred-"
        f"{annotate_fallback_source('devnode-round-robin', fallback_failure_reason)}"
    )
    sys.exit(0)

if devnode_devices:
    fallback_device = devnode_devices[(selection_attempt - 1) % len(devnode_devices)]
    print(f"{fallback_device}\t{annotate_fallback_source('devnode-round-robin', fallback_failure_reason)}")
    sys.exit(0)

sys.exit(1)
PY
}

verify_explicit_multicard_scope_idle() {
  local visible_devices=$1
  local npu_smi_bin

  npu_smi_bin=$(resolve_npu_smi_bin 2>/dev/null) || {
    echo "npu-smi is required to prove every explicitly scoped multi-card device is idle" >&2
    return 2
  }
  NPU_SMI_BIN="$npu_smi_bin" VISIBLE_DEVICES="$visible_devices" \
    NPU_SMI_TIMEOUT_SECONDS="$NPU_SMI_TIMEOUT_SECONDS" "$HOST_PYTHON_BIN" - <<'PY'
import os
import subprocess
import sys


def run(*args: str) -> str:
    try:
        result = subprocess.run(
            [os.environ["NPU_SMI_BIN"], *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(os.environ.get("NPU_SMI_TIMEOUT_SECONDS", "20")),
        )
    except Exception as error:
        raise RuntimeError(f"npu-smi {' '.join(args)} could not be inspected: {error}") from error
    if result.returncode != 0:
        raise RuntimeError(
            f"npu-smi {' '.join(args)} failed with exit {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout


try:
    requested = {int(value) for value in os.environ["VISIBLE_DEVICES"].split(",")}
    mapping_output = run("info", "-m")
    info_output = run("info")
    mapped = set()
    physical_to_logical = {}
    for line in mapping_output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and all(part.isdigit() for part in parts[:3]):
            npu_id, chip_id, logical_id = map(int, parts[:3])
            mapped.add(logical_id)
            physical_to_logical[(npu_id, chip_id)] = logical_id
    if not requested <= mapped:
        missing = sorted(requested - mapped)
        raise RuntimeError(f"npu-smi mapping does not prove requested devices exist: {missing}")

    busy = set()
    in_process_section = False
    process_header_seen = False
    for raw_line in info_output.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) >= 2 and parts[1] == "Process id":
            in_process_section = True
            process_header_seen = True
            continue
        if not in_process_section or len(parts) < 2:
            continue
        left = parts[0].split()
        if len(left) >= 2 and left[0].isdigit() and left[1].isdigit() and parts[1].isdigit():
            physical = (int(left[0]), int(left[1]))
            if physical not in physical_to_logical:
                raise RuntimeError(f"process table references an unmapped device: {physical}")
            busy.add(physical_to_logical[physical])
    if not process_header_seen:
        raise RuntimeError("npu-smi output did not contain a process table")
    occupied = sorted(requested & busy)
    if occupied:
        raise RuntimeError(f"explicitly scoped Ascend devices have active processes: {occupied}")
except Exception as error:
    print(f"[goal-baseline] cannot prove explicit multi-card scope is idle: {error}", file=sys.stderr)
    sys.exit(2)
PY
}

validate_explicit_device_scope() {
  local visible_devices=$1
  local expected_count=${CHIP_COUNT:-1}
  local actual_count
  local unique_count

  if [[ ! "$expected_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid chip_count for device scoping: ${expected_count}" >&2
    return 2
  fi
  if [[ ! "$visible_devices" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Explicit Ascend device scope must be a comma-separated list of numeric IDs" >&2
    return 2
  fi
  actual_count=$(tr ',' '\n' <<< "$visible_devices" | wc -l)
  unique_count=$(tr ',' '\n' <<< "$visible_devices" | sort -u | wc -l)
  if [[ "$actual_count" -ne "$unique_count" ]]; then
    echo "Explicit Ascend device scope contains duplicate device IDs" >&2
    return 2
  fi
  if [[ "$actual_count" -ne "$expected_count" ]]; then
    echo "Explicit Ascend device scope cardinality ${actual_count} does not match chip_count ${expected_count}" >&2
    return 2
  fi
  if [[ "$expected_count" -gt 1 ]]; then
    verify_explicit_multicard_scope_idle "$visible_devices"
  fi
}

configure_single_card_ascend_device() {
  local start_attempt=${1:-1}
  local busy_exit_code=${RESOURCE_BUSY_EXIT_CODE:-75}
  local resolved_visible_devices=""
  local resolved_rt_visible_devices=""
  local preferred_device=""
  local selected_device_info=""
  local selected_device=""
  local selected_source=""
  local npu_smi_bin=""

  unset GOAL_BASELINE_DEVICE_SELECTION_REASON

  capture_initial_ascend_device_scope

  resolved_visible_devices=$(normalize_visible_devices "${GOAL_BASELINE_INITIAL_ASCEND_VISIBLE_DEVICES:-}" 2>/dev/null || true)
  resolved_rt_visible_devices=$(normalize_visible_devices "${GOAL_BASELINE_INITIAL_ASCEND_RT_VISIBLE_DEVICES:-}" 2>/dev/null || true)

  if [[ -z "$resolved_rt_visible_devices" && -n "$resolved_visible_devices" ]]; then
    set_ascend_visible_devices_scope "$resolved_visible_devices"
    echo "[goal-baseline] derived Ascend visible devices from ASCEND_VISIBLE_DEVICES: $ASCEND_VISIBLE_DEVICES"
  elif [[ -n "$resolved_rt_visible_devices" ]]; then
    set_ascend_visible_devices_scope "$resolved_rt_visible_devices"
  elif [[ "${GOAL_BASELINE_INITIAL_ASCEND_RT_VISIBLE_DEVICES_IS_SET:-0}" == "1" ]]; then
    set_ascend_visible_devices_scope ""
    echo "[goal-baseline] ignoring empty ASCEND_RT_VISIBLE_DEVICES from parent environment"
  else
    set_ascend_visible_devices_scope ""
  fi

  if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    validate_explicit_device_scope "$ASCEND_RT_VISIBLE_DEVICES" || return $?
    GOAL_BASELINE_DEVICE_SELECTION_REASON="explicit"
    export VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE="${VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE:-npu:0}"
    echo "[goal-baseline] using explicit Ascend visible devices: ${ASCEND_VISIBLE_DEVICES:-$ASCEND_RT_VISIBLE_DEVICES}"
    return 0
  fi

  if [[ "${CHIP_COUNT:-1}" -gt 1 ]]; then
    GOAL_BASELINE_DEVICE_SELECTION_REASON="unscoped-multicard"
    echo "chip_count=${CHIP_COUNT} requires an explicit ASCEND_RT_VISIBLE_DEVICES or ASCEND_VISIBLE_DEVICES scope" >&2
    return 2
  fi

  npu_smi_bin=$(resolve_npu_smi_bin 2>/dev/null || true)
  if [[ -n "$npu_smi_bin" ]]; then
    echo "[goal-baseline] using npu-smi for device selection: $npu_smi_bin"
  fi

  preferred_device=$(read_preferred_ascend_device 2>/dev/null || true)
  if [[ -n "$preferred_device" ]]; then
    echo "[goal-baseline] preferring previously selected Ascend device: $preferred_device"
  fi

  selected_device_info=$(select_ascend_device "$start_attempt" "$npu_smi_bin" "$preferred_device" 2>/dev/null || true)
  if [[ -n "$selected_device_info" ]]; then
    IFS=$'\t' read -r selected_device selected_source <<< "$selected_device_info"
    if [[ "$selected_device" == "__ALL_BUSY__" ]]; then
      GOAL_BASELINE_DEVICE_SELECTION_REASON="all-busy"
      set_ascend_visible_devices_scope ""
      unset VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE
      echo "[goal-baseline] all detected Ascend devices are currently busy: ${selected_source:-unknown}" >&2
      return "$busy_exit_code"
    fi
    if [[ -n "$selected_device" ]]; then
      case "$selected_source" in
        *+npu-smi-device-used*)
          echo "[goal-baseline] npu-smi could not inspect busy devices for the current user because DCMI reported 'device is used'; falling back to ${selected_source%%+*}" >&2
          ;;
        *+npu-smi-permission-limited*)
          echo "[goal-baseline] npu-smi device inspection appears permission-limited for the current user; falling back to ${selected_source%%+*}" >&2
          ;;
        *+npu-smi-exit-*)
          echo "[goal-baseline] npu-smi device inspection failed for the current user (${selected_source#*+npu-smi-}); falling back to ${selected_source%%+*}" >&2
          ;;
      esac
      GOAL_BASELINE_DEVICE_SELECTION_REASON="selected"
      set_ascend_visible_devices_scope "$selected_device"
      persist_preferred_ascend_device "$selected_device"
      export VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE="npu:0"
      echo "[goal-baseline] selected single-card Ascend device: ${ASCEND_VISIBLE_DEVICES:-$selected_device} (${selected_source:-auto})"
      return 0
    fi
  fi

  GOAL_BASELINE_DEVICE_SELECTION_REASON="unscoped"
  set_ascend_visible_devices_scope ""
  unset VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE
  echo "[goal-baseline] could not resolve a single-card Ascend device; running without device scoping" >&2
}

run_server_command() {
  (
    cd "$OFFICIAL_RUNTIME_CWD"
    source_ascend_runtime_env
    export VLLM_CACHE_ROOT="$OFFICIAL_VLLM_CACHE_ROOT"
    export HF_HOME="${HF_HOME:-/data/shared_datasets/vllm-hust-benchmark/huggingface}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
    if [[ -n "${OFFICIAL_CORE_VERSION:-}" ]]; then
      export VLLM_VERSION="$OFFICIAL_CORE_VERSION"
    fi
    if [[ -n "${OFFICIAL_RUNTIME_VLLM_BATCH_INVARIANT:-}" ]]; then
      export VLLM_BATCH_INVARIANT="$OFFICIAL_RUNTIME_VLLM_BATCH_INVARIANT"
    else
      unset VLLM_BATCH_INVARIANT
    fi
    export PATH="$GOAL_BASELINE_ENV_PREFIX/bin:$PATH"
    PYTHONUNBUFFERED=1 \
      PYTHONPATH="$OFFICIAL_RUNTIME_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
      "$OFFICIAL_RUNTIME_PYTHON" -u -m vllm.entrypoints.openai.api_server $SERVER_ARGS
  )
}

run_client_command() {
  if [[ -n "${TRACE_TARGET_ID:-}" ]]; then
    run_in_official_runtime "$REPO_ROOT/src:$OFFICIAL_RUNTIME_PYTHONPATH" \
      python -m vllm_hust_benchmark.trace_replay replay \
      "$TRACE_TARGET_ID" \
      --trace-path "$TRACE_ASSET_PATH" \
      --model "$RUNTIME_MODEL" \
      --base-url "http://${CLIENT_HOST}:${CLIENT_PORT}" \
      --endpoint "$TRACE_ENDPOINT" \
      --max-model-len "$TRACE_MAX_MODEL_LEN" \
      --max-requests "$TRACE_MAX_REQUESTS" \
      --max-concurrency "$TRACE_MAX_CONCURRENCY" \
      --timeout-s "$TRACE_TIMEOUT_S" \
      --overflow-policy "$TRACE_OVERFLOW_POLICY" \
      --time-scale "$TRACE_TIME_SCALE" \
      --max-interarrival-s "$TRACE_MAX_INTERARRIVAL_S" \
      --output "$TRACE_DETAIL_RESULT_FILE" \
      --summary-output "$RAW_RESULT_FILE"
    return $?
  fi

  case "$BENCHMARK_TYPE" in
    serve)
      VLLM_HUST_IMMUTABLE_INPUT_ATTESTATION_FILE="$IMMUTABLE_INPUT_ATTESTATION_FILE" \
      VLLM_HUST_IMMUTABLE_INPUT_METADATA="$IMMUTABLE_INPUT_METADATA" \
      run_in_official_runtime "$OFFICIAL_RUNTIME_PYTHONPATH" \
        python "$VLLM_CLI_COMPAT" bench serve \
        --save-result \
        --result-dir "$RESULT_DIR" \
        --result-filename "$(basename "$RAW_RESULT_FILE")" \
        $CLIENT_ARGS
      ;;
    throughput|latency)
      prepare_offline_benchmark_runtime || return $?

      run_offline_client_command "$CLIENT_ARGS"
      ;;
    *)
      echo "Unsupported benchmark type for official baseline runner: $BENCHMARK_TYPE" >&2
      return 2
      ;;
  esac
}

run_offline_client_command() {
  local effective_client_args=${1:-$CLIENT_ARGS}

  VLLM_HUST_IMMUTABLE_INPUT_ATTESTATION_FILE="$IMMUTABLE_INPUT_ATTESTATION_FILE" \
  VLLM_HUST_IMMUTABLE_INPUT_METADATA="$IMMUTABLE_INPUT_METADATA" \
  run_in_official_runtime "$OFFICIAL_RUNTIME_PYTHONPATH" \
    python "$VLLM_CLI_COMPAT" bench "$BENCHMARK_TYPE" \
    --output-json "$RAW_RESULT_FILE" \
    $effective_client_args
}

prepare_offline_benchmark_runtime() {
  local selection_status=0
  local runtime_ready_status=0

  if wait_for_single_card_ascend_device; then
    selection_status=0
  else
    selection_status=$?
  fi

  if [[ "$selection_status" -ne 0 ]]; then
    if [[ "$selection_status" -eq "$RESOURCE_BUSY_EXIT_CODE" && "${GOAL_BASELINE_DEVICE_SELECTION_REASON:-}" == "all-busy" ]]; then
      echo "[goal-baseline] All detected Ascend devices remained busy after ${DEVICE_SELECTION_RETRIES} selection attempt(s)" >&2
    fi
    return "$selection_status"
  fi

  echo "[goal-baseline] Ascend visible devices: ${ASCEND_VISIBLE_DEVICES:-<unset>} (rt=${ASCEND_RT_VISIBLE_DEVICES:-<unset>})"

  if wait_for_ascend_runtime_ready; then
    return 0
  fi

  runtime_ready_status=$?
  echo "[goal-baseline] Ascend runtime did not become ready after ${ASCEND_RUNTIME_READY_TIMEOUT_SECONDS}s" >&2
  return "$runtime_ready_status"
}

resolve_same_spec() {
  local resolve_args=(
    "$HOST_PYTHON_BIN" -m vllm_hust_benchmark.same_spec resolve
    --spec-file "$SPEC_FILE"
    --output-file "$SAME_SPEC_FILE"
    --runtime-model "$RUNTIME_MODEL"
  )

  if [[ "$BENCHMARK_TYPE" == "serve" ]]; then
    resolve_args+=(
      --server-port "$OFFICIAL_SERVER_PORT"
      --client-port "$OFFICIAL_CLIENT_PORT"
    )

    if [[ -n "$OFFICIAL_SERVER_HOST" ]]; then
      resolve_args+=(--server-host "$OFFICIAL_SERVER_HOST")
    fi
    if [[ -n "$OFFICIAL_CLIENT_HOST" ]]; then
      resolve_args+=(--client-host "$OFFICIAL_CLIENT_HOST")
    fi
  fi

  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "${resolve_args[@]}"
}

resolve_scenario_benchmark_type() {
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    env SCENARIO_NAME="$SCENARIO" \
    "$HOST_PYTHON_BIN" -c "import os; from vllm_hust_benchmark.registry import get_scenario; print(get_scenario(os.environ['SCENARIO_NAME']).benchmark_type)"
}

append_export_arg_from_spec() {
  local flag=$1
  local jq_filter=$2
  local value

  value=$(jq -r "$jq_filter // empty" "$SPEC_FILE")
  if [[ -n "$value" ]] && [[ "$value" != "null" ]]; then
    EXPORT_ARGS+=("$flag" "$value")
  fi
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
  local expected_commit
  local actual_commit
  local tracked_status

  expected_commit=$(git -C "$source_repo" rev-parse --verify "${ref_name}^{commit}") || {
    echo "Cannot resolve official source ref ${ref_name} in ${source_repo}" >&2
    return 2
  }
  if [[ ! -d "$target_dir/.git" && ! -f "$target_dir/.git" ]]; then
    git -C "$source_repo" worktree add --detach "$target_dir" "$expected_commit"
  fi

  actual_commit=$(git -C "$target_dir" rev-parse --verify HEAD 2>/dev/null) || {
    echo "Official worktree is not a Git worktree: ${target_dir}" >&2
    return 2
  }
  if [[ "$actual_commit" != "$expected_commit" ]]; then
    echo "Official worktree HEAD mismatch for ${target_dir}: expected ${expected_commit}, got ${actual_commit}" >&2
    return 2
  fi
  tracked_status=$(git -C "$target_dir" status --porcelain --untracked-files=no)
  if [[ -n "$tracked_status" ]]; then
    echo "Official worktree has tracked modifications: ${target_dir}" >&2
    return 2
  fi
  echo "[goal-baseline] verified source ${source_repo}@${ref_name}: ${actual_commit}"
}

json2args() {
  local json_string=$1
  echo "$json_string" | jq -r '
    to_entries |
    map(
      if (.value == null or .value == false or (.value | tostring) == "") then
        empty
      elif .value == true then
        "--" + (.key | gsub("_"; "-"))
      else
        "--" + (.key | gsub("_"; "-")) + " " + (.value | tostring)
      end
    ) |
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
    if [[ -f "$target_file" ]] && "$HOST_PYTHON_BIN" -m json.tool "$target_file" >/dev/null 2>&1; then
      return 0
    fi

    rm -f "$tmp_file"
    download_file "$url" "$tmp_file"
    "$HOST_PYTHON_BIN" -m json.tool "$tmp_file" >/dev/null
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
      sharegpt_target="$OFFICIAL_BENCHMARK_DATASET_ROOT/$dataset_path"
      if [[ -f "$sharegpt_target" ]] && "$HOST_PYTHON_BIN" -m json.tool "$sharegpt_target" >/dev/null 2>&1; then
        return 0
      fi
      echo "[goal-baseline] downloading ShareGPT benchmark dataset to $sharegpt_target"
      download_json_file_atomic "$OFFICIAL_SHAREGPT_DATASET_URL" "$sharegpt_target"
      ;;
    benchmarks/*)
      if [[ ! -f "$OFFICIAL_VLLM_WORKTREE/$dataset_path" ]]; then
        echo "benchmark dataset path not found in official vllm worktree: $OFFICIAL_VLLM_WORKTREE/$dataset_path" >&2
        return 2
      fi
      ;;
  esac
}

verify_immutable_input_contract() {
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    SPEC_FILE="$SPEC_FILE" \
    BENCHMARK_REPO="$REPO_ROOT" \
    OFFICIAL_VLLM_WORKTREE="$OFFICIAL_VLLM_WORKTREE" \
    OFFICIAL_BENCHMARK_DATASET_ROOT="$OFFICIAL_BENCHMARK_DATASET_ROOT" \
    OFFICIAL_SHAREGPT_DATASET_URL="$OFFICIAL_SHAREGPT_DATASET_URL" \
    TRACE_ASSET_PATH="${TRACE_ASSET_PATH:-}" \
    "$HOST_PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from vllm_hust_benchmark.immutable_input_attestation import (
    build_metadata,
    verify_data_contract,
)

spec = json.loads(Path(os.environ["SPEC_FILE"]).read_text(encoding="utf-8"))
metadata = build_metadata(spec)
trace_path = os.environ.get("TRACE_ASSET_PATH")
verify_data_contract(
    metadata["data_identity"],
    benchmark_repo=Path(os.environ["BENCHMARK_REPO"]),
    vllm_worktree=Path(os.environ["OFFICIAL_VLLM_WORKTREE"]),
    dataset_root=Path(os.environ["OFFICIAL_BENCHMARK_DATASET_ROOT"]),
    sharegpt_url=os.environ["OFFICIAL_SHAREGPT_DATASET_URL"],
    trace_asset_path=Path(trace_path) if trace_path else None,
)
print(json.dumps(metadata, separators=(",", ":"), sort_keys=True))
PY
}

finalize_trace_immutable_input_attestation() {
  if [[ -z "${TRACE_TARGET_ID:-}" ]]; then
    return 0
  fi
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    IMMUTABLE_INPUT_ATTESTATION_FILE="$IMMUTABLE_INPUT_ATTESTATION_FILE" \
    IMMUTABLE_INPUT_METADATA="$IMMUTABLE_INPUT_METADATA" \
    RAW_RESULT_FILE="$RAW_RESULT_FILE" \
    "$HOST_PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from vllm_hust_benchmark.immutable_input_attestation import write_trace_attestation

write_trace_attestation(
    Path(os.environ["IMMUTABLE_INPUT_ATTESTATION_FILE"]),
    json.loads(os.environ["IMMUTABLE_INPUT_METADATA"]),
    json.loads(Path(os.environ["RAW_RESULT_FILE"]).read_text(encoding="utf-8")),
)
PY
}

normalized_server_parameters_json() {
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    SAME_SPEC_FILE="$SAME_SPEC_FILE" \
    "$HOST_PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from vllm_hust_benchmark.official_runtime_inputs import normalize_server_parameters

payload = json.loads(Path(os.environ["SAME_SPEC_FILE"]).read_text(encoding="utf-8"))
normalized = normalize_server_parameters(payload["resolved_server_parameters"])
print(
    json.dumps(
    normalized,
        separators=(",", ":"),
        ensure_ascii=True,
    )
)
PY
}

normalized_client_parameters_json() {
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    SAME_SPEC_FILE="$SAME_SPEC_FILE" \
    BENCHMARK_TYPE="$BENCHMARK_TYPE" \
    CLIENT_READY_CHECK_TIMEOUT_SECONDS="$CLIENT_READY_CHECK_TIMEOUT_SECONDS" \
    OFFICIAL_VLLM_WORKTREE="$OFFICIAL_VLLM_WORKTREE" \
    BENCHMARK_REPO="$REPO_ROOT" \
    OFFICIAL_BENCHMARK_DATASET_ROOT="$OFFICIAL_BENCHMARK_DATASET_ROOT" \
    "$HOST_PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from vllm_hust_benchmark.official_runtime_inputs import normalize_client_parameters
from vllm_hust_benchmark.official_runtime_inputs import (
    normalize_offline_benchmark_parameters,
)

payload = json.loads(Path(os.environ["SAME_SPEC_FILE"]).read_text(encoding="utf-8"))
ready_timeout = int(os.environ.get("CLIENT_READY_CHECK_TIMEOUT_SECONDS") or 0)
benchmark_type = os.environ["BENCHMARK_TYPE"]
normalizer = normalize_client_parameters
normalizer_args = (payload["resolved_client_parameters"],)
if benchmark_type in {"throughput", "latency"}:
    normalizer = normalize_offline_benchmark_parameters
    normalizer_args = (
        payload["resolved_client_parameters"],
        payload["resolved_server_parameters"],
    )
print(
    json.dumps(
        normalizer(
            *normalizer_args,
            benchmark_type=benchmark_type,
            ready_check_timeout_sec=ready_timeout,
            vllm_worktree=os.environ.get("OFFICIAL_VLLM_WORKTREE"),
            benchmark_repo=os.environ.get("BENCHMARK_REPO"),
            dataset_cache_root=os.environ.get("OFFICIAL_BENCHMARK_DATASET_ROOT"),
        ),
        separators=(",", ":"),
        ensure_ascii=True,
    )
)
PY
}

resolve_runtime_model() {
  local runtime_model_candidate=""
  local complete_runtime_model=""

  if [[ ! "${MODEL_REVISION:-}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Official model requires an exact 40-character revision" >&2
    return 2
  fi

  if [[ -n "$OFFICIAL_MODEL_PATH" ]]; then
    runtime_model_candidate=$(realpath "$OFFICIAL_MODEL_PATH")
    if [[ -n "${TRACE_TARGET_ID:-}" ]]; then
      verify_runtime_model_artifact "$runtime_model_candidate"
    elif [[ "$(basename "$runtime_model_candidate")" != "$MODEL_REVISION" ]] || \
         [[ "$(basename "$(dirname "$runtime_model_candidate")")" != "snapshots" ]]; then
      echo "OFFICIAL_MODEL_PATH is not the exact model snapshot ${MODEL_REVISION}: ${runtime_model_candidate}" >&2
      return 2
    fi
    printf '%s\n' "$runtime_model_candidate"
    return 0
  fi

  runtime_model_candidate=$(run_in_official_runtime "$OFFICIAL_RUNTIME_PYTHONPATH" \
    env MODEL_ID="$MODEL" MODEL_REVISION="${MODEL_REVISION:-}" \
    python -c "import os; from huggingface_hub import snapshot_download; print(snapshot_download(os.environ['MODEL_ID'], revision=os.environ['MODEL_REVISION'], local_files_only=True))" \
    2>/dev/null) || return 1

  if [[ -n "${TRACE_TARGET_ID:-}" ]]; then
    verify_runtime_model_artifact "$runtime_model_candidate"
    printf '%s\n' "$runtime_model_candidate"
    return 0
  fi
  complete_runtime_model=$(resolve_complete_local_runtime_model_candidate "$runtime_model_candidate") || return 2
  if [[ "$(basename "$complete_runtime_model")" != "$MODEL_REVISION" ]] || \
     [[ "$(basename "$(dirname "$complete_runtime_model")")" != "snapshots" ]]; then
    echo "Resolved model is not the exact snapshot ${MODEL_REVISION}: ${complete_runtime_model}" >&2
    return 2
  fi
  printf '%s\n' "$complete_runtime_model"
}

verify_runtime_model_artifact() {
  local model_path=$1
  local provenance_file="$RESULT_DIR/model_artifact_provenance.json"

  if [[ -z "${MODEL_REVISION:-}" ]]; then
    echo "Trace targets require a pinned server_parameters.revision" >&2
    return 2
  fi
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$HOST_PYTHON_BIN" - "$model_path" "$MODEL_REVISION" "$provenance_file" <<'PY'
import json
from pathlib import Path
import sys

from vllm_hust_benchmark.model_artifact import verify_local_hf_model

payload = verify_local_hf_model(sys.argv[1], sys.argv[2])
output = Path(sys.argv[3])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[goal-baseline] verified model artifact digest: {payload['model_artifact_digest']}", file=sys.stderr)
PY
}

verify_trace_runtime_packages() {
  local provenance_file="$RESULT_DIR/runtime_package_provenance.json"
  local expected_packages
  local expected_image
  local expected_image_digest
  local expected_environment

  expected_packages=$(jq -c '.baseline_target.runtime_packages // {}' "$SPEC_FILE")
  expected_image=$(jq -r '.baseline_target.runtime_image // empty' "$SPEC_FILE")
  expected_image_digest=$(jq -r '.baseline_target.runtime_image_digest // empty' "$SPEC_FILE")
  expected_environment=$(jq -c '.baseline_target.runtime_environment // {}' "$SPEC_FILE")
  if [[ "$expected_packages" == "{}" || -z "$expected_image" || -z "$expected_image_digest" ]]; then
    echo "Production-trace targets require pinned runtime packages and official image digest" >&2
    return 2
  fi
  if [[ "$expected_image" != *@"$expected_image_digest" ]]; then
    echo "Production-trace runtime image does not match its pinned digest" >&2
    return 2
  fi
  if [[ -z "$OFFICIAL_RUNTIME_IMAGE" || "$OFFICIAL_RUNTIME_IMAGE" != "$expected_image" ]]; then
    echo "OFFICIAL_RUNTIME_IMAGE must exactly match the production-trace image digest: $expected_image" >&2
    return 2
  fi
  EXPECTED_PACKAGES="$expected_packages" EXPECTED_IMAGE="$expected_image" \
  EXPECTED_IMAGE_DIGEST="$expected_image_digest" PROVENANCE_FILE="$provenance_file" \
  EXPECTED_ENVIRONMENT="$expected_environment" \
  run_in_official_runtime \
    "$REPO_ROOT/src:$OFFICIAL_RUNTIME_PYTHONPATH" python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path

expected = json.loads(os.environ["EXPECTED_PACKAGES"])
expected_environment = json.loads(os.environ["EXPECTED_ENVIRONMENT"])
actual = {}
for package, expected_version in expected.items():
    try:
        actual_version = version(package)
    except PackageNotFoundError as error:
        raise RuntimeError(f"required runtime package is missing: {package}") from error
    if actual_version != expected_version:
        raise RuntimeError(
            f"runtime package mismatch for {package}: expected {expected_version}, got {actual_version}"
        )
    actual[package] = actual_version

payload = {
    "runtime_packages": actual,
    "runtime_image": os.environ["EXPECTED_IMAGE"],
    "runtime_image_digest": os.environ["EXPECTED_IMAGE_DIGEST"],
    "runtime_environment": {
        key: os.environ.get(key) for key in expected_environment
    },
}
if payload["runtime_environment"] != expected_environment:
    raise RuntimeError(
        "runtime environment mismatch: "
        f"expected {expected_environment}, got {payload['runtime_environment']}"
    )
Path(os.environ["PROVENANCE_FILE"]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(f"[goal-baseline] verified trace runtime packages: {actual}")
PY
}

prepare_trace_startup_evidence() {
  TRACE_DRY_RUN_PLAN_FILE="$RESULT_DIR/trace_replay_plan.json"
  STARTUP_EVIDENCE_FILE="$RESULT_DIR/startup_evidence.json"
  STARTUP_INSTANCE_ID=$(
    "$HOST_PYTHON_BIN" -c 'import uuid; print(uuid.uuid4())'
  )
  run_in_official_runtime "$REPO_ROOT/src:$OFFICIAL_RUNTIME_PYTHONPATH" \
    python -m vllm_hust_benchmark.trace_replay replay \
    "$TRACE_TARGET_ID" \
    --trace-path "$TRACE_ASSET_PATH" \
    --model "$RUNTIME_MODEL" \
    --max-model-len "$TRACE_MAX_MODEL_LEN" \
    --max-requests "$TRACE_MAX_REQUESTS" \
    --max-concurrency "$TRACE_MAX_CONCURRENCY" \
    --timeout-s "$TRACE_TIMEOUT_S" \
    --overflow-policy "$TRACE_OVERFLOW_POLICY" \
    --time-scale "$TRACE_TIME_SCALE" \
    --max-interarrival-s "$TRACE_MAX_INTERARRIVAL_S" \
    --dry-run > "$TRACE_DRY_RUN_PLAN_FILE"

  RUN_ID="$RUN_ID" STARTUP_INSTANCE_ID="$STARTUP_INSTANCE_ID" \
  TRACE_TARGET_ID="$TRACE_TARGET_ID" TRACE_DRY_RUN_PLAN_FILE="$TRACE_DRY_RUN_PLAN_FILE" \
  MODEL_PROVENANCE_FILE="$RESULT_DIR/model_artifact_provenance.json" \
  RUNTIME_PACKAGE_PROVENANCE_FILE="$RESULT_DIR/runtime_package_provenance.json" \
  OFFICIAL_CORE_SOURCE_COMMIT="$OFFICIAL_CORE_SOURCE_COMMIT" \
  OFFICIAL_BACKEND_SOURCE_COMMIT="$OFFICIAL_BACKEND_SOURCE_COMMIT" \
  STARTUP_EVIDENCE_FILE="$STARTUP_EVIDENCE_FILE" "$HOST_PYTHON_BIN" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

plan = json.loads(Path(os.environ["TRACE_DRY_RUN_PLAN_FILE"]).read_text(encoding="utf-8"))
model = json.loads(Path(os.environ["MODEL_PROVENANCE_FILE"]).read_text(encoding="utf-8"))
runtime = json.loads(
    Path(os.environ["RUNTIME_PACKAGE_PROVENANCE_FILE"]).read_text(encoding="utf-8")
)
payload = {
    "schema_version": "official-trace-startup-evidence/v1",
    "startup_instance_id": os.environ["STARTUP_INSTANCE_ID"],
    "run_id": os.environ["RUN_ID"],
    "trace_target_id": os.environ["TRACE_TARGET_ID"],
    "started_at": datetime.now(timezone.utc).isoformat(),
    "engine_source_commit": os.environ["OFFICIAL_CORE_SOURCE_COMMIT"],
    "plugin_source_commit": os.environ["OFFICIAL_BACKEND_SOURCE_COMMIT"],
    "model_artifact_digest": model["model_artifact_digest"],
    "runtime_packages": runtime["runtime_packages"],
    "runtime_image": runtime["runtime_image"],
    "runtime_image_digest": runtime["runtime_image_digest"],
    "runtime_environment": runtime["runtime_environment"],
    "trace_asset_sha256": plan["cohort"]["setting_signature_payload"]["trace_asset_sha256"],
    "cohort_setting_signature": plan["cohort_setting_signature"],
    "dry_run_plan": os.environ["TRACE_DRY_RUN_PLAN_FILE"],
}
Path(os.environ["STARTUP_EVIDENCE_FILE"]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
}

finalize_trace_startup_evidence() {
  STARTUP_EVIDENCE_FILE="$STARTUP_EVIDENCE_FILE" RAW_RESULT_FILE="$RAW_RESULT_FILE" \
  TRACE_DETAIL_RESULT_FILE="$TRACE_DETAIL_RESULT_FILE" "$HOST_PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

evidence_path = Path(os.environ["STARTUP_EVIDENCE_FILE"])
payload = json.loads(evidence_path.read_text(encoding="utf-8"))
raw = Path(os.environ["RAW_RESULT_FILE"])
detail = Path(os.environ["TRACE_DETAIL_RESULT_FILE"])
if not raw.is_file() or not detail.is_file():
    raise FileNotFoundError("trace run did not produce both raw and detail results")
payload["finished_at"] = datetime.now(timezone.utc).isoformat()
payload["result_hashes"] = {"raw_sha256": sha256(raw), "detail_sha256": sha256(detail)}
evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

path_has_matching_file() {
  local path=$1
  shift
  local pattern
  local candidate
  local had_nullglob=0

  if shopt -q nullglob; then
    had_nullglob=1
  fi
  shopt -s nullglob

  for pattern in "$@"; do
    for candidate in "$path"/$pattern; do
      if [[ -f "$candidate" ]]; then
        if [[ "$had_nullglob" == "0" ]]; then
          shopt -u nullglob
        fi
        return 0
      fi
    done
  done

  if [[ "$had_nullglob" == "0" ]]; then
    shopt -u nullglob
  fi
  return 1
}

path_has_complete_indexed_weights() {
  local runtime_model_candidate=$1
  local index_file

  for index_file in \
    "$runtime_model_candidate/model.safetensors.index.json" \
    "$runtime_model_candidate/pytorch_model.bin.index.json"; do
    [[ -f "$index_file" ]] || continue

    if MODEL_DIR="$runtime_model_candidate" INDEX_FILE="$index_file" "$OFFICIAL_RUNTIME_PYTHON" - <<'PY' >/dev/null 2>&1
import json
import os

model_dir = os.environ["MODEL_DIR"]
index_file = os.environ["INDEX_FILE"]

with open(index_file, encoding="utf-8") as fh:
    payload = json.load(fh)

weight_map = payload.get("weight_map") or {}
if not weight_map:
    raise SystemExit(1)

missing = [
    filename for filename in sorted(set(weight_map.values()))
    if not os.path.isfile(os.path.join(model_dir, filename))
]

raise SystemExit(0 if not missing else 1)
PY
    then
      return 0
    fi
  done

  return 1
}

local_runtime_model_has_required_artifacts() {
  local runtime_model_candidate=$1

  [[ -d "$runtime_model_candidate" ]] || return 1

  path_has_matching_file "$runtime_model_candidate" "config.json" || return 1
  path_has_matching_file \
    "$runtime_model_candidate" \
    "tokenizer.json" \
    "tokenizer.model" \
    "spiece.model" \
    "sentencepiece.bpe.model" \
    "vocab.json" \
    "vocab.txt" || return 1
  if path_has_matching_file \
    "$runtime_model_candidate" \
    "*.safetensors" \
    "*.bin" \
    "*.pt" \
    "*.pth"; then
    return 0
  fi

  path_has_complete_indexed_weights "$runtime_model_candidate"
}

resolve_complete_local_runtime_model_candidate() {
  local runtime_model_candidate=$1
  local snapshots_dir=""
  local sibling_candidate
  local had_nullglob=0

  if local_runtime_model_has_required_artifacts "$runtime_model_candidate"; then
    printf '%s\n' "$runtime_model_candidate"
    return 0
  fi

  if [[ "$(basename "$(dirname "$runtime_model_candidate")")" != "snapshots" ]]; then
    return 1
  fi

  snapshots_dir=$(dirname "$runtime_model_candidate")
  if shopt -q nullglob; then
    had_nullglob=1
  fi
  shopt -s nullglob

  for sibling_candidate in "$snapshots_dir"/*; do
    [[ "$sibling_candidate" == "$runtime_model_candidate" ]] && continue
    [[ -d "$sibling_candidate" ]] || continue

    if local_runtime_model_has_required_artifacts "$sibling_candidate"; then
      if [[ "$had_nullglob" == "0" ]]; then
        shopt -u nullglob
      fi
      printf '%s\n' "$sibling_candidate"
      return 0
    fi
  done

  if [[ "$had_nullglob" == "0" ]]; then
    shopt -u nullglob
  fi
  return 1
}

normalize_engine_version() {
  local version=${1:-}

  version=$(printf '%s' "$version" | tr -d '[:space:]')
  case "$version" in
    ""|unknown|Unknown|not-installed|N/A|n/a|dev)
      return 1
      ;;
  esac
  version=${version#v}
  version=${version#V}

  if [[ "$version" =~ ^[0-9]+(\.[0-9]+){1,2}([A-Za-z0-9._-]+)?$ ]]; then
    printf '%s' "$version"
    return 0
  fi

  return 1
}

is_valid_engine_version() {
  normalize_engine_version "$1" >/dev/null
}

detect_official_core_version() {
  local raw_output=""
  local detected=""
  local fallback=""

  raw_output=$(run_in_official_runtime_python "$OFFICIAL_RUNTIME_PYTHONPATH" <<'PY'
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

  if detected=$(normalize_engine_version "$detected"); then
    printf '%s' "$detected"
    return 0
  fi

  fallback=$(git -C "$OFFICIAL_VLLM_WORKTREE" describe --tags --always HEAD 2>/dev/null || true)
  if fallback=$(normalize_engine_version "$fallback"); then
    printf '%s' "$fallback"
    return 0
  fi

  printf '%s' "unknown"
}

detect_official_backend_version() {
  local raw_output=""
  local detected=""
  local fallback=""

  raw_output=$(run_in_official_runtime_python "$OFFICIAL_RUNTIME_PYTHONPATH" <<'PY'
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

  if detected=$(normalize_engine_version "$detected"); then
    printf '%s' "$detected"
    return 0
  fi

  fallback=$(git -C "$OFFICIAL_VLLM_ASCEND_WORKTREE" describe --tags --always HEAD 2>/dev/null || true)
  if fallback=$(normalize_engine_version "$fallback"); then
    printf '%s' "$fallback"
    return 0
  fi

  printf '%s' "unknown"
}

server_log_indicates_resource_busy() {
  local log_file=$1

  [[ -f "$log_file" ]] || return 1

  grep -Eq "DrvMngGetConsoleLogLevel failed|dcmi model initialized failed|ret is -8020|drvRet=87|drvRetCode=87|ErrCode=507899|error code is 507899|rtGetDeviceCount|Can't get ascend_hal device count|driver error:internal error|Resource_Busy\(EL0005\)|The resources are busy" "$log_file"
}

server_log_indicates_fatal_startup_error() {
  local log_file=$1

  [[ -f "$log_file" ]] || return 1

  grep -Eq "EngineCore failed to start|Engine core initialization failed|Worker failed with error|aclnn[A-Za-z0-9_]+ or aclnn[A-Za-z0-9_]+GetWorkspaceSize not in libopapi" "$log_file"
}

wait_for_ascend_runtime_ready() {
  local max_attempts
  max_attempts=$(((ASCEND_RUNTIME_READY_TIMEOUT_SECONDS + ASCEND_RUNTIME_READY_POLL_SECONDS - 1) / ASCEND_RUNTIME_READY_POLL_SECONDS))
  if (( max_attempts < 1 )); then
    max_attempts=1
  fi

  for runtime_attempt in $(seq 1 "$max_attempts"); do
    if run_in_official_runtime_python "$OFFICIAL_RUNTIME_PYTHONPATH" <<'PY' >"$RUNTIME_READY_LOG" 2>&1
import torch_npu

torch_npu.npu.get_soc_version()
PY
    then
      return 0
    fi

    cat "$RUNTIME_READY_LOG" >&2

    if [[ "$runtime_attempt" -eq "$max_attempts" ]]; then
      if server_log_indicates_resource_busy "$RUNTIME_READY_LOG"; then
        return "$RESOURCE_BUSY_EXIT_CODE"
      fi
      return 1
    fi

    echo "[goal-baseline] Ascend runtime not ready yet; waiting ${ASCEND_RUNTIME_READY_POLL_SECONDS}s before retrying device initialization (${runtime_attempt}/${max_attempts})" >&2
    sleep "$ASCEND_RUNTIME_READY_POLL_SECONDS"
  done
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
  local timeout_sec=$READY_TIMEOUT_SECONDS
  local status_interval_sec=${READY_STATUS_INTERVAL_SECONDS:-30}
  local next_status_at=0

  if (( status_interval_sec <= 0 )); then
    status_interval_sec=30
  fi

  while (( waited < timeout_sec )); do
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Official baseline server exited before becoming ready at ${host}:${port}" >&2
      if [[ -n "${SERVER_STDOUT_LOG:-}" && -f "$SERVER_STDOUT_LOG" ]]; then
        tail -n 40 "$SERVER_STDOUT_LOG" >&2 || true
        if server_log_indicates_resource_busy "$SERVER_STDOUT_LOG"; then
          return "$RESOURCE_BUSY_EXIT_CODE"
        fi
      fi
      return 1
    fi

    if probe_server_ready "$host" "$port"; then
      if (( waited > 0 )); then
        echo "[goal-baseline] official baseline server became ready after ${waited}s"
      fi
      return 0
    fi

    if [[ -n "${SERVER_STDOUT_LOG:-}" ]] && server_log_indicates_fatal_startup_error "$SERVER_STDOUT_LOG"; then
      echo "Official baseline server reported a fatal startup error at ${host}:${port}" >&2
      tail -n 40 "$SERVER_STDOUT_LOG" >&2 || true
      return 1
    fi

    if (( waited >= next_status_at )); then
      echo "[goal-baseline] waiting for official baseline server at ${host}:${port} (${waited}s/${timeout_sec}s)" >&2
      next_status_at=$((waited + status_interval_sec))
    fi

    sleep 1
    ((waited += 1))
  done

  echo "Timed out waiting for official baseline server at ${host}:${port}" >&2
  if [[ -n "${SERVER_STDOUT_LOG:-}" && -f "$SERVER_STDOUT_LOG" ]]; then
    tail -n 40 "$SERVER_STDOUT_LOG" >&2 || true
    if server_log_indicates_resource_busy "$SERVER_STDOUT_LOG"; then
      return "$RESOURCE_BUSY_EXIT_CODE"
    fi
  fi
  return 1
}

wait_for_single_card_ascend_device() {
  local max_attempts=${DEVICE_SELECTION_RETRIES:-1}
  local retry_delay=${DEVICE_SELECTION_RETRY_DELAY_SECONDS:-0}
  local selection_attempt
  local selection_status=0

  if (( max_attempts < 1 )); then
    max_attempts=1
  fi

  for selection_attempt in $(seq 1 "$max_attempts"); do
    if configure_single_card_ascend_device "$selection_attempt"; then
      return 0
    else
      selection_status=$?
    fi

    if [[ "$selection_status" -ne "$RESOURCE_BUSY_EXIT_CODE" ]]; then
      return "$selection_status"
    fi

    if [[ "$selection_attempt" -ge "$max_attempts" ]]; then
      return "$selection_status"
    fi

    if [[ "${GOAL_BASELINE_DEVICE_SELECTION_REASON:-}" == "all-busy" ]]; then
      echo "[goal-baseline] All detected Ascend devices are busy; waiting ${retry_delay}s for an idle card (attempt ${selection_attempt}/${max_attempts})" >&2
    else
      echo "[goal-baseline] No idle Ascend device is currently available; retrying device selection in ${retry_delay}s (attempt ${selection_attempt}/${max_attempts})" >&2
    fi
    sleep "$retry_delay"
  done

  return "$selection_status"
}

stop_peak_hbm_sampler() {
  if [[ -n "${PEAK_HBM_SAMPLER_PID:-}" ]]; then
    kill -TERM "$PEAK_HBM_SAMPLER_PID" >/dev/null 2>&1 || true
    wait "$PEAK_HBM_SAMPLER_PID" >/dev/null 2>&1 || true
    PEAK_HBM_SAMPLER_PID=""
  fi
}

start_peak_hbm_sampler() {
  local device_scope=${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-}}
  stop_peak_hbm_sampler
  if [[ -z "$device_scope" ]]; then
    echo "Explicit Ascend device scope is required for peak HBM evidence" >&2
    exit 2
  fi
  PEAK_HBM_EVIDENCE_FILE="$RESULT_DIR/peak_hbm_evidence.json"
  "$HOST_PYTHON_BIN" "$REPO_ROOT/scripts/sample_ascend_peak_hbm.py" \
    --devices "$device_scope" --output "$PEAK_HBM_EVIDENCE_FILE" &
  PEAK_HBM_SAMPLER_PID=$!
}

kill_server() {
  stop_peak_hbm_sampler
  cleanup_managed_server || true
}

trap kill_server EXIT

OFFICIAL_VLLM_REF=${OFFICIAL_VLLM_REF:-$(jq -r '.baseline_target.vllm_ref // "v0.18.0"' "$SPEC_FILE")}
OFFICIAL_VLLM_ASCEND_REF=${OFFICIAL_VLLM_ASCEND_REF:-$(jq -r '.baseline_target.vllm_ascend_ref // "v0.18.0"' "$SPEC_FILE")}
ensure_worktree "$OFFICIAL_VLLM_REPO" "$OFFICIAL_VLLM_WORKTREE" "$OFFICIAL_VLLM_REF"
ensure_worktree "$OFFICIAL_VLLM_ASCEND_REPO" "$OFFICIAL_VLLM_ASCEND_WORKTREE" "$OFFICIAL_VLLM_ASCEND_REF"

OFFICIAL_RUNTIME_PYTHONPATH="$OFFICIAL_VLLM_ASCEND_WORKTREE:$OFFICIAL_VLLM_WORKTREE"

mkdir -p "$RESULT_DIR"
mkdir -p "$OFFICIAL_VLLM_CACHE_ROOT"
RUNNER_STATE_DIR="$RESULT_DIR/.runtime-state"
RUNNER_LOCK_FILE="$RUNNER_STATE_DIR/runner.lock"
MANAGED_SERVER_PORT_FILE="$RUNNER_STATE_DIR/server.port"
MANAGED_SERVER_WRAPPER_PID_FILE="$RUNNER_STATE_DIR/server.wrapper.pid"
MANAGED_SERVER_LISTENER_PIDS_FILE="$RUNNER_STATE_DIR/server.listener.pids"
SERVER_STDOUT_LOG="$RESULT_DIR/server.stdout.log"
RUNTIME_READY_LOG="$RESULT_DIR/runtime-ready.log"

acquire_runner_lock
cleanup_managed_server

SCENARIO=$(jq -r '.scenario' "$SPEC_FILE")
MODEL=$(jq -r '.model' "$SPEC_FILE")
TRACE_TARGET_ID=$(jq -r '.client_parameters.trace_target_id // empty' "$SPEC_FILE")
OFFICIAL_RUNTIME_ENVIRONMENT=$(jq -c '.baseline_target.runtime_environment // {}' "$SPEC_FILE")
if [[ "$OFFICIAL_RUNTIME_ENVIRONMENT" != "{}" ]]; then
  if [[ "$OFFICIAL_RUNTIME_ENVIRONMENT" != '{"VLLM_BATCH_INVARIANT":"1"}' ]]; then
    echo "Unsupported official runtime environment: $OFFICIAL_RUNTIME_ENVIRONMENT" >&2
    exit 2
  fi
  OFFICIAL_RUNTIME_VLLM_BATCH_INVARIANT=1
  export OFFICIAL_RUNTIME_VLLM_BATCH_INVARIANT
else
  unset OFFICIAL_RUNTIME_VLLM_BATCH_INVARIANT
fi
MODEL_REVISION=$(jq -r '.server_parameters.revision // empty' "$SPEC_FILE")
MODEL_PARAMETERS=$(jq -r '.model_parameters' "$SPEC_FILE")
MODEL_PRECISION=$(jq -r '.model_precision' "$SPEC_FILE")
MODEL_QUANTIZATION=$(jq -r '.model_quantization // empty' "$SPEC_FILE")
# Allow env var override for quantization (used by CI workflow)
if [[ -n "${CURRENT_MODEL_QUANTIZATION:-}" ]]; then
  MODEL_QUANTIZATION="$CURRENT_MODEL_QUANTIZATION"
fi
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
BENCHMARK_TYPE=$(resolve_scenario_benchmark_type)
OFFICIAL_CORE_SOURCE_REPOSITORY=${OFFICIAL_CORE_SOURCE_REPOSITORY:-"vllm-project/vllm"}
OFFICIAL_BACKEND_SOURCE_ENGINE=${OFFICIAL_BACKEND_SOURCE_ENGINE:-"vllm-ascend"}
OFFICIAL_BACKEND_SOURCE_REPOSITORY=${OFFICIAL_BACKEND_SOURCE_REPOSITORY:-"vllm-project/vllm-ascend"}
OFFICIAL_CORE_SOURCE_REF=$(jq -r '.baseline_target.vllm_ref // empty' "$SPEC_FILE")
OFFICIAL_BACKEND_SOURCE_REF=$(jq -r '.baseline_target.vllm_ascend_ref // empty' "$SPEC_FILE")

if [[ -z "$OFFICIAL_CORE_SOURCE_REF" ]] || [[ "$OFFICIAL_CORE_SOURCE_REF" == "null" ]]; then
  OFFICIAL_CORE_SOURCE_REF="$OFFICIAL_VLLM_REF"
fi
if [[ -z "$OFFICIAL_BACKEND_SOURCE_REF" ]] || [[ "$OFFICIAL_BACKEND_SOURCE_REF" == "null" ]]; then
  OFFICIAL_BACKEND_SOURCE_REF="$OFFICIAL_VLLM_ASCEND_REF"
fi

OFFICIAL_CORE_SOURCE_COMMIT=$(git -C "$OFFICIAL_VLLM_WORKTREE" rev-parse HEAD 2>/dev/null || true)
OFFICIAL_BACKEND_SOURCE_COMMIT=$(git -C "$OFFICIAL_VLLM_ASCEND_WORKTREE" rev-parse HEAD 2>/dev/null || true)
if [[ -z "$OFFICIAL_BACKEND_SOURCE_COMMIT" ]]; then
  OFFICIAL_BACKEND_SOURCE_COMMIT="$GIT_COMMIT"
fi
DECLARED_CORE_SOURCE_COMMIT=$(jq -r '.baseline_target.vllm_commit // empty' "$SPEC_FILE")
DECLARED_BACKEND_SOURCE_COMMIT=$(jq -r '.baseline_target.vllm_ascend_commit // empty' "$SPEC_FILE")
if [[ -n "$DECLARED_CORE_SOURCE_COMMIT" && "$OFFICIAL_CORE_SOURCE_COMMIT" != "$DECLARED_CORE_SOURCE_COMMIT" ]]; then
  echo "Official vLLM source commit mismatch: expected ${DECLARED_CORE_SOURCE_COMMIT}, got ${OFFICIAL_CORE_SOURCE_COMMIT}" >&2
  exit 2
fi
if [[ -n "$DECLARED_BACKEND_SOURCE_COMMIT" && "$OFFICIAL_BACKEND_SOURCE_COMMIT" != "$DECLARED_BACKEND_SOURCE_COMMIT" ]]; then
  echo "Official vLLM Ascend source commit mismatch: expected ${DECLARED_BACKEND_SOURCE_COMMIT}, got ${OFFICIAL_BACKEND_SOURCE_COMMIT}" >&2
  exit 2
fi

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

if [[ -n "$TRACE_TARGET_ID" ]]; then
  verify_trace_runtime_packages
fi

RUNTIME_MODEL="$MODEL"
cached_model_status=0
if cached_model_path=$(resolve_runtime_model); then
  RUNTIME_MODEL="$cached_model_path"
else
  cached_model_status=$?
  if [[ -n "$TRACE_TARGET_ID" ]]; then
    echo "Trace target model could not be resolved and verified at revision ${MODEL_REVISION:-<missing>}" >&2
    exit "$cached_model_status"
  fi
  if [[ "$cached_model_status" -eq 2 ]]; then
    echo "[goal-baseline] cached local snapshot is missing tokenizer or weight artifacts; falling back to model ID ${MODEL}" >&2
  fi
fi

SAME_SPEC_FILE="$RESULT_DIR/resolved_same_spec.json"
resolve_same_spec

resolved_dataset_path=$(jq -r '.resolved_client_parameters.dataset_path // empty' "$SAME_SPEC_FILE")
ensure_runtime_dataset_available "$resolved_dataset_path"

CLIENT_ARGS=$(json2args "$(normalized_client_parameters_json)")

RAW_RESULT_FILE="$RESULT_DIR/raw_benchmark_result.json"
ARTIFACT_DIR="$RESULT_DIR/submission"
TRACE_TARGET_ID=$(jq -r '.resolved_client_parameters.trace_target_id // empty' "$SAME_SPEC_FILE")
TRACE_ASSET_NAME=$(jq -r '.resolved_client_parameters.trace_asset // empty' "$SAME_SPEC_FILE")
TRACE_ASSET_PATH=""
TRACE_ENDPOINT=$(jq -r '.resolved_client_parameters.endpoint // "/v1/completions"' "$SAME_SPEC_FILE")
TRACE_MAX_MODEL_LEN=$(jq -r '.resolved_server_parameters.max_model_len' "$SAME_SPEC_FILE")
TRACE_MAX_REQUESTS=$(jq -r '.resolved_client_parameters.max_requests // 1000' "$SAME_SPEC_FILE")
TRACE_MAX_CONCURRENCY=$(jq -r '.resolved_client_parameters.max_concurrency // 64' "$SAME_SPEC_FILE")
TRACE_TIMEOUT_S=$(jq -r '.resolved_client_parameters.timeout_s // 600' "$SAME_SPEC_FILE")
TRACE_OVERFLOW_POLICY=$(jq -r '.resolved_client_parameters.overflow_policy // "reject"' "$SAME_SPEC_FILE")
TRACE_TIME_SCALE=$(jq -r '.resolved_client_parameters.time_scale // 1' "$SAME_SPEC_FILE")
TRACE_MAX_INTERARRIVAL_S=$(jq -r '.resolved_client_parameters.max_interarrival_s // 1' "$SAME_SPEC_FILE")
TRACE_DETAIL_RESULT_FILE="$RESULT_DIR/trace_replay_results.jsonl"
if [[ -n "$TRACE_TARGET_ID" ]]; then
  TRACE_ASSET_PATH="$OFFICIAL_TRACE_ASSET_ROOT/$TRACE_ASSET_NAME"
  if [[ ! -f "$TRACE_ASSET_PATH" ]]; then
    echo "official trace asset not found: $TRACE_ASSET_PATH" >&2
    exit 2
  fi
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$HOST_PYTHON_BIN" - <<'PY' "$TRACE_TARGET_ID" "$TRACE_ASSET_PATH"
from pathlib import Path
import sys
from vllm_hust_benchmark.official_trace_targets import get_trace_target, verify_trace_asset
verify_trace_asset(get_trace_target(sys.argv[1]), Path(sys.argv[2]))
PY
  prepare_trace_startup_evidence
fi

IMMUTABLE_INPUT_ATTESTATION_FILE="$RESULT_DIR/immutable-input-attestation.json"
IMMUTABLE_INPUT_METADATA=$(verify_immutable_input_contract)

echo "[goal-baseline] using worktrees: $OFFICIAL_VLLM_WORKTREE and $OFFICIAL_VLLM_ASCEND_WORKTREE"
echo "[goal-baseline] neutral cwd: $OFFICIAL_RUNTIME_CWD"
echo "[goal-baseline] vllm cache root: $OFFICIAL_VLLM_CACHE_ROOT"
echo "[goal-baseline] benchmark type: $BENCHMARK_TYPE"
echo "[goal-baseline] export model id: $MODEL"
echo "[goal-baseline] runtime model source: $RUNTIME_MODEL"
run_in_official_runtime_python "$OFFICIAL_RUNTIME_PYTHONPATH" <<'PY'
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

case "$BENCHMARK_TYPE" in
  serve)
    SERVER_HOST=$(jq -r '.resolved_server_parameters.host // "127.0.0.1"' "$SAME_SPEC_FILE")
    SERVER_PORT=$(jq -r '.resolved_server_parameters.port' "$SAME_SPEC_FILE")
    CLIENT_HOST=$(jq -r '.resolved_client_parameters.host // "127.0.0.1"' "$SAME_SPEC_FILE")
    CLIENT_PORT=$(jq -r '.resolved_client_parameters.port' "$SAME_SPEC_FILE")
    SERVER_ARGS=$(json2args "$(normalized_server_parameters_json | jq -c 'del(.disable_log_requests)')")

    BENCHMARK_SERVER_PORT="$SERVER_PORT" \
    PREPARE_BENCHMARK_ADMISSION_ONLY=1 \
    ENV_PREFIX="$GOAL_BASELINE_ENV_PREFIX" \
    VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
    bash "$PREPARE_SCRIPT"

    assert_target_port_available "Official baseline" "$CLIENT_HOST" "$CLIENT_PORT"

    SERVER_COMMAND="PYTHONUNBUFFERED=1 VLLM_VERSION=$OFFICIAL_CORE_VERSION PYTHONPATH=$OFFICIAL_RUNTIME_PYTHONPATH\${PYTHONPATH:+:\$PYTHONPATH} $OFFICIAL_RUNTIME_PYTHON -u -m vllm.entrypoints.openai.api_server $SERVER_ARGS"
    if [[ -n "$TRACE_TARGET_ID" ]]; then
      CLIENT_COMMAND="python -m vllm_hust_benchmark.trace_replay replay $TRACE_TARGET_ID --trace-path $TRACE_ASSET_PATH --max-requests $TRACE_MAX_REQUESTS --max-concurrency $TRACE_MAX_CONCURRENCY --timeout-s $TRACE_TIMEOUT_S --overflow-policy $TRACE_OVERFLOW_POLICY"
    else
      CLIENT_COMMAND="VLLM_VERSION=$OFFICIAL_CORE_VERSION PYTHONPATH=$OFFICIAL_RUNTIME_PYTHONPATH\${PYTHONPATH:+:\$PYTHONPATH} $OFFICIAL_RUNTIME_PYTHON $VLLM_CLI_COMPAT bench serve --save-result --result-dir $RESULT_DIR --result-filename $(basename "$RAW_RESULT_FILE") $CLIENT_ARGS"
    fi

    echo "[goal-baseline] benchmark endpoint: ${CLIENT_HOST}:${CLIENT_PORT}"
    echo "[goal-baseline] server command: $SERVER_COMMAND"
    server_ready=0
    for start_attempt in $(seq 1 "$SERVER_START_RETRIES"); do
      if wait_for_single_card_ascend_device; then
        selection_status=0
      else
        selection_status=$?
      fi

      if [[ "$selection_status" -ne 0 ]]; then
        if [[ "$selection_status" -eq "$RESOURCE_BUSY_EXIT_CODE" && "${GOAL_BASELINE_DEVICE_SELECTION_REASON:-}" == "all-busy" ]]; then
          echo "[goal-baseline] All detected Ascend devices remained busy after ${DEVICE_SELECTION_RETRIES} selection attempt(s)" >&2
        fi
        exit "$selection_status"
      fi

      echo "[goal-baseline] Ascend visible devices: ${ASCEND_VISIBLE_DEVICES:-<unset>} (rt=${ASCEND_RT_VISIBLE_DEVICES:-<unset>})"
      start_peak_hbm_sampler

      if wait_for_ascend_runtime_ready; then
        runtime_ready_status=0
      else
        runtime_ready_status=$?
      fi

      if [[ "$runtime_ready_status" -ne 0 ]]; then
        echo "[goal-baseline] Ascend runtime did not become ready after ${ASCEND_RUNTIME_READY_TIMEOUT_SECONDS}s" >&2
        if [[ "$start_attempt" -lt "$SERVER_START_RETRIES" ]]; then
          echo "[goal-baseline] Retrying server start after runtime readiness failure in ${SERVER_START_RETRY_DELAY_SECONDS}s (attempt ${start_attempt}/${SERVER_START_RETRIES})" >&2
          sleep "$SERVER_START_RETRY_DELAY_SECONDS"
          continue
        fi
        exit "$runtime_ready_status"
      fi

      : > "$SERVER_STDOUT_LOG"
      run_server_command >"$SERVER_STDOUT_LOG" 2>&1 &
      SERVER_PID=$!
      persist_managed_server_state

      if wait_for_server "$CLIENT_HOST" "$CLIENT_PORT"; then
        persist_managed_server_state
        server_ready=1
        break
      else
        # Capture the condition status inside the else branch. The status of
        # an `if` compound with no executed branch is zero in bash, which used
        # to turn an early server crash into a successful runner exit.
        server_wait_status=$?
      fi

      if [[ "$server_wait_status" -eq "$RESOURCE_BUSY_EXIT_CODE" && "$start_attempt" -lt "$SERVER_START_RETRIES" ]]; then
        echo "[goal-baseline] Detected transient Ascend resource busy state; retrying server start in ${SERVER_START_RETRY_DELAY_SECONDS}s (attempt ${start_attempt}/${SERVER_START_RETRIES})" >&2
        cleanup_managed_server || true
        sleep "$SERVER_START_RETRY_DELAY_SECONDS"
        continue
      fi

      exit "$server_wait_status"
    done

    if [[ "$server_ready" != "1" ]]; then
      echo "[goal-baseline] vLLM server did not become ready after ${SERVER_START_RETRIES} start attempt(s)" >&2
      exit 1
    fi
    ;;
  throughput|latency)
    wait_for_single_card_ascend_device
    echo "[goal-baseline] Ascend visible devices: ${ASCEND_VISIBLE_DEVICES:-<unset>} (rt=${ASCEND_RT_VISIBLE_DEVICES:-<unset>})"
    start_peak_hbm_sampler
    CLIENT_COMMAND="VLLM_VERSION=$OFFICIAL_CORE_VERSION PYTHONPATH=$OFFICIAL_RUNTIME_PYTHONPATH\${PYTHONPATH:+:\$PYTHONPATH} $OFFICIAL_RUNTIME_PYTHON $VLLM_CLI_COMPAT bench $BENCHMARK_TYPE --output-json $RAW_RESULT_FILE $CLIENT_ARGS"
    ;;
  *)
    echo "Unsupported benchmark type for official baseline runner: $BENCHMARK_TYPE" >&2
    exit 2
    ;;
esac

echo "[goal-baseline] client command: $CLIENT_COMMAND"
run_client_command
finalize_trace_immutable_input_attestation
if [[ -n "$TRACE_TARGET_ID" ]]; then
  finalize_trace_startup_evidence
fi
stop_peak_hbm_sampler
if [[ ! -f "$PEAK_HBM_EVIDENCE_FILE" ]] || ! jq -e \
  '.sample_count > 0 and .peak_hbm_mb > 0' "$PEAK_HBM_EVIDENCE_FILE" >/dev/null; then
  echo "Peak HBM sampling did not produce valid evidence" >&2
  exit 2
fi
PEAK_HBM_MB=$(jq -r '.peak_hbm_mb' "$PEAK_HBM_EVIDENCE_FILE")
SPEC_REPRO_PATH=$(realpath --relative-to="$REPO_ROOT" "$SPEC_FILE")
printf -v REPRODUCIBLE_CMD 'GOAL_BASELINE_ENV_PREFIX=<env-prefix> bash scripts/run-official-ascend-goal-baseline.sh %q' "$SPEC_REPRO_PATH"

EXPORT_ARGS=(
  python -m vllm_hust_benchmark.cli export-leaderboard-artifact
  "$SCENARIO"
  --benchmark-result-file "$RAW_RESULT_FILE"
  --constraints-file "$CONSTRAINTS_FILE"
  --same-spec-file "$SAME_SPEC_FILE"
  --output-dir "$ARTIFACT_DIR"
  --run-id "$RUN_ID"
  --engine "$ENGINE"
  --engine-version "$ENGINE_VERSION"
  --core-version N/A
  --backend-version N/A
  --model-name "$MODEL"
  --model-parameters "$MODEL_PARAMETERS"
  --model-precision "$MODEL_PRECISION"
  ${MODEL_QUANTIZATION:+--quantization "$MODEL_QUANTIZATION"}
  --hardware-vendor "$HARDWARE_VENDOR"
  --hardware-chip-model "$HARDWARE_CHIP_MODEL"
  --chip-count "$CHIP_COUNT"
  --node-count "$NODE_COUNT"
  --submitter "$SUBMITTER"
  --baseline-engine "$BASELINE_ENGINE"
  --data-source "$DATA_SOURCE"
  --git-commit "$GIT_COMMIT"
  --github-repository "$GITHUB_REPOSITORY"
  --github-ref "$GITHUB_REF"
  --peak-mem-mb "$PEAK_HBM_MB"
  --reproducible-cmd "$REPRODUCIBLE_CMD"
  --runtime-python "$OFFICIAL_RUNTIME_PYTHON"
  --engine-source-repository "$OFFICIAL_CORE_SOURCE_REPOSITORY"
  --engine-source-ref "$OFFICIAL_CORE_SOURCE_REF"
  --engine-source-commit "$OFFICIAL_CORE_SOURCE_COMMIT"
  --plugin-source-engine "$OFFICIAL_BACKEND_SOURCE_ENGINE"
  --plugin-source-repository "$OFFICIAL_BACKEND_SOURCE_REPOSITORY"
  --plugin-source-ref "$OFFICIAL_BACKEND_SOURCE_REF"
  --plugin-source-commit "$OFFICIAL_BACKEND_SOURCE_COMMIT"
)

append_export_arg_from_spec --input-length '.client_parameters.input_len'
append_export_arg_from_spec --output-length '.client_parameters.output_len'
append_export_arg_from_spec --batch-size '.client_parameters.batch_size'
append_export_arg_from_spec --concurrent-requests '.client_parameters.max_concurrency'

run_in_official_runtime "$REPO_ROOT/src:$OFFICIAL_RUNTIME_PYTHONPATH" "${EXPORT_ARGS[@]}"

echo "[goal-baseline] exported leaderboard artifact to $ARTIFACT_DIR"
