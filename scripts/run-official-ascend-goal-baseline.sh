#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PREPARE_SCRIPT=${PREPARE_SCRIPT:-"$REPO_ROOT/scripts/prepare-official-ascend-baseline-env.sh"}
VLLM_CLI_COMPAT=${VLLM_CLI_COMPAT:-"$REPO_ROOT/scripts/run_vllm_cli_compat.py"}
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
    if [[ -n "${OFFICIAL_CORE_VERSION:-}" ]]; then
      export VLLM_VERSION="$OFFICIAL_CORE_VERSION"
    fi
    PYTHONPATH="$pythonpath_prefix${PYTHONPATH:+:$PYTHONPATH}" \
      conda run -p "$GOAL_BASELINE_ENV_PREFIX" "$@"
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

  ASCEND_DEVICE_SELECTION_ATTEMPT="$selection_attempt" \
    NPU_SMI_BIN="$npu_smi_bin" \
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


def select_best_idle_device(
  info_output: str,
  logical_map: dict[tuple[str, str], int],
  busy_devices: set[int],
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

info_result = run_npu_smi("info")
info_failure_reason = classify_npu_smi_failure(info_result)
if info_result is not None and info_result.returncode == 0:
    busy_devices = list_process_busy_devices(info_result.stdout)

    selected_device = select_best_idle_device(info_result.stdout, logical_map, busy_devices)
    if selected_device is not None:
        device_id, device_source = selected_device
        print(f"{device_id}\t{device_source}")
        sys.exit(0)

    status_devices = list_status_devices(info_result.stdout)
    if busy_devices:
        status_devices = [device for device in status_devices if device not in busy_devices]

    if status_devices:
        fallback_device = status_devices[(selection_attempt - 1) % len(status_devices)]
        print(f"{fallback_device}\tstatus-round-robin")
        sys.exit(0)

    if busy_devices:
        busy_device_list = sorted(busy_devices)
        print("__ALL_BUSY__\t" + ",".join(str(device) for device in busy_device_list))
        sys.exit(0)

fallback_failure_reason = info_failure_reason or mapping_failure_reason

if logical_devices:
    fallback_device = logical_devices[(selection_attempt - 1) % len(logical_devices)]
    print(f"{fallback_device}\t{annotate_fallback_source('logical-round-robin', fallback_failure_reason)}")
    sys.exit(0)

devnode_devices = list_devnode_devices()
if devnode_devices:
    fallback_device = devnode_devices[(selection_attempt - 1) % len(devnode_devices)]
    print(f"{fallback_device}\t{annotate_fallback_source('devnode-round-robin', fallback_failure_reason)}")
    sys.exit(0)

sys.exit(1)
PY
}

configure_single_card_ascend_device() {
  local start_attempt=${1:-1}
  local busy_exit_code=${RESOURCE_BUSY_EXIT_CODE:-75}
  local resolved_visible_devices=""
  local resolved_rt_visible_devices=""
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
    GOAL_BASELINE_DEVICE_SELECTION_REASON="explicit"
    export VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE="${VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE:-npu:0}"
    echo "[goal-baseline] using explicit Ascend visible devices: ${ASCEND_VISIBLE_DEVICES:-$ASCEND_RT_VISIBLE_DEVICES}"
    return 0
  fi

  npu_smi_bin=$(resolve_npu_smi_bin 2>/dev/null || true)
  if [[ -n "$npu_smi_bin" ]]; then
    echo "[goal-baseline] using npu-smi for device selection: $npu_smi_bin"
  fi

  selected_device_info=$(select_ascend_device "$start_attempt" "$npu_smi_bin" 2>/dev/null || true)
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
    if [[ -n "${OFFICIAL_CORE_VERSION:-}" ]]; then
      export VLLM_VERSION="$OFFICIAL_CORE_VERSION"
    fi
    PYTHONUNBUFFERED=1 \
      PYTHONPATH="$OFFICIAL_RUNTIME_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
      conda run --no-capture-output -p "$GOAL_BASELINE_ENV_PREFIX" \
      python -u -m vllm.entrypoints.openai.api_server $SERVER_ARGS
  )
}

run_client_command() {
  case "$BENCHMARK_TYPE" in
    serve)
      run_in_official_runtime "$OFFICIAL_RUNTIME_PYTHONPATH" \
        python "$VLLM_CLI_COMPAT" bench serve \
        --save-result \
        --result-dir "$RESULT_DIR" \
        --result-filename "$(basename "$RAW_RESULT_FILE")" \
        $CLIENT_ARGS
      ;;
    throughput|latency)
      run_in_official_runtime "$OFFICIAL_RUNTIME_PYTHONPATH" \
        python "$VLLM_CLI_COMPAT" bench "$BENCHMARK_TYPE" \
        --output-json "$RAW_RESULT_FILE" \
        $CLIENT_ARGS
      ;;
    *)
      echo "Unsupported benchmark type for official baseline runner: $BENCHMARK_TYPE" >&2
      return 2
      ;;
  esac
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

  run_in_official_runtime_python "$REPO_ROOT/src${OFFICIAL_RUNTIME_PYTHONPATH:+:$OFFICIAL_RUNTIME_PYTHONPATH}" \
    env RUNTIME_MODEL_CANDIDATE="$runtime_model_candidate" <<'PY' >/dev/null
import os

from vllm_hust_benchmark.same_spec import runtime_model_path_has_required_artifacts

raise SystemExit(
    0 if runtime_model_path_has_required_artifacts(os.environ["RUNTIME_MODEL_CANDIDATE"]) else 1
)
PY
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

  grep -Eq "DrvMngGetConsoleLogLevel failed|dcmi model initialized failed|ret is -8020|drvRet=87|drvRetCode=87|ErrCode=507899|error code is 507899|rtGetDeviceCount|Can't get ascend_hal device count|driver error:internal error|Resource_Busy\(EL0005\)|The resources are busy|ERR99999 UNKNOWN applicaiton exception|ERR99999 UNKNOWN application exception|Engine core initialization failed" "$log_file"
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
SERVER_STDOUT_LOG="$RESULT_DIR/server.stdout.log"
RUNTIME_READY_LOG="$RESULT_DIR/runtime-ready.log"

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
BENCHMARK_TYPE=$(resolve_scenario_benchmark_type)

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
    SERVER_HOST=$(jq -r '.resolved_server_parameters.host' "$SAME_SPEC_FILE")
    SERVER_PORT=$(jq -r '.resolved_server_parameters.port' "$SAME_SPEC_FILE")
    CLIENT_HOST=$(jq -r '.resolved_client_parameters.host' "$SAME_SPEC_FILE")
    CLIENT_PORT=$(jq -r '.resolved_client_parameters.port' "$SAME_SPEC_FILE")
    SERVER_ARGS=$(json2args "$(jq -c '.resolved_server_parameters | del(.disable_log_requests)' "$SAME_SPEC_FILE")")

    BENCHMARK_SERVER_PORT="$SERVER_PORT" \
    PREPARE_BENCHMARK_ADMISSION_ONLY=1 \
    ENV_PREFIX="$GOAL_BASELINE_ENV_PREFIX" \
    VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
    bash "$PREPARE_SCRIPT"

    assert_target_port_available "Official baseline" "$CLIENT_HOST" "$CLIENT_PORT"

    SERVER_COMMAND="PYTHONUNBUFFERED=1 VLLM_VERSION=$OFFICIAL_CORE_VERSION PYTHONPATH=$OFFICIAL_RUNTIME_PYTHONPATH\${PYTHONPATH:+:\$PYTHONPATH} conda run --no-capture-output -p $GOAL_BASELINE_ENV_PREFIX python -u -m vllm.entrypoints.openai.api_server $SERVER_ARGS"
    CLIENT_COMMAND="VLLM_VERSION=$OFFICIAL_CORE_VERSION PYTHONPATH=$OFFICIAL_RUNTIME_PYTHONPATH\${PYTHONPATH:+:\$PYTHONPATH} conda run -p $GOAL_BASELINE_ENV_PREFIX python $VLLM_CLI_COMPAT bench serve --save-result --result-dir $RESULT_DIR --result-filename $(basename "$RAW_RESULT_FILE") $CLIENT_ARGS"

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
      fi

      server_wait_status=$?
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
    CLIENT_COMMAND="VLLM_VERSION=$OFFICIAL_CORE_VERSION PYTHONPATH=$OFFICIAL_RUNTIME_PYTHONPATH\${PYTHONPATH:+:\$PYTHONPATH} conda run -p $GOAL_BASELINE_ENV_PREFIX python $VLLM_CLI_COMPAT bench $BENCHMARK_TYPE --output-json $RAW_RESULT_FILE $CLIENT_ARGS"
    ;;
  *)
    echo "Unsupported benchmark type for official baseline runner: $BENCHMARK_TYPE" >&2
    exit 2
    ;;
esac

echo "[goal-baseline] client command: $CLIENT_COMMAND"
run_client_command

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
  --core-version "$OFFICIAL_CORE_VERSION"
  --backend-version "$OFFICIAL_BACKEND_VERSION"
  --model-name "$MODEL"
  --model-parameters "$MODEL_PARAMETERS"
  --model-precision "$MODEL_PRECISION"
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
)

append_export_arg_from_spec --input-length '.client_parameters.input_len'
append_export_arg_from_spec --output-length '.client_parameters.output_len'
append_export_arg_from_spec --batch-size '.client_parameters.batch_size'

run_in_official_runtime "$REPO_ROOT/src:$OFFICIAL_RUNTIME_PYTHONPATH" "${EXPORT_ARGS[@]}"

echo "[goal-baseline] exported leaderboard artifact to $ARTIFACT_DIR"