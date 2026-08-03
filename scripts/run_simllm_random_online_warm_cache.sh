#!/bin/bash
set -euo pipefail

# SimLLM random-online A/B runner with a warm-cache phase.
#
# Flow:
#   1. Baseline: run the official same-spec benchmark with SimLLM disabled.
#   2. SimLLM: start one server, run an unmeasured warmup pass, then run the
#      measured benchmark against the still-live server so KVManager remains hot.
#
# Example:
#   cd /workspace/vllm-hust-benchmark
#   ASCEND_RT_VISIBLE_DEVICES=6 \
#   CURRENT_MODEL_PATH=/data/shared_models/Qwen2.5-14B-Instruct \
#   RESULT_DIR=/workspace/vllm-hust-benchmark/.benchmarks/simllm-random-online-warm-cache \
#   bash scripts/run_simllm_random_online_warm_cache.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=${VLLM_HUST_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}
BENCHMARK_REPO=${VLLM_HUST_BENCHMARK_REPO:-"$WORKSPACE_ROOT/vllm-hust-benchmark"}
DEFAULT_SPEC_FILE="$BENCHMARK_REPO/docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json"
VLLM_CLI_COMPAT=${VLLM_CLI_COMPAT:-"$BENCHMARK_REPO/scripts/run_vllm_cli_compat.py"}
SPEC_FILE=${1:-"$DEFAULT_SPEC_FILE"}
CONSTRAINTS_FILE=${CONSTRAINTS_FILE:-"$BENCHMARK_REPO/docs/official-baselines/official-ascend-constraints.stub.json"}

CURRENT_RUNTIME_CWD=${CURRENT_RUNTIME_CWD:-"/tmp"}
CURRENT_VLLM_HUST_REPO=${CURRENT_VLLM_HUST_REPO:-"$WORKSPACE_ROOT/vllm-hust"}
CURRENT_VLLM_ASCEND_HUST_REPO=${CURRENT_VLLM_ASCEND_HUST_REPO:-"$WORKSPACE_ROOT/vllm-ascend-hust"}
CURRENT_ENV_PREFIX=${CURRENT_ENV_PREFIX:-"/root/miniconda3/envs/vllm-hust-dev"}
CURRENT_RUNTIME_PYTHON=${CURRENT_RUNTIME_PYTHON:-"$CURRENT_ENV_PREFIX/bin/python"}
CURRENT_RUNTIME_IMAGE=${CURRENT_RUNTIME_IMAGE:-}
CURRENT_RUNTIME_IMAGE_DIGEST=${CURRENT_RUNTIME_IMAGE_DIGEST:-}
CURRENT_DEVICE_ID=${CURRENT_DEVICE_ID:-${ASCEND_RT_VISIBLE_DEVICES:-}}
CURRENT_NPU_SMI_DEVICE_ID=${CURRENT_NPU_SMI_DEVICE_ID:-$CURRENT_DEVICE_ID}
CURRENT_VLLM_CACHE_ROOT=${CURRENT_VLLM_CACHE_ROOT:-"$REPO_ROOT/.cache/simllm-warm-cache"}
CURRENT_ENGINE=${CURRENT_ENGINE:-"vllm-hust"}
CURRENT_BASELINE_ARM_ENGINE=${CURRENT_BASELINE_ARM_ENGINE:-"vllm-hust"}
CURRENT_SIMLLM_ARM_ENGINE=${CURRENT_SIMLLM_ARM_ENGINE:-"vllm-hust-simllm"}
CURRENT_BASELINE_ENGINE=${CURRENT_BASELINE_ENGINE:-"vllm"}
CURRENT_DATA_SOURCE=${CURRENT_DATA_SOURCE:-"vllm-hust-benchmark"}
CURRENT_SUBMITTER=${CURRENT_SUBMITTER:-"simllm-warm-cache"}
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
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-600}
READY_STATUS_INTERVAL_SECONDS=${READY_STATUS_INTERVAL_SECONDS:-30}
CLIENT_READY_CHECK_TIMEOUT_SECONDS=${CLIENT_READY_CHECK_TIMEOUT_SECONDS:-$READY_TIMEOUT_SECONDS}

CURRENT_SERVER_HOST=${CURRENT_SERVER_HOST:-}
CURRENT_SERVER_PORT=${CURRENT_SERVER_PORT:-"8021"}
CURRENT_CLIENT_HOST=${CURRENT_CLIENT_HOST:-}
CURRENT_CLIENT_PORT=${CURRENT_CLIENT_PORT:-$CURRENT_SERVER_PORT}
CURRENT_CLIENT_TEMPERATURE=${CURRENT_CLIENT_TEMPERATURE:-}
CURRENT_DTYPE=${CURRENT_DTYPE:-}
CURRENT_MODEL_NAME=${CURRENT_MODEL_NAME:-}
CURRENT_MODEL_PARAMETERS=${CURRENT_MODEL_PARAMETERS:-}
CURRENT_MODEL_PRECISION=${CURRENT_MODEL_PRECISION:-}
CURRENT_MODEL_QUANTIZATION=${CURRENT_MODEL_QUANTIZATION:-}
CURRENT_HARDWARE_CHIP_MODEL=${CURRENT_HARDWARE_CHIP_MODEL:-}
BASELINE_SERVER_PORT=${BASELINE_SERVER_PORT:-$CURRENT_SERVER_PORT}
SIMLLM_SERVER_PORT=${SIMLLM_SERVER_PORT:-$CURRENT_SERVER_PORT}

RESULT_DIR=${RESULT_DIR:-"$REPO_ROOT/.benchmarks/simllm-random-online-warm-cache"}
BASELINE_DIR=${BASELINE_DIR:-"$RESULT_DIR/baseline-disabled"}
SIMLLM_DIR=${SIMLLM_DIR:-"$RESULT_DIR/enabled-warm-cache"}
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_SIMLLM=${RUN_SIMLLM:-1}
RUN_TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)

SIMLLM_COSINE_THRESHOLD=${VLLM_ASCEND_SIMLLM_COSINE_THRESHOLD:-${SIMLLM_COSINE_THRESHOLD:-0.8}}
SIMLLM_LSH_NUM_BITS=${VLLM_ASCEND_SIMLLM_LSH_NUM_BITS:-${SIMLLM_LSH_NUM_BITS:-64}}
SIMLLM_LSH_BATCH_THRESHOLD=${VLLM_ASCEND_SIMLLM_LSH_BATCH_THRESHOLD:-${SIMLLM_LSH_BATCH_THRESHOLD:-32}}
SIMLLM_KV_CACHE_SIZE=${VLLM_ASCEND_SIMLLM_KV_CACHE_SIZE:-${SIMLLM_KV_CACHE_SIZE:-1024}}
SIMLLM_SANDWICH_BOTTOM=${VLLM_ASCEND_SIMLLM_SANDWICH_BOTTOM:-${SIMLLM_SANDWICH_BOTTOM:-3}}
SIMLLM_SANDWICH_TOP=${VLLM_ASCEND_SIMLLM_SANDWICH_TOP:-${SIMLLM_SANDWICH_TOP:-3}}
SIMLLM_UNMATCHED_STORE_MODE=${VLLM_ASCEND_SIMLLM_UNMATCHED_STORE_MODE:-${SIMLLM_UNMATCHED_STORE_MODE:-top}}
SIMLLM_PROFILE=${VLLM_ASCEND_SIMLLM_PROFILE:-${SIMLLM_PROFILE:-0}}
SIMLLM_PROFILE_INTERVAL=${VLLM_ASCEND_SIMLLM_PROFILE_INTERVAL:-${SIMLLM_PROFILE_INTERVAL:-20}}
SIMLLM_REQUIRE_REWRITE_EVIDENCE=${SIMLLM_REQUIRE_REWRITE_EVIDENCE:-1}
SIMLLM_OFFICIAL_EVIDENCE=${SIMLLM_OFFICIAL_EVIDENCE:-1}

SIMLLM_WARMCACHE_PASSES=${SIMLLM_WARMCACHE_PASSES:-1}
SIMLLM_WARMCACHE_NUM_PROMPTS=${SIMLLM_WARMCACHE_NUM_PROMPTS:-}
SIMLLM_WARMCACHE_REQUEST_RATE=${SIMLLM_WARMCACHE_REQUEST_RATE:-}
SIMLLM_WARMCACHE_SEED=${SIMLLM_WARMCACHE_SEED:-0}
SIMLLM_MEASURE_SEED=${SIMLLM_MEASURE_SEED:-$SIMLLM_WARMCACHE_SEED}
SIMLLM_WARMCACHE_PAUSE_SECONDS=${SIMLLM_WARMCACHE_PAUSE_SECONDS:-5}

ACTIVE_SERVER_PID=""
ACTIVE_ARM_LABEL=""
CURRENT_RUNTIME_SOURCE_PYTHONPATH="$BENCHMARK_REPO/src:$CURRENT_VLLM_ASCEND_HUST_REPO:$CURRENT_VLLM_HUST_REPO"
CURRENT_RUNTIME_PYTHONPATH="${CURRENT_RUNTIME_SOURCE_PYTHONPATH}${CURRENT_RUNTIME_PYTHONPATH:+:$CURRENT_RUNTIME_PYTHONPATH}"

disable_proxy_env() {
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1}"
  export no_proxy="$NO_PROXY"
}

usage() {
  echo "Usage: $0 [random-online-official-spec.json]" >&2
}

if [[ -z "$SPEC_FILE" ]]; then
  usage
  exit 2
fi

if [[ ! -d "$BENCHMARK_REPO" ]]; then
  echo "Benchmark repo not found: $BENCHMARK_REPO" >&2
  echo "Set VLLM_HUST_BENCHMARK_REPO to the vllm-hust-benchmark checkout." >&2
  exit 2
fi
SPEC_FILE=$(cd "$(dirname "$SPEC_FILE")" && pwd)/$(basename "$SPEC_FILE")
if [[ ! -f "$SPEC_FILE" ]]; then
  echo "Spec file not found: $SPEC_FILE" >&2
  exit 2
fi
if [[ ! -f "$CONSTRAINTS_FILE" ]]; then
  echo "Constraints stub not found: $CONSTRAINTS_FILE" >&2
  exit 2
fi
if [[ ! -x "$CURRENT_RUNTIME_PYTHON" ]]; then
  echo "CURRENT_RUNTIME_PYTHON is not executable: $CURRENT_RUNTIME_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$VLLM_CLI_COMPAT" ]]; then
  echo "CLI compatibility wrapper not found: $VLLM_CLI_COMPAT" >&2
  exit 2
fi

run_in_runtime() {
  local pythonpath_prefix=${1:-$CURRENT_RUNTIME_PYTHONPATH}
  shift
  (
    cd "$CURRENT_RUNTIME_CWD"
    disable_proxy_env
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
    export VLLM_CACHE_ROOT="$CURRENT_VLLM_CACHE_ROOT"
    PYTHONPATH="$pythonpath_prefix${PYTHONPATH:+:$PYTHONPATH}" "$@"
  )
}

json2args() {
  local json_string=$1
  echo "$json_string" | jq -r '
    to_entries
    | map(
        select(.value != null and .value != false)
        | if .value == true or (.value | tostring) == ""
          then "--" + (.key | gsub("_"; "-"))
          else "--" + (.key | gsub("_"; "-")) + " " + (.value | tostring)
          end
      )
    | join(" ")
  '
}

maybe_git_describe() {
  local repo=$1
  local commit=${2:-}
  local described=""

  if [[ -n "$commit" ]]; then
    described=$(git -C "$repo" describe --tags --always "$commit" 2>/dev/null || true)
  fi
  if [[ -z "$described" ]]; then
    described=$(git -C "$repo" describe --tags --always HEAD 2>/dev/null || true)
  fi
  printf '%s' "${described:-unknown}"
}

append_export_arg_if_present() {
  local flag=$1
  local value=$2

  if [[ -n "$value" && "$value" != "null" ]]; then
    EXPORT_ARGS+=("$flag" "$value")
  fi
}

resolve_benchmark_type() {
  local scenario=$1
  run_in_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
    env SCENARIO_NAME="$scenario" \
    "$CURRENT_RUNTIME_PYTHON" -c \
    "import os; from vllm_hust_benchmark.registry import get_scenario; print(get_scenario(os.environ['SCENARIO_NAME']).benchmark_type)"
}

resolve_spec() {
  local spec_file=$1
  local output_file=$2
  local runtime_model=$3
  local port=$4
  local resolve_args=(
    "$CURRENT_RUNTIME_PYTHON" -m vllm_hust_benchmark.same_spec resolve
    --spec-file "$spec_file"
    --output-file "$output_file"
    --runtime-model "$runtime_model"
    --server-port "$port"
    --client-port "$port"
  )

  if [[ -n "$CURRENT_SERVER_HOST" ]]; then
    resolve_args+=(--server-host "$CURRENT_SERVER_HOST")
  fi
  if [[ -n "$CURRENT_CLIENT_HOST" ]]; then
    resolve_args+=(--client-host "$CURRENT_CLIENT_HOST")
  fi
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
  if [[ -n "$CURRENT_MODEL_QUANTIZATION" ]]; then
    resolve_args+=(--model-quantization "$CURRENT_MODEL_QUANTIZATION")
  fi
  if [[ -n "$CURRENT_HARDWARE_CHIP_MODEL" ]]; then
    resolve_args+=(--hardware-chip-model "$CURRENT_HARDWARE_CHIP_MODEL")
  fi

  run_in_runtime "$CURRENT_RUNTIME_PYTHONPATH" "${resolve_args[@]}"
}

normalized_client_parameters_json() {
  local same_spec_file=$1
  run_in_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
    env SAME_SPEC_FILE="$same_spec_file" \
    BENCHMARK_TYPE=serve \
    CLIENT_READY_CHECK_TIMEOUT_SECONDS="$CLIENT_READY_CHECK_TIMEOUT_SECONDS" \
    CURRENT_CLIENT_TEMPERATURE="$CURRENT_CLIENT_TEMPERATURE" \
    CURRENT_VLLM_WORKTREE="$CURRENT_VLLM_HUST_REPO" \
    "$CURRENT_RUNTIME_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

from vllm_hust_benchmark.official_runtime_inputs import normalize_client_parameters

payload = json.loads(Path(os.environ["SAME_SPEC_FILE"]).read_text(encoding="utf-8"))
client_temperature = os.environ.get("CURRENT_CLIENT_TEMPERATURE", "").strip()
parameters = normalize_client_parameters(
    payload["resolved_client_parameters"],
    benchmark_type=os.environ["BENCHMARK_TYPE"],
    ready_check_timeout_sec=int(os.environ.get("CLIENT_READY_CHECK_TIMEOUT_SECONDS") or 0),
    vllm_worktree=os.environ.get("CURRENT_VLLM_WORKTREE"),
)
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

client_json_with_seed() {
  local client_json=$1
  local seed=$2

  if [[ -z "$seed" ]]; then
    printf '%s' "$client_json"
    return 0
  fi

  echo "$client_json" | jq -c --arg seed "$seed" '. + {seed: ($seed | tonumber)}'
}

warmup_client_json() {
  local client_json=$1

  echo "$client_json" | jq -c \
    --arg seed "$SIMLLM_WARMCACHE_SEED" \
    --arg num_prompts "$SIMLLM_WARMCACHE_NUM_PROMPTS" \
    --arg request_rate "$SIMLLM_WARMCACHE_REQUEST_RATE" '
      def maybe_number($value): try ($value | tonumber) catch $value;
      . + {seed: ($seed | tonumber)}
      | if $num_prompts == "" then . else . + {num_prompts: maybe_number($num_prompts)} end
      | if $request_rate == "" then . else . + {request_rate: maybe_number($request_rate)} end
    '
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

port_has_listener() {
  [[ -n "$(list_port_listener_pids "$1")" ]]
}

assert_port_available() {
  local port=$1

  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "Target port ${port} already has a healthy service; refusing to reuse it." >&2
    return 1
  fi
  if port_has_listener "$port"; then
    echo "Target port ${port} already has a listener; choose another port or stop it first." >&2
    return 1
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

terminate_pid_tree() {
  local pid=$1
  local description=$2
  local tree_pids
  local tree_list
  local tree_pid
  local still_running

  tree_pids=$(collect_process_tree_pids "$pid" | sort -u)
  [[ -z "$tree_pids" ]] && return 0
  tree_list=$(echo "$tree_pids" | tr '\n' ' ')

  echo "[simllm-warm-cache] stopping ${description}: ${tree_list}"
  kill $tree_list 2>/dev/null || true

  for _ in $(seq 1 10); do
    still_running=0
    while IFS= read -r tree_pid; do
      [[ -z "$tree_pid" ]] && continue
      if kill -0 "$tree_pid" 2>/dev/null; then
        still_running=1
        break
      fi
    done <<< "$tree_pids"
    [[ "$still_running" == "0" ]] && return 0
    sleep 1
  done

  kill -9 $tree_list 2>/dev/null || true
}

cleanup_active_server() {
  if [[ -n "${ACTIVE_SERVER_PID:-}" ]]; then
    terminate_pid_tree "$ACTIVE_SERVER_PID" "active SimLLM benchmark server" || true
    ACTIVE_SERVER_PID=""
    ACTIVE_ARM_LABEL=""
  fi
}

trap cleanup_active_server EXIT

probe_server_ready() {
  local host=$1
  local port=$2

  curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1 \
    || curl -fsS "http://${host}:${port}/v1/models" >/dev/null 2>&1
}

wait_for_server() {
  local host=$1
  local port=$2
  local log_file=$3
  local waited=0
  local next_status_at=0

  while (( waited < READY_TIMEOUT_SECONDS )); do
    if [[ -n "$ACTIVE_SERVER_PID" ]] && ! kill -0 "$ACTIVE_SERVER_PID" 2>/dev/null; then
      echo "Server exited before becoming ready." >&2
      tail -n 60 "$log_file" >&2 || true
      return 1
    fi

    if probe_server_ready "$host" "$port"; then
      echo "[simllm-warm-cache] server ready after ${waited}s"
      return 0
    fi

    if (( waited >= next_status_at )); then
      echo "[simllm-warm-cache] waiting for server at ${host}:${port} (${waited}s/${READY_TIMEOUT_SECONDS}s)" >&2
      next_status_at=$((waited + READY_STATUS_INTERVAL_SECONDS))
    fi
    sleep 1
    ((waited += 1))
  done

  echo "Timed out waiting for server at ${host}:${port}" >&2
  tail -n 60 "$log_file" >&2 || true
  return 1
}

start_server() {
  local simllm_enabled=$1
  local server_args=$2
  local log_file=$3
  local arm_label=$4

  : >"$log_file"
  (
    cd "$CURRENT_RUNTIME_CWD"
    disable_proxy_env
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
    export VLLM_CACHE_ROOT="$CURRENT_VLLM_CACHE_ROOT"
    export VLLM_ASCEND_SIMLLM_ENABLED="$simllm_enabled"
    export VLLM_ASCEND_SIMLLM_COSINE_THRESHOLD="$SIMLLM_COSINE_THRESHOLD"
    export VLLM_ASCEND_SIMLLM_LSH_NUM_BITS="$SIMLLM_LSH_NUM_BITS"
    export VLLM_ASCEND_SIMLLM_LSH_BATCH_THRESHOLD="$SIMLLM_LSH_BATCH_THRESHOLD"
    export VLLM_ASCEND_SIMLLM_KV_CACHE_SIZE="$SIMLLM_KV_CACHE_SIZE"
    export VLLM_ASCEND_SIMLLM_SANDWICH_BOTTOM="$SIMLLM_SANDWICH_BOTTOM"
    export VLLM_ASCEND_SIMLLM_SANDWICH_TOP="$SIMLLM_SANDWICH_TOP"
    export VLLM_ASCEND_SIMLLM_UNMATCHED_STORE_MODE="$SIMLLM_UNMATCHED_STORE_MODE"
    export VLLM_ASCEND_SIMLLM_PROFILE="$SIMLLM_PROFILE"
    export VLLM_ASCEND_SIMLLM_PROFILE_INTERVAL="$SIMLLM_PROFILE_INTERVAL"
    # Both arms use the same log level.  Debug is required for the current
    # SimLLM implementation's per-batch rewrite/cache evidence and therefore
    # must not be enabled only on the optimized arm.
    export VLLM_LOGGING_LEVEL=DEBUG
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="$CURRENT_RUNTIME_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
    "$CURRENT_RUNTIME_PYTHON" -u -m vllm.entrypoints.openai.api_server $server_args
  ) >"$log_file" 2>&1 &

  ACTIVE_SERVER_PID=$!
  ACTIVE_ARM_LABEL=$arm_label
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

capture_device_state() {
  local output_file=$1
  local device_id=${CURRENT_NPU_SMI_DEVICE_ID%%,*}
  if [[ -z "$device_id" ]] || ! command -v npu-smi >/dev/null 2>&1; then
    if [[ "$SIMLLM_OFFICIAL_EVIDENCE" == "1" ]]; then
      echo "official SimLLM evidence requires CURRENT_NPU_SMI_DEVICE_ID and npu-smi" >&2
      return 1
    fi
    return 0
  fi
  {
    echo "logical_device_id=${CURRENT_DEVICE_ID%%,*}"
    echo "physical_npu_smi_device_id=$device_id"
    npu-smi info -t usages -i "$device_id"
    npu-smi info -t proc-mem -i "$device_id"
  } >"$output_file" 2>&1
}

wait_for_device_release() {
  local output_file=$1
  local device_id=${CURRENT_NPU_SMI_DEVICE_ID%%,*}
  if [[ -z "$device_id" ]] || ! command -v npu-smi >/dev/null 2>&1; then
    [[ "$SIMLLM_OFFICIAL_EVIDENCE" != "1" ]]
    return
  fi
  local state=""
  for _ in $(seq 1 30); do
    state=$(npu-smi info -t proc-mem -i "$device_id" 2>&1 || true)
    if grep -q 'No process in device' <<<"$state"; then
      capture_device_state "$output_file"
      return 0
    fi
    sleep 1
  done
  printf '%s\n' "$state" >"$output_file"
  echo "device $device_id still has a process after server cleanup" >&2
  return 1
}

validate_benchmark_result() {
  local label=$1
  local result_file=$2
  local expected_requests=$3

  "$CURRENT_RUNTIME_PYTHON" - "$label" "$result_file" "$expected_requests" <<'PY'
import json
import sys
from pathlib import Path

label, raw_path, expected_text = sys.argv[1:]
payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
expected = int(expected_text)
completed = int(payload.get("completed") or 0)
errors = payload.get("errors") or []
failed = int(payload.get("failed") or 0)
if isinstance(errors, list):
    failed = max(failed, sum(bool(str(item).strip()) for item in errors))
if completed != expected or failed != 0:
    raise SystemExit(
        f"{label}: incomplete benchmark evidence: "
        f"completed={completed}/{expected} failed={failed}"
    )
PY
}

write_prompt_cohort_evidence() {
  local same_spec_file=$1
  local output_file=$2
  local runtime_model=$3
  local client_json=$4

  run_in_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
    env SAME_SPEC_FILE="$same_spec_file" OUTPUT_FILE="$output_file" \
    RUNTIME_MODEL="$runtime_model" CLIENT_JSON="$client_json" \
    "$CURRENT_RUNTIME_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from vllm.benchmarks.datasets import RandomDataset

client = json.loads(os.environ["CLIENT_JSON"])
same_spec = json.loads(Path(os.environ["SAME_SPEC_FILE"]).read_text(encoding="utf-8"))
if client.get("dataset_name") != "random":
    raise SystemExit("official SimLLM prompt evidence only supports deterministic random")

seed = int(client.get("seed", 0))
num_prompts = int(client["num_prompts"])
input_len_value = client.get("input_len", client.get("random_input_len"))
output_len_value = client.get("output_len", client.get("random_output_len"))
if input_len_value is None or output_len_value is None:
    raise SystemExit(
        "official SimLLM prompt evidence requires input/output token lengths"
    )
input_len = int(input_len_value)
output_len = int(output_len_value)
prefix_len = int(client.get("random_prefix_len") or client.get("prefix_len") or 0)
range_ratio = float(client.get("random_range_ratio") or 0)
tokenizer = AutoTokenizer.from_pretrained(
    os.environ["RUNTIME_MODEL"], trust_remote_code=False
)
dataset = RandomDataset(random_seed=seed)
requests = dataset.sample(
    tokenizer=tokenizer,
    num_requests=num_prompts,
    prefix_len=prefix_len,
    range_ratio=range_ratio,
    input_len=input_len,
    output_len=output_len,
)
rows = []
for index, request in enumerate(requests):
    token_ids = tokenizer.encode(request.prompt, add_special_tokens=False)
    rows.append(
        {
            "index": index,
            "prompt_token_ids_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "prompt_tokens": len(token_ids),
            "requested_output_tokens": int(request.expected_output_len or 0),
        }
    )
cohort_payload = {
    "seed": seed,
    "num_prompts": num_prompts,
    "input_len": input_len,
    "output_len": output_len,
    "prefix_len": prefix_len,
    "range_ratio": range_ratio,
    "requests": rows,
}
cohort_sha256 = hashlib.sha256(
    json.dumps(cohort_payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
payload = {
    "schema_version": "simllm-prompt-cohort-evidence/v1",
    "spec_id": same_spec["spec_id"],
    "resolved_spec_hash": same_spec["resolved_spec_hash"],
    "cohort_sha256": cohort_sha256,
    **cohort_payload,
}
Path(os.environ["OUTPUT_FILE"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

write_arm_evidence() {
  local label=$1
  local simllm_enabled=$2
  local did_warmup=$3
  local result_dir=$4
  local same_spec_file=$5
  local cohort_file=$6
  local measured_client_json=$7

  local raw_result_file="$result_dir/raw_benchmark_result.json"
  local server_log="$result_dir/server.stdout.log"
  local rewrite_events=0
  local rewritten_requests=0
  local patch_applied=false
  rewrite_events=$(grep -c 'SimLLM rewrite_scheduler: skipped prefill' "$server_log" 2>/dev/null || true)
  # A disabled baseline legitimately has no rewrite records.  Keep that case at
  # zero without letting grep's no-match status abort this set -e/pipefail runner.
  rewritten_requests=$({ grep 'SimLLM rewrite_scheduler: skipped prefill' "$server_log" 2>/dev/null || true; } \
    | sed -nE 's/.*skipped prefill for ([0-9]+) matched requests.*/\1/p' \
    | awk '{sum += $1} END {print sum + 0}')
  if grep -q 'Sim-LLM patch applied' "$server_log" 2>/dev/null \
    || grep -Eq 'SimLLM worker patch state: .*execute_model=vllm_ascend\.simllm\.patch\.patch_model_runner\._simllm_execute_model .*_model_forward=vllm_ascend\.simllm\.patch\.patch_model_runner\._simllm_model_forward' "$server_log" 2>/dev/null; then
    patch_applied=true
  fi

  LABEL="$label" SIMLLM_ENABLED="$simllm_enabled" DID_WARMUP="$did_warmup" \
  RAW_RESULT_FILE="$raw_result_file" SERVER_LOG="$server_log" \
  SAME_SPEC_FILE="$same_spec_file" COHORT_FILE="$cohort_file" \
  MEASURED_CLIENT_JSON="$measured_client_json" PATCH_APPLIED="$patch_applied" \
  REWRITE_EVENTS="$rewrite_events" REWRITTEN_REQUESTS="$rewritten_requests" \
  CURRENT_BASELINE_ARM_ENGINE="$CURRENT_BASELINE_ARM_ENGINE" \
  CURRENT_SIMLLM_ARM_ENGINE="$CURRENT_SIMLLM_ARM_ENGINE" \
  CURRENT_GIT_COMMIT="$CURRENT_GIT_COMMIT" \
  CURRENT_PLUGIN_GIT_COMMIT="$CURRENT_PLUGIN_GIT_COMMIT" \
  CURRENT_RUNTIME_PYTHON="$CURRENT_RUNTIME_PYTHON" \
  CURRENT_RUNTIME_IMAGE="$CURRENT_RUNTIME_IMAGE" \
  CURRENT_RUNTIME_IMAGE_DIGEST="$CURRENT_RUNTIME_IMAGE_DIGEST" \
  CURRENT_DEVICE_ID="$CURRENT_DEVICE_ID" \
  CURRENT_NPU_SMI_DEVICE_ID="$CURRENT_NPU_SMI_DEVICE_ID" \
  SIMLLM_COSINE_THRESHOLD="$SIMLLM_COSINE_THRESHOLD" \
  SIMLLM_LSH_NUM_BITS="$SIMLLM_LSH_NUM_BITS" \
  SIMLLM_LSH_BATCH_THRESHOLD="$SIMLLM_LSH_BATCH_THRESHOLD" \
  SIMLLM_KV_CACHE_SIZE="$SIMLLM_KV_CACHE_SIZE" \
  SIMLLM_SANDWICH_BOTTOM="$SIMLLM_SANDWICH_BOTTOM" \
  SIMLLM_SANDWICH_TOP="$SIMLLM_SANDWICH_TOP" \
  SIMLLM_UNMATCHED_STORE_MODE="$SIMLLM_UNMATCHED_STORE_MODE" \
  "$CURRENT_RUNTIME_PYTHON" - "$result_dir/arm_evidence.json" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

raw = json.loads(Path(os.environ["RAW_RESULT_FILE"]).read_text(encoding="utf-8"))
same_spec = json.loads(Path(os.environ["SAME_SPEC_FILE"]).read_text(encoding="utf-8"))
cohort = json.loads(Path(os.environ["COHORT_FILE"]).read_text(encoding="utf-8"))
enabled = os.environ["SIMLLM_ENABLED"] == "1"
runtime_packages = {}
for package in (
    "vllm",
    "vllm-ascend",
    "torch",
    "torch-npu",
    "transformers",
    "huggingface-hub",
    "click",
):
    runtime_packages[package] = importlib.metadata.version(package)
payload = {
    "schema_version": "simllm-official-arm-evidence/v1",
    "arm": "simllm-enabled-warm-cache" if enabled else "baseline-disabled",
    "engine": (
        os.environ["CURRENT_SIMLLM_ARM_ENGINE"]
        if enabled else os.environ["CURRENT_BASELINE_ARM_ENGINE"]
    ),
    "simllm_enabled": enabled,
    "warmup_performed": os.environ["DID_WARMUP"] == "1",
    "spec_id": same_spec["spec_id"],
    "resolved_spec_hash": same_spec["resolved_spec_hash"],
    "measured_client_parameters": json.loads(os.environ["MEASURED_CLIENT_JSON"]),
    "prompt_cohort_sha256": cohort["cohort_sha256"],
    "core_commit": os.environ["CURRENT_GIT_COMMIT"],
    "backend_commit": os.environ["CURRENT_PLUGIN_GIT_COMMIT"],
    "runtime": {
        "image": os.environ["CURRENT_RUNTIME_IMAGE"],
        "image_digest": os.environ["CURRENT_RUNTIME_IMAGE_DIGEST"],
        "python_executable": os.environ["CURRENT_RUNTIME_PYTHON"],
        "python_version": platform.python_version(),
        "packages": runtime_packages,
        "cann_home": os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME"),
        "visible_device": os.environ["CURRENT_DEVICE_ID"],
        "physical_npu_smi_device": os.environ["CURRENT_NPU_SMI_DEVICE_ID"],
    },
    "simllm_config": {
        "cosine_threshold": float(os.environ["SIMLLM_COSINE_THRESHOLD"]),
        "lsh_num_bits": int(os.environ["SIMLLM_LSH_NUM_BITS"]),
        "lsh_batch_threshold": int(os.environ["SIMLLM_LSH_BATCH_THRESHOLD"]),
        "kv_cache_size": int(os.environ["SIMLLM_KV_CACHE_SIZE"]),
        "sandwich_bottom": int(os.environ["SIMLLM_SANDWICH_BOTTOM"]),
        "sandwich_top": int(os.environ["SIMLLM_SANDWICH_TOP"]),
        "unmatched_store_mode": os.environ["SIMLLM_UNMATCHED_STORE_MODE"],
    },
    "completed": int(raw.get("completed") or 0),
    "failed": int(raw.get("failed") or 0),
    "patch_applied": os.environ["PATCH_APPLIED"] == "true",
    "rewrite_events": int(os.environ["REWRITE_EVENTS"]),
    "rewritten_requests": int(os.environ["REWRITTEN_REQUESTS"]),
    "hashes": {
        "raw_result_sha256": sha(os.environ["RAW_RESULT_FILE"]),
        "server_log_sha256": sha(os.environ["SERVER_LOG"]),
        "same_spec_sha256": sha(os.environ["SAME_SPEC_FILE"]),
        "prompt_cohort_evidence_sha256": sha(os.environ["COHORT_FILE"]),
        "device_state_before_sha256": sha(str(Path(os.environ["RAW_RESULT_FILE"]).parent / "device_state_before.txt")),
        "device_state_after_sha256": sha(str(Path(os.environ["RAW_RESULT_FILE"]).parent / "device_state_after.txt")),
    },
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

  if [[ "$SIMLLM_OFFICIAL_EVIDENCE" == "1" && "$simllm_enabled" == "1" ]]; then
    if [[ "$patch_applied" != "true" ]]; then
      echo "[$label] SimLLM patch activation evidence is missing" >&2
      return 1
    fi
    if [[ "$SIMLLM_REQUIRE_REWRITE_EVIDENCE" == "1" && "$rewritten_requests" -le 0 ]]; then
      echo "[$label] SimLLM cache/rewrite evidence is missing" >&2
      return 1
    fi
  fi
}

collect_server_info() {
  local result_dir=$1
  local server_log=$2

  {
    echo "=== SimLLM patch state ==="
    grep -i "SimLLM.*patch\|SimLLM.*state\|SimLLM.*rewrite\|SimLLM.*sandwich\|SimLLM.*enabled\|SimLLM.*cache_size" "$server_log" 2>/dev/null | head -40 || true
    echo "=== SimLLM errors ==="
    grep -i "SimLLM.*failed\|SimLLM.*error\|SimLLM.*OOM\|preprocess_from_scheduler\|extract_kv.*failed" "$server_log" 2>/dev/null | head -40 || true
  } > "$result_dir/server_info.txt"
}

export_leaderboard_artifact() {
  local label=$1
  local result_dir=$2
  local same_spec_file=$3
  local raw_result_file=$4
  local artifact_dir=$5
  local scenario=$6
  local arm_engine=$7

  local model
  local model_parameters
  local model_precision
  local model_quantization
  local hardware_vendor
  local hardware_chip_model
  local chip_count
  local node_count
  local input_len
  local output_len

  model=${CURRENT_MODEL_NAME:-$(jq -r '.model' "$SPEC_FILE")}
  model_parameters=${CURRENT_MODEL_PARAMETERS:-$(jq -r '.model_parameters' "$SPEC_FILE")}
  model_precision=${CURRENT_MODEL_PRECISION:-$(jq -r '.model_precision' "$SPEC_FILE")}
  model_quantization=${CURRENT_MODEL_QUANTIZATION:-$(jq -r '.model_quantization // empty' "$SPEC_FILE")}
  hardware_vendor=$(jq -r '.hardware_vendor' "$SPEC_FILE")
  hardware_chip_model=${CURRENT_HARDWARE_CHIP_MODEL:-$(jq -r '.hardware_chip_model' "$SPEC_FILE")}
  chip_count=$(jq -r '.chip_count' "$SPEC_FILE")
  node_count=$(jq -r '.node_count' "$SPEC_FILE")
  input_len=$(jq -r '.client_parameters.input_len // empty' "$SPEC_FILE")
  output_len=$(jq -r '.client_parameters.output_len // empty' "$SPEC_FILE")

  EXPORT_ARGS=(
    "$scenario"
    --benchmark-result-file "$raw_result_file"
    --constraints-file "$CONSTRAINTS_FILE"
    --same-spec-file "$same_spec_file"
    --output-dir "$artifact_dir"
    --run-id "$label-$RUN_TIMESTAMP"
    --engine "$arm_engine"
    --engine-version "${CURRENT_ENGINE_VERSION:-$(maybe_git_describe "$CURRENT_VLLM_HUST_REPO" "$CURRENT_GIT_COMMIT")}"
    --core-version "${CURRENT_CORE_VERSION:-${CURRENT_ENGINE_VERSION:-$(maybe_git_describe "$CURRENT_VLLM_HUST_REPO" "$CURRENT_GIT_COMMIT")}}"
    --backend-version "${CURRENT_BACKEND_VERSION:-$(maybe_git_describe "$CURRENT_VLLM_ASCEND_HUST_REPO" "$CURRENT_PLUGIN_GIT_COMMIT")}"
    --model-name "$model"
    --model-parameters "$model_parameters"
    --model-precision "$model_precision"
    --hardware-vendor "$hardware_vendor"
    --hardware-chip-model "$hardware_chip_model"
    --chip-count "$chip_count"
    --node-count "$node_count"
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

  append_export_arg_if_present --quantization "$model_quantization"
  append_export_arg_if_present --input-length "$input_len"
  append_export_arg_if_present --output-length "$output_len"

  run_in_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
    "$CURRENT_RUNTIME_PYTHON" -m vllm_hust_benchmark.cli export-leaderboard-artifact \
    "${EXPORT_ARGS[@]}"
}

run_experiment() {
  local label=$1
  local simllm_enabled=$2
  local do_warmup=$3
  local result_dir=$4
  local port=$5

  local same_spec_file="$result_dir/resolved_same_spec.json"
  local raw_result_file="$result_dir/raw_benchmark_result.json"
  local server_log="$result_dir/server.stdout.log"
  local artifact_dir="$result_dir/submission"
  local scenario
  local benchmark_type
  local runtime_model
  local server_host
  local client_host
  local client_port
  local base_client_json
  local measure_client_json
  local measure_args
  local warmup_json
  local warmup_args
  local server_args
  local expected_requests
  local prompt_cohort_file
  local arm_engine
  local status

  mkdir -p "$result_dir" "$artifact_dir"
  capture_device_state "$result_dir/device_state_before.txt"
  if [[ "$SIMLLM_OFFICIAL_EVIDENCE" == "1" ]] \
    && ! grep -q 'No process in device' "$result_dir/device_state_before.txt"; then
    echo "target device is not idle before $label" >&2
    return 1
  fi

  scenario=$(jq -r '.scenario' "$SPEC_FILE")
  benchmark_type=$(resolve_benchmark_type "$scenario")
  if [[ "$benchmark_type" != "serve" ]]; then
    echo "This warm-cache runner only supports serve benchmarks; $scenario is $benchmark_type." >&2
    return 2
  fi

  runtime_model="${CURRENT_MODEL_PATH:-$(jq -r '.model' "$SPEC_FILE")}"

  echo ""
  echo "============================================================"
  echo "  $label"
  echo "  SimLLM=$simllm_enabled Warmup=$do_warmup Port=$port"
  echo "============================================================"

  resolve_spec "$SPEC_FILE" "$same_spec_file" "$runtime_model" "$port"
  server_host=$(jq -r '.resolved_server_parameters.host' "$same_spec_file")
  client_host=$(jq -r '.resolved_client_parameters.host' "$same_spec_file")
  client_port=$(jq -r '.resolved_client_parameters.port' "$same_spec_file")
  server_args=$(json2args "$(jq -c '.resolved_server_parameters | del(.disable_log_requests)' "$same_spec_file")")
  base_client_json=$(normalized_client_parameters_json "$same_spec_file")
  measure_client_json=$(client_json_with_seed "$base_client_json" "$SIMLLM_MEASURE_SEED")
  measure_args=$(json2args "$measure_client_json")
  expected_requests=$(echo "$measure_client_json" | jq -er '.num_prompts')
  prompt_cohort_file="$result_dir/prompt_cohort_evidence.json"
  arm_engine="$CURRENT_BASELINE_ARM_ENGINE"
  if [[ "$simllm_enabled" == "1" ]]; then
    arm_engine="$CURRENT_SIMLLM_ARM_ENGINE"
  fi

  write_prompt_cohort_evidence \
    "$same_spec_file" "$prompt_cohort_file" "$runtime_model" "$measure_client_json"

  assert_port_available "$client_port"

  echo "[$label] resolved spec: $same_spec_file"
  echo "[$label] runtime model: $runtime_model"
  echo "[$label] endpoint: ${client_host}:${client_port}"
  echo "[$label] measured seed: ${SIMLLM_MEASURE_SEED:-<benchmark-default>}"

  start_server "$simllm_enabled" "$server_args" "$server_log" "$label"
  echo "$ACTIVE_SERVER_PID" > "$result_dir/server.pid"

  if ! wait_for_server "$client_host" "$client_port" "$server_log"; then
    cleanup_active_server
    return 1
  fi

  if [[ "$do_warmup" == "1" ]]; then
    warmup_json=$(warmup_client_json "$base_client_json")
    warmup_args=$(json2args "$warmup_json")
    echo "[$label] warm-cache passes: $SIMLLM_WARMCACHE_PASSES"
    echo "[$label] warm-cache seed: $SIMLLM_WARMCACHE_SEED"

    for pass in $(seq 1 "$SIMLLM_WARMCACHE_PASSES"); do
      echo "[$label] warm-cache pass ${pass}/${SIMLLM_WARMCACHE_PASSES}"
      set +e
      run_in_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
        "$CURRENT_RUNTIME_PYTHON" "$VLLM_CLI_COMPAT" bench serve \
        --save-result \
        --result-dir "$result_dir" \
        --result-filename "warmup_pass_${pass}.json" \
        $warmup_args \
        > "$result_dir/warmup_pass_${pass}.log" 2>&1
      status=$?
      set -e
      if [[ "$status" -ne 0 ]]; then
        echo "[$label] warm-cache pass ${pass} failed; see $result_dir/warmup_pass_${pass}.log" >&2
        cleanup_active_server
        return "$status"
      fi
      validate_benchmark_result \
        "$label warm-cache pass ${pass}" \
        "$result_dir/warmup_pass_${pass}.json" \
        "$(echo "$warmup_json" | jq -er '.num_prompts')"
    done

    if (( SIMLLM_WARMCACHE_PAUSE_SECONDS > 0 )); then
      sleep "$SIMLLM_WARMCACHE_PAUSE_SECONDS"
    fi
  fi

  echo "[$label] measured benchmark"
  set +e
  run_in_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
    "$CURRENT_RUNTIME_PYTHON" "$VLLM_CLI_COMPAT" bench serve \
    --save-result \
    --result-dir "$result_dir" \
    --result-filename "$(basename "$raw_result_file")" \
    $measure_args
  status=$?
  set -e

  collect_server_info "$result_dir" "$server_log"
  cleanup_active_server

  if [[ "$status" -ne 0 ]]; then
    echo "[$label] measured benchmark failed." >&2
    return "$status"
  fi

  wait_for_device_release "$result_dir/device_state_after.txt"

  validate_benchmark_result "$label measured pass" "$raw_result_file" "$expected_requests"
  write_arm_evidence \
    "$label" "$simllm_enabled" "$do_warmup" "$result_dir" \
    "$same_spec_file" "$prompt_cohort_file" "$measure_client_json"

  if port_has_listener "$client_port"; then
    echo "[$label] service port ${client_port} remains occupied after cleanup" >&2
    return 1
  fi

  export_leaderboard_artifact \
    "$label" "$result_dir" "$same_spec_file" "$raw_result_file" \
    "$artifact_dir" "$scenario" "$arm_engine"
  echo "[$label] done: $result_dir"
}

mkdir -p "$RESULT_DIR" "$CURRENT_VLLM_CACHE_ROOT"

if [[ "$RUN_BASELINE" == "1" ]]; then
  run_experiment "baseline" "0" "0" "$BASELINE_DIR" "$BASELINE_SERVER_PORT"
fi

if [[ "$RUN_SIMLLM" == "1" ]]; then
  run_experiment "simllm-warm-cache" "1" "1" "$SIMLLM_DIR" "$SIMLLM_SERVER_PORT"
fi

if [[ "$RUN_BASELINE" == "1" && "$RUN_SIMLLM" == "1" ]]; then
  BASELINE_EVIDENCE="$BASELINE_DIR/arm_evidence.json"
  SIMLLM_EVIDENCE="$SIMLLM_DIR/arm_evidence.json"
  if [[ ! -f "$BASELINE_EVIDENCE" || ! -f "$SIMLLM_EVIDENCE" ]]; then
    echo "paired SimLLM arm evidence is incomplete" >&2
    exit 1
  fi
  BASELINE_EVIDENCE="$BASELINE_EVIDENCE" SIMLLM_EVIDENCE="$SIMLLM_EVIDENCE" \
  PAIR_OUTPUT="$RESULT_DIR/paired_protocol_evidence.json" \
  "$CURRENT_RUNTIME_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def load(name: str) -> dict:
    return json.loads(Path(os.environ[name]).read_text(encoding="utf-8"))

def sha(name: str) -> str:
    return hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()

baseline = load("BASELINE_EVIDENCE")
simllm = load("SIMLLM_EVIDENCE")
if baseline["simllm_enabled"] or baseline["warmup_performed"]:
    raise SystemExit("baseline arm incorrectly enables SimLLM or warmup")
if not simllm["simllm_enabled"] or not simllm["warmup_performed"]:
    raise SystemExit("SimLLM arm is missing enablement or warmup evidence")
for field in (
    "spec_id",
    "resolved_spec_hash",
    "measured_client_parameters",
    "prompt_cohort_sha256",
    "core_commit",
    "backend_commit",
    "runtime",
    "simllm_config",
    "completed",
):
    if baseline[field] != simllm[field]:
        raise SystemExit(f"paired SimLLM arms differ at {field}")
if baseline["failed"] or simllm["failed"]:
    raise SystemExit("paired SimLLM result contains failed requests")
if simllm["engine"] == baseline["engine"]:
    raise SystemExit("paired SimLLM arms require distinct leaderboard engine labels")
if not simllm["patch_applied"] or simllm["rewritten_requests"] <= 0:
    raise SystemExit("SimLLM arm lacks positive patch/rewrite evidence")
payload = {
    "schema_version": "simllm-official-paired-protocol/v1",
    "spec_id": baseline["spec_id"],
    "resolved_spec_hash": baseline["resolved_spec_hash"],
    "prompt_cohort_sha256": baseline["prompt_cohort_sha256"],
    "core_commit": baseline["core_commit"],
    "backend_commit": baseline["backend_commit"],
    "allowed_arm_differences": ["engine", "simllm_enabled", "warmup_performed"],
    "exact_measured_setting_match": True,
    "zero_failed_requests": True,
    "baseline": {
        "engine": baseline["engine"],
        "evidence_sha256": sha("BASELINE_EVIDENCE"),
    },
    "simllm": {
        "engine": simllm["engine"],
        "evidence_sha256": sha("SIMLLM_EVIDENCE"),
        "rewrite_events": simllm["rewrite_events"],
        "rewritten_requests": simllm["rewritten_requests"],
    },
}
Path(os.environ["PAIR_OUTPUT"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
fi

echo ""
echo "============================================================"
echo "  Comparison: Baseline vs SimLLM Warm-Cache"
echo "============================================================"

BL_LEADERBOARD="$BASELINE_DIR/submission/run_leaderboard.json"
SL_LEADERBOARD="$SIMLLM_DIR/submission/run_leaderboard.json"

if [[ -f "$BL_LEADERBOARD" && -f "$SL_LEADERBOARD" ]]; then
  run_in_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
    "$CURRENT_RUNTIME_PYTHON" -m vllm_hust_benchmark.perfgate compare \
    --current "$SL_LEADERBOARD" \
    --baseline "$BL_LEADERBOARD" \
    --report-file "$RESULT_DIR/perfgate_report.md" \
    --mode report || true
  echo "Report: $RESULT_DIR/perfgate_report.md"
else
  echo "Skipping comparison because one leaderboard artifact is missing:"
  echo "  Baseline: $BL_LEADERBOARD"
  echo "  SimLLM:   $SL_LEADERBOARD"
fi

echo ""
echo "Outputs:"
echo "  Baseline:    $BASELINE_DIR"
echo "  SimLLM warm: $SIMLLM_DIR"
