#!/bin/bash
set -euo pipefail

# SimLLM prefill-heavy warm-cache runner for ShareGPT-backed official specs.
#
# This script accepts either the official ShareGPT throughput spec or the
# official ShareGPT online spec and rewrites the measurement workload into a
# temporary prefill-heavy random workload:
#   - longer prefill: 4096 input tokens
#   - short decode: 32 output tokens
#   - higher request count: 200 prompts
#   - high arrival rate for serve benchmarks: request_rate=4
#   - greedy decoding on serve benchmarks to avoid sampling-path instability
#
# Behavior is benchmark-type aware:
#   - sharegpt-online -> reuse the existing warm-cache serve runner
#   - sharegpt-throughput -> run throughput A/B locally and use benchmark warmup
#
# Example:
#   bash scripts/run_simllm_sharegpt_prefill_heavy_warm_cache.sh \
#     docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-throughput-qwen25-14b-910b2.json
#
#   bash scripts/run_simllm_sharegpt_prefill_heavy_warm_cache.sh \
#     docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-910b2.json

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=${VLLM_HUST_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}
BENCHMARK_REPO=${VLLM_HUST_BENCHMARK_REPO:-"$WORKSPACE_ROOT/vllm-hust-benchmark"}
SERVE_WARM_CACHE_RUNNER=${SERVE_WARM_CACHE_RUNNER:-"$SCRIPT_DIR/run_simllm_random_online_warm_cache.sh"}
VLLM_CLI_COMPAT=${VLLM_CLI_COMPAT:-"$BENCHMARK_REPO/scripts/run_vllm_cli_compat.py"}

DEFAULT_BASE_SPEC_FILE="$REPO_ROOT/docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-throughput-qwen25-14b-910b2.json"
BASE_SPEC_FILE=${1:-${BASE_SPEC_FILE:-$DEFAULT_BASE_SPEC_FILE}}

CURRENT_RUNTIME_CWD=${CURRENT_RUNTIME_CWD:-"/tmp"}
CURRENT_VLLM_HUST_REPO=${CURRENT_VLLM_HUST_REPO:-"$WORKSPACE_ROOT/vllm-hust"}
CURRENT_VLLM_ASCEND_HUST_REPO=${CURRENT_VLLM_ASCEND_HUST_REPO:-"$WORKSPACE_ROOT/vllm-ascend-hust"}
CURRENT_ENV_PREFIX=${CURRENT_ENV_PREFIX:-"/root/miniconda3/envs/vllm-hust-dev"}
CURRENT_RUNTIME_PYTHON=${CURRENT_RUNTIME_PYTHON:-"$CURRENT_ENV_PREFIX/bin/python"}
CURRENT_VLLM_CACHE_ROOT=${CURRENT_VLLM_CACHE_ROOT:-"$REPO_ROOT/.cache/simllm-prefill-heavy"}

ASCEND_TOOLKIT_SET_ENV=${ASCEND_TOOLKIT_SET_ENV:-"/usr/local/Ascend/ascend-toolkit/set_env.sh"}
ASCEND_ATB_SET_ENV=${ASCEND_ATB_SET_ENV:-"/usr/local/Ascend/nnal/atb/set_env.sh"}
ASCEND_ATB_CXX_ABI=${ASCEND_ATB_CXX_ABI:-"1"}

CURRENT_DTYPE=${CURRENT_DTYPE:-}
CURRENT_MODEL_NAME=${CURRENT_MODEL_NAME:-}
CURRENT_MODEL_PARAMETERS=${CURRENT_MODEL_PARAMETERS:-}
CURRENT_MODEL_PRECISION=${CURRENT_MODEL_PRECISION:-}
CURRENT_MODEL_QUANTIZATION=${CURRENT_MODEL_QUANTIZATION:-}
CURRENT_HARDWARE_CHIP_MODEL=${CURRENT_HARDWARE_CHIP_MODEL:-}
CURRENT_SERVER_HOST=${CURRENT_SERVER_HOST:-}
CURRENT_SERVER_PORT=${CURRENT_SERVER_PORT:-8000}
CURRENT_CLIENT_HOST=${CURRENT_CLIENT_HOST:-}
CURRENT_CLIENT_PORT=${CURRENT_CLIENT_PORT:-$CURRENT_SERVER_PORT}

CURRENT_DATA_SOURCE=${CURRENT_DATA_SOURCE:-"vllm-ascend-hust-ci-simllm-prefill-heavy"}
CURRENT_SUBMITTER=${CURRENT_SUBMITTER:-"simllm-prefill-heavy"}
CURRENT_ENGINE=${CURRENT_ENGINE:-"vllm-hust"}
CURRENT_BASELINE_ENGINE=${CURRENT_BASELINE_ENGINE:-"vllm"}
CURRENT_GITHUB_REPOSITORY=${CURRENT_GITHUB_REPOSITORY:-"vLLM-HUST/vllm-hust"}
CURRENT_GITHUB_REF=${CURRENT_GITHUB_REF:-$(git -C "$CURRENT_VLLM_HUST_REPO" branch --show-current 2>/dev/null || echo main)}
CURRENT_GIT_COMMIT=${CURRENT_GIT_COMMIT:-$(git -C "$CURRENT_VLLM_HUST_REPO" rev-parse HEAD 2>/dev/null || true)}
CURRENT_PLUGIN_ENGINE=${CURRENT_PLUGIN_ENGINE:-"vllm-ascend-hust"}
CURRENT_PLUGIN_GITHUB_REPOSITORY=${CURRENT_PLUGIN_GITHUB_REPOSITORY:-"vLLM-HUST/vllm-ascend-hust"}
CURRENT_PLUGIN_GITHUB_REF=${CURRENT_PLUGIN_GITHUB_REF:-$(git -C "$CURRENT_VLLM_ASCEND_HUST_REPO" branch --show-current 2>/dev/null || echo main)}
CURRENT_PLUGIN_GIT_COMMIT=${CURRENT_PLUGIN_GIT_COMMIT:-$(git -C "$CURRENT_VLLM_ASCEND_HUST_REPO" rev-parse HEAD 2>/dev/null || true)}

RESULT_DIR=${RESULT_DIR:-}
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

SIMLLM_PREFILL_HEAVY_INPUT_LEN=${SIMLLM_PREFILL_HEAVY_INPUT_LEN:-4096}
SIMLLM_PREFILL_HEAVY_OUTPUT_LEN=${SIMLLM_PREFILL_HEAVY_OUTPUT_LEN:-32}
SIMLLM_PREFILL_HEAVY_NUM_PROMPTS=${SIMLLM_PREFILL_HEAVY_NUM_PROMPTS:-200}
SIMLLM_PREFILL_HEAVY_REQUEST_RATE=${SIMLLM_PREFILL_HEAVY_REQUEST_RATE:-4}
SIMLLM_PREFILL_HEAVY_MAX_CONCURRENCY=${SIMLLM_PREFILL_HEAVY_MAX_CONCURRENCY:-4}
SIMLLM_PREFILL_HEAVY_NUM_WARMUPS=${SIMLLM_PREFILL_HEAVY_NUM_WARMUPS:-32}
SIMLLM_PREFILL_HEAVY_CLIENT_TEMPERATURE=${SIMLLM_PREFILL_HEAVY_CLIENT_TEMPERATURE:-0}

SIMLLM_WARMCACHE_PASSES=${SIMLLM_WARMCACHE_PASSES:-1}
SIMLLM_WARMCACHE_SEED=${SIMLLM_WARMCACHE_SEED:-0}
SIMLLM_MEASURE_SEED=${SIMLLM_MEASURE_SEED:-$SIMLLM_WARMCACHE_SEED}
SIMLLM_WARMCACHE_PAUSE_SECONDS=${SIMLLM_WARMCACHE_PAUSE_SECONDS:-5}

CURRENT_RUNTIME_SOURCE_PYTHONPATH="$BENCHMARK_REPO/src:$CURRENT_VLLM_ASCEND_HUST_REPO:$CURRENT_VLLM_HUST_REPO"
CURRENT_RUNTIME_PYTHONPATH="${CURRENT_RUNTIME_SOURCE_PYTHONPATH}${CURRENT_RUNTIME_PYTHONPATH:+:$CURRENT_RUNTIME_PYTHONPATH}"

disable_proxy_env() {
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1}"
  export no_proxy="$NO_PROXY"
}

usage() {
  echo "Usage: $0 [sharegpt-official-spec.json]" >&2
}

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
    if [[ -d "$CURRENT_ENV_PREFIX/lib" ]]; then
      export LD_LIBRARY_PATH="$CURRENT_ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    fi
    export VLLM_CACHE_ROOT="$CURRENT_VLLM_CACHE_ROOT"
    PYTHONPATH="$pythonpath_prefix${PYTHONPATH:+:$PYTHONPATH}" "$@"
  )
}

json2args() {
  local json_string=$1
  JSON2ARGS_PAYLOAD="$json_string" "$CURRENT_RUNTIME_PYTHON" - <<'PY'
import json
import os

payload = json.loads(os.environ["JSON2ARGS_PAYLOAD"])
args: list[str] = []
for key, value in payload.items():
    if value is None or value is False or value == "":
        continue
    flag = "--" + key.replace("_", "-")
    if value is True:
        args.append(flag)
        continue
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    else:
        rendered = str(value)
    args.extend([flag, rendered])
print(" ".join(args))
PY
}

benchmark_type_for_spec() {
  local spec_file=$1
  run_in_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
    env SPEC_FILE="$spec_file" \
    "$CURRENT_RUNTIME_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

from vllm_hust_benchmark.registry import get_scenario

payload = json.loads(Path(os.environ["SPEC_FILE"]).read_text(encoding="utf-8"))
print(get_scenario(payload["scenario"]).benchmark_type)
PY
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

build_throughput_client_json() {
  local same_spec_file=$1
  local num_warmups=$2
  run_in_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
    env SAME_SPEC_FILE="$same_spec_file" \
    BENCHMARK_TYPE=throughput \
    NUM_WARMUPS="$num_warmups" \
    CURRENT_VLLM_WORKTREE="$CURRENT_VLLM_HUST_REPO" \
    BENCHMARK_REPO_ROOT="$BENCHMARK_REPO" \
    CURRENT_BENCHMARK_DATASET_ROOT="${CURRENT_BENCHMARK_DATASET_ROOT:-}" \
    "$CURRENT_RUNTIME_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

from vllm_hust_benchmark.official_runtime_inputs import (
    normalize_offline_benchmark_parameters,
)

payload = json.loads(Path(os.environ["SAME_SPEC_FILE"]).read_text(encoding="utf-8"))
client = normalize_offline_benchmark_parameters(
    payload["resolved_client_parameters"],
    payload["resolved_server_parameters"],
    benchmark_type=os.environ["BENCHMARK_TYPE"],
    ready_check_timeout_sec=0,
    vllm_worktree=os.environ.get("CURRENT_VLLM_WORKTREE"),
    benchmark_repo=os.environ.get("BENCHMARK_REPO_ROOT"),
    dataset_cache_root=os.environ.get("CURRENT_BENCHMARK_DATASET_ROOT") or None,
    force_eager=False,
)
num_warmups = int(os.environ.get("NUM_WARMUPS") or 0)
if num_warmups > 0:
    client["num_warmups"] = num_warmups
print(json.dumps(client, separators=(",", ":"), ensure_ascii=True))
PY
}

build_temp_spec() {
  local base_spec_file=$1
  local output_file=$2
  local benchmark_type=$3
  local scenario_tag=$4

  jq \
    --arg benchmark_type "$benchmark_type" \
    --arg scenario_tag "$scenario_tag" \
    --arg endpoint "/v1/completions" \
    --argjson input_len "$SIMLLM_PREFILL_HEAVY_INPUT_LEN" \
    --argjson output_len "$SIMLLM_PREFILL_HEAVY_OUTPUT_LEN" \
    --argjson num_prompts "$SIMLLM_PREFILL_HEAVY_NUM_PROMPTS" \
    --argjson request_rate "$SIMLLM_PREFILL_HEAVY_REQUEST_RATE" \
    --argjson max_concurrency "$SIMLLM_PREFILL_HEAVY_MAX_CONCURRENCY" \
    --argjson num_warmups "$SIMLLM_PREFILL_HEAVY_NUM_WARMUPS" '
      .id = ("simllm-" + $scenario_tag + "-prefill-heavy-warm-cache")
      | .label = ("SimLLM " + $scenario_tag + " prefill-heavy warm-cache")
      | .client_parameters = ((.client_parameters // {})
          + {
              backend: "vllm",
              dataset_name: "random",
              input_len: $input_len,
              output_len: $output_len,
              num_prompts: $num_prompts
            })
      | .client_parameters |= del(.dataset_path)
      | if $benchmark_type == "serve" then
          .client_parameters += {
            endpoint: $endpoint,
            request_rate: $request_rate,
            max_concurrency: $max_concurrency
          }
          | .client_parameters |= del(.num_warmups)
        else
          .client_parameters += {num_warmups: $num_warmups}
          | .client_parameters |= del(.request_rate, .endpoint, .max_concurrency)
        end
    ' "$base_spec_file" > "$output_file"
}

compare_throughput_results() {
  local baseline_file=$1
  local simllm_file=$2
  local output_file=$3

  jq -n \
    --slurpfile baseline "$baseline_file" \
    --slurpfile simllm "$simllm_file" '
      {
        baseline: $baseline[0],
        simllm: $simllm[0],
        delta: {
          requests_per_second: ($simllm[0].requests_per_second - $baseline[0].requests_per_second),
          tokens_per_second: ($simllm[0].tokens_per_second - $baseline[0].tokens_per_second),
          requests_per_second_ratio: ($simllm[0].requests_per_second / $baseline[0].requests_per_second),
          tokens_per_second_ratio: ($simllm[0].tokens_per_second / $baseline[0].tokens_per_second)
        }
      }
    ' > "$output_file"
}

run_throughput_case() {
  local label=$1
  local simllm_enabled=$2
  local warmup_prompts=$3
  local result_dir=$4
  local spec_file=$5
  local runtime_model=$6
  local port=$7

  local same_spec_file="$result_dir/resolved_same_spec.json"
  local raw_result_file="$result_dir/raw_benchmark_result.json"
  local log_file="$result_dir/throughput.log"
  local client_json
  local client_args
  local status

  mkdir -p "$result_dir"

  resolve_spec "$spec_file" "$same_spec_file" "$runtime_model" "$port"
  client_json=$(build_throughput_client_json "$same_spec_file" "$warmup_prompts")
  client_args=$(json2args "$client_json")

  {
    echo "[$label] resolved spec: $same_spec_file"
    echo "[$label] runtime model: $runtime_model"
    echo "[$label] warmup prompts: $warmup_prompts"
    echo "[$label] benchmark args: $client_json"
  } | tee "$result_dir/run_info.txt"

  set +e
  run_in_runtime "$CURRENT_RUNTIME_PYTHONPATH" \
    env \
      VLLM_ASCEND_SIMLLM_ENABLED="$simllm_enabled" \
      VLLM_ASCEND_SIMLLM_COSINE_THRESHOLD="$SIMLLM_COSINE_THRESHOLD" \
      VLLM_ASCEND_SIMLLM_LSH_NUM_BITS="$SIMLLM_LSH_NUM_BITS" \
      VLLM_ASCEND_SIMLLM_LSH_BATCH_THRESHOLD="$SIMLLM_LSH_BATCH_THRESHOLD" \
      VLLM_ASCEND_SIMLLM_KV_CACHE_SIZE="$SIMLLM_KV_CACHE_SIZE" \
      VLLM_ASCEND_SIMLLM_SANDWICH_BOTTOM="$SIMLLM_SANDWICH_BOTTOM" \
      VLLM_ASCEND_SIMLLM_SANDWICH_TOP="$SIMLLM_SANDWICH_TOP" \
      VLLM_ASCEND_SIMLLM_UNMATCHED_STORE_MODE="$SIMLLM_UNMATCHED_STORE_MODE" \
      VLLM_ASCEND_SIMLLM_PROFILE="$SIMLLM_PROFILE" \
      VLLM_ASCEND_SIMLLM_PROFILE_INTERVAL="$SIMLLM_PROFILE_INTERVAL" \
      "$CURRENT_RUNTIME_PYTHON" "$VLLM_CLI_COMPAT" bench throughput \
      --output-json "$raw_result_file" \
      $client_args \
      > "$log_file" 2>&1
  status=$?
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "[$label] throughput benchmark failed; see $log_file" >&2
    tail -n 80 "$log_file" >&2 || true
    return "$status"
  fi

  jq '.' "$raw_result_file" > "$result_dir/raw_benchmark_result.pretty.json"
}

if [[ -z "$BASE_SPEC_FILE" ]]; then
  usage
  exit 2
fi

if [[ ! -d "$BENCHMARK_REPO" ]]; then
  echo "Benchmark repo not found: $BENCHMARK_REPO" >&2
  exit 2
fi
if [[ ! -d "$CURRENT_VLLM_HUST_REPO" ]]; then
  echo "vllm repo not found: $CURRENT_VLLM_HUST_REPO" >&2
  exit 2
fi
if [[ ! -d "$CURRENT_VLLM_ASCEND_HUST_REPO" ]]; then
  echo "vllm-ascend-hust repo not found: $CURRENT_VLLM_ASCEND_HUST_REPO" >&2
  exit 2
fi

BASE_SPEC_FILE=$(cd "$(dirname "$BASE_SPEC_FILE")" && pwd)/$(basename "$BASE_SPEC_FILE")
if [[ ! -f "$BASE_SPEC_FILE" ]]; then
  echo "Base spec file not found: $BASE_SPEC_FILE" >&2
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

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required for this script." >&2
  exit 2
fi

SCENARIO=$(jq -r '.scenario' "$BASE_SPEC_FILE")
BENCHMARK_TYPE=$(benchmark_type_for_spec "$BASE_SPEC_FILE")
SPEC_TAG=$(printf '%s' "$SCENARIO" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')

if [[ -z "$RESULT_DIR" ]]; then
  RESULT_DIR="$REPO_ROOT/.benchmarks/simllm-${SPEC_TAG}-prefill-heavy-warm-cache"
fi
BASELINE_DIR="$RESULT_DIR/baseline-disabled"
SIMLLM_DIR="$RESULT_DIR/enabled-warm-cache"
TEMP_SPEC_FILE="$RESULT_DIR/prefill-heavy-same-spec.json"

mkdir -p "$RESULT_DIR"
build_temp_spec "$BASE_SPEC_FILE" "$TEMP_SPEC_FILE" "$BENCHMARK_TYPE" "$SPEC_TAG"

echo "[simllm-prefill-heavy] base spec: $BASE_SPEC_FILE"
echo "[simllm-prefill-heavy] temp spec: $TEMP_SPEC_FILE"
echo "[simllm-prefill-heavy] benchmark type: $BENCHMARK_TYPE"
echo "[simllm-prefill-heavy] result dir: $RESULT_DIR"
echo "[simllm-prefill-heavy] workload: input=${SIMLLM_PREFILL_HEAVY_INPUT_LEN} output=${SIMLLM_PREFILL_HEAVY_OUTPUT_LEN} prompts=${SIMLLM_PREFILL_HEAVY_NUM_PROMPTS}"

runtime_model="${CURRENT_MODEL_PATH:-$(jq -r '.model' "$BASE_SPEC_FILE")}"
if [[ "$runtime_model" == "null" || -z "$runtime_model" ]]; then
  echo "Unable to determine runtime model. Set CURRENT_MODEL_PATH explicitly." >&2
  exit 2
fi

if [[ "$BENCHMARK_TYPE" == "serve" ]]; then
  export RESULT_DIR
  export CURRENT_DATA_SOURCE
  export CURRENT_CLIENT_TEMPERATURE="$SIMLLM_PREFILL_HEAVY_CLIENT_TEMPERATURE"
  export SIMLLM_WARMCACHE_SEED
  export SIMLLM_MEASURE_SEED
  export SIMLLM_WARMCACHE_PASSES
  export SIMLLM_WARMCACHE_PAUSE_SECONDS
  export SIMLLM_WARMCACHE_REQUEST_RATE="$SIMLLM_PREFILL_HEAVY_REQUEST_RATE"
  export SIMLLM_WARMCACHE_NUM_PROMPTS="$SIMLLM_PREFILL_HEAVY_NUM_PROMPTS"
  exec "$SERVE_WARM_CACHE_RUNNER" "$TEMP_SPEC_FILE"
fi

run_throughput_case "baseline-disabled" 0 0 "$BASELINE_DIR" "$TEMP_SPEC_FILE" "$runtime_model" 8000
run_throughput_case "enabled-warm-cache" 1 "$SIMLLM_PREFILL_HEAVY_NUM_WARMUPS" "$SIMLLM_DIR" "$TEMP_SPEC_FILE" "$runtime_model" 8000

compare_throughput_results \
  "$BASELINE_DIR/raw_benchmark_result.json" \
  "$SIMLLM_DIR/raw_benchmark_result.json" \
  "$RESULT_DIR/comparison.json"

echo ""
echo "===== SimLLM throughput comparison ====="
jq -r '
  "baseline requests/s: \(.baseline.requests_per_second)\n" +
  "simllm   requests/s: \(.simllm.requests_per_second)\n" +
  "baseline tokens/s:   \(.baseline.tokens_per_second)\n" +
  "simllm   tokens/s:   \(.simllm.tokens_per_second)\n" +
  "request ratio:       \(.delta.requests_per_second_ratio)\n" +
  "token ratio:         \(.delta.tokens_per_second_ratio)"
' "$RESULT_DIR/comparison.json"
