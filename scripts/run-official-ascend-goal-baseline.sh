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
ASCEND_TOOLKIT_SET_ENV=${ASCEND_TOOLKIT_SET_ENV:-"/usr/local/Ascend/ascend-toolkit/set_env.sh"}
ASCEND_ATB_SET_ENV=${ASCEND_ATB_SET_ENV:-"/usr/local/Ascend/nnal/atb/set_env.sh"}
ASCEND_ATB_CXX_ABI=${ASCEND_ATB_CXX_ABI:-"1"}
GOAL_BASELINE_ENV_PREFIX=${GOAL_BASELINE_ENV_PREFIX:-}
RESULT_DIR=${RESULT_DIR:-"$REPO_ROOT/.benchmarks/official-ascend-goal-baseline"}
RUN_ID=${RUN_ID:-"official-ascend-jan-2026-$(date -u +%Y%m%dT%H%M%SZ)"}

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
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" || true
    wait "$SERVER_PID" || true
  fi
}

trap kill_server EXIT

ensure_worktree "$OFFICIAL_VLLM_REPO" "$OFFICIAL_VLLM_WORKTREE" "v0.11.0"
ensure_worktree "$OFFICIAL_VLLM_ASCEND_REPO" "$OFFICIAL_VLLM_ASCEND_WORKTREE" "v0.11.0"

OFFICIAL_RUNTIME_PYTHONPATH="$OFFICIAL_VLLM_ASCEND_WORKTREE:$OFFICIAL_VLLM_WORKTREE"

mkdir -p "$RESULT_DIR"
mkdir -p "$OFFICIAL_VLLM_CACHE_ROOT"

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
SERVER_HOST=$(jq -r '.server_parameters.host' "$SPEC_FILE")
SERVER_PORT=$(jq -r '.server_parameters.port' "$SPEC_FILE")
CLIENT_HOST=$(jq -r '.client_parameters.host' "$SPEC_FILE")
CLIENT_PORT=$(jq -r '.client_parameters.port' "$SPEC_FILE")
INPUT_LEN=$(jq -r '.client_parameters.input_len' "$SPEC_FILE")
OUTPUT_LEN=$(jq -r '.client_parameters.output_len' "$SPEC_FILE")

BENCHMARK_SERVER_PORT="$SERVER_PORT" \
PREPARE_BENCHMARK_ADMISSION_ONLY=1 \
ENV_PREFIX="$GOAL_BASELINE_ENV_PREFIX" \
VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
bash "$PREPARE_SCRIPT"

RUNTIME_MODEL="$MODEL"
if cached_model_path=$(resolve_runtime_model); then
  RUNTIME_MODEL="$cached_model_path"
fi

SERVER_ARGS=$(json2args "$(jq -c --arg model "$RUNTIME_MODEL" '
  .server_parameters + {model: $model}
  | if has("enforce_eager") then . else . + {enforce_eager: ""} end
' "$SPEC_FILE")")
CLIENT_ARGS=$(json2args "$(jq -c --arg model "$RUNTIME_MODEL" '
  .client_parameters + {model: $model}
  | if .dataset_name == "random" then
      (if has("input_len") then . + {random_input_len: .input_len} | del(.input_len) else . end)
      | (if has("output_len") then . + {random_output_len: .output_len} | del(.output_len) else . end)
    else
      .
    end
' "$SPEC_FILE")")

RAW_RESULT_FILE="$RESULT_DIR/raw_benchmark_result.json"
ARTIFACT_DIR="$RESULT_DIR/submission"

SERVER_COMMAND="PYTHONUNBUFFERED=1 PYTHONPATH=$OFFICIAL_RUNTIME_PYTHONPATH\${PYTHONPATH:+:\$PYTHONPATH} conda run --no-capture-output -p $GOAL_BASELINE_ENV_PREFIX python -u -m vllm.entrypoints.openai.api_server $SERVER_ARGS"
CLIENT_COMMAND="PYTHONPATH=$OFFICIAL_RUNTIME_PYTHONPATH\${PYTHONPATH:+:\$PYTHONPATH} conda run -p $GOAL_BASELINE_ENV_PREFIX python -m vllm.entrypoints.cli.main bench serve --save-result --result-dir $RESULT_DIR --result-filename $(basename "$RAW_RESULT_FILE") $CLIENT_ARGS"

echo "[goal-baseline] using worktrees: $OFFICIAL_VLLM_WORKTREE and $OFFICIAL_VLLM_ASCEND_WORKTREE"
echo "[goal-baseline] neutral cwd: $OFFICIAL_RUNTIME_CWD"
echo "[goal-baseline] vllm cache root: $OFFICIAL_VLLM_CACHE_ROOT"
echo "[goal-baseline] export model id: $MODEL"
echo "[goal-baseline] runtime model source: $RUNTIME_MODEL"
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

wait_for_server "$CLIENT_HOST" "$CLIENT_PORT"

echo "[goal-baseline] client command: $CLIENT_COMMAND"
run_client_command

run_in_official_runtime "$REPO_ROOT/src:$OFFICIAL_RUNTIME_PYTHONPATH" \
python -m vllm_hust_benchmark.cli export-leaderboard-artifact \
  "$SCENARIO" \
  --benchmark-result-file "$RAW_RESULT_FILE" \
  --constraints-file "$CONSTRAINTS_FILE" \
  --output-dir "$ARTIFACT_DIR" \
  --run-id "$RUN_ID" \
  --engine "$ENGINE" \
  --engine-version "$ENGINE_VERSION" \
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