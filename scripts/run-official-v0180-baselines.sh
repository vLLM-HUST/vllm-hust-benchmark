#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

DEFAULT_DATA_ROOT="/data/shared_datasets/vllm-hust-benchmark"
DEFAULT_MODEL_PATH="/data/shared_models/Qwen--Qwen2.5-14B-Instruct"
DEFAULT_ENV_PREFIX="$(conda info --base 2>/dev/null || printf '%s\n' "$HOME/miniconda3")/envs/vllm-ascend-official-v0180"
DEFAULT_ASCEND_TOOLKIT_SET_ENV="/opt/hust-ascend-cann/Ascend/cann-8.5.0/set_env.sh"
DEFAULT_ASCEND_ATB_SET_ENV="/opt/hust-ascend-cann/Ascend/nnal/atb/set_env.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-official-v0180-baselines.sh [options] [<spec-file-or-dir> ...]

One-command wrapper for the official vLLM 0.18.0 + vLLM-Ascend 0.18.0
baseline matrix. Defaults are tuned for the vLLM-HUST 910B2 machine:

  - Uses /data for benchmark artifacts and Hugging Face caches
  - Uses hf-mirror.com for model/dataset downloads
  - Uses the pinned official conda env vllm-ascend-official-v0180
  - Sets SKIP_OFFICIAL_ASCEND_C_EXTENSION_BUILD=1 by default so the fragile
    official custom-op build is skipped and the sampler fallback is used
  - Runs through scripts/run-official-ascend-goal-baseline-matrix.sh

Options:
  --repeat-count N       Successful repeats per missing canonical spec (default: 1)
  --result-root DIR      Matrix artifact root (default: /data/.../official-baseline-runs/<timestamp>)
  --model-path DIR       Local model path (default: /data/shared_models/Qwen--Qwen2.5-14B-Instruct if present)
  --env-prefix DIR       Official conda env path
  --devices LIST         ASCEND_VISIBLE_DEVICES/ASCEND_RT_VISIBLE_DEVICES, e.g. 0 or 1,2
  --publish-website     Rebuild local website data after the matrix
  --review-existing     Rerun specs that already have canonical submissions without replacing them
  --force-repair-env    Force a full official env repair before running
  --no-prepare-env      Skip official env preparation
  -h, --help            Show this help

Examples:
  bash scripts/run-official-v0180-baselines.sh
  bash scripts/run-official-v0180-baselines.sh --repeat-count 3 docs/official-baselines/
  bash scripts/run-official-v0180-baselines.sh --devices 0 \
    docs/official-baselines/official-ascend-jan-2026-v0180-agent-research-online-qwen25-14b-910b2.json
EOF
}

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
repeat_count=${REPEAT_COUNT:-1}
result_root=${MATRIX_RESULT_ROOT:-"$DEFAULT_DATA_ROOT/official-baseline-runs/official-v0180-$timestamp"}
model_path=${OFFICIAL_MODEL_PATH:-}
env_prefix=${GOAL_BASELINE_ENV_PREFIX:-$DEFAULT_ENV_PREFIX}
publish_website=${PUBLISH_WEBSITE:-0}
force_run_existing=${FORCE_RUN_EXISTING:-0}
prepare_official_env=${PREPARE_OFFICIAL_ENV:-1}
force_repair_env=${FORCE_REPAIR_OFFICIAL_ENV:-0}
devices=${ASCEND_VISIBLE_DEVICES:-}
spec_paths=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeat-count)
      [[ $# -ge 2 ]] || { echo "--repeat-count requires a value" >&2; exit 2; }
      repeat_count=$2
      shift 2
      ;;
    --result-root)
      [[ $# -ge 2 ]] || { echo "--result-root requires a value" >&2; exit 2; }
      result_root=$2
      shift 2
      ;;
    --model-path)
      [[ $# -ge 2 ]] || { echo "--model-path requires a value" >&2; exit 2; }
      model_path=$2
      shift 2
      ;;
    --env-prefix)
      [[ $# -ge 2 ]] || { echo "--env-prefix requires a value" >&2; exit 2; }
      env_prefix=$2
      shift 2
      ;;
    --devices)
      [[ $# -ge 2 ]] || { echo "--devices requires a value" >&2; exit 2; }
      devices=$2
      shift 2
      ;;
    --publish-website)
      publish_website=1
      shift
      ;;
    --review-existing)
      force_run_existing=1
      shift
      ;;
    --force-repair-env)
      force_repair_env=1
      shift
      ;;
    --no-prepare-env)
      prepare_official_env=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      spec_paths+=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      spec_paths+=("$1")
      shift
      ;;
  esac
done

if [[ ${#spec_paths[@]} -eq 0 ]]; then
  spec_paths=("$REPO_ROOT/docs/official-baselines")
fi

if [[ -z "$model_path" && -d "$DEFAULT_MODEL_PATH" ]]; then
  model_path=$DEFAULT_MODEL_PATH
fi

mkdir -p "$result_root"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$DEFAULT_DATA_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export GOAL_BASELINE_ENV_PREFIX="$env_prefix"
export MATRIX_RESULT_ROOT="$result_root"
export REPEAT_COUNT="$repeat_count"
export PREPARE_OFFICIAL_ENV="$prepare_official_env"
export FORCE_REPAIR_OFFICIAL_ENV="$force_repair_env"
export PUBLISH_WEBSITE="$publish_website"
export FORCE_RUN_EXISTING="$force_run_existing"
export SKIP_OFFICIAL_ASCEND_C_EXTENSION_BUILD="${SKIP_OFFICIAL_ASCEND_C_EXTENSION_BUILD:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -n "$model_path" ]]; then
  export OFFICIAL_MODEL_PATH="$model_path"
fi

if [[ -f "$DEFAULT_ASCEND_TOOLKIT_SET_ENV" ]]; then
  export ASCEND_TOOLKIT_SET_ENV="${ASCEND_TOOLKIT_SET_ENV:-$DEFAULT_ASCEND_TOOLKIT_SET_ENV}"
fi

if [[ -f "$DEFAULT_ASCEND_ATB_SET_ENV" ]]; then
  export ASCEND_ATB_SET_ENV="${ASCEND_ATB_SET_ENV:-$DEFAULT_ASCEND_ATB_SET_ENV}"
fi

if [[ -n "$devices" ]]; then
  export ASCEND_VISIBLE_DEVICES="$devices"
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$devices}"
fi

cat <<EOF
[official-v0180] env_prefix: $GOAL_BASELINE_ENV_PREFIX
[official-v0180] result_root: $MATRIX_RESULT_ROOT
[official-v0180] repeat_count: $REPEAT_COUNT
[official-v0180] prepare_env: $PREPARE_OFFICIAL_ENV
[official-v0180] force_repair_env: $FORCE_REPAIR_OFFICIAL_ENV
[official-v0180] force_run_existing: $FORCE_RUN_EXISTING
[official-v0180] publish_website: $PUBLISH_WEBSITE
[official-v0180] model_path: ${OFFICIAL_MODEL_PATH:-auto}
[official-v0180] devices: ${ASCEND_VISIBLE_DEVICES:-auto}
[official-v0180] hf_endpoint: $HF_ENDPOINT
[official-v0180] specs: ${spec_paths[*]}
EOF

exec bash "$SCRIPT_DIR/run-official-ascend-goal-baseline-matrix.sh" "${spec_paths[@]}"
