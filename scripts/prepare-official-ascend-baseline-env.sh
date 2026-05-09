#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=${VLLM_HUST_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}

ENV_PREFIX=${ENV_PREFIX:-"/root/miniconda3/envs/vllm-ascend-official-v0110"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.11"}
OFFICIAL_VLLM_REPO=${OFFICIAL_VLLM_REPO:-"$WORKSPACE_ROOT/reference-repos/vllm"}
OFFICIAL_VLLM_ASCEND_REPO=${OFFICIAL_VLLM_ASCEND_REPO:-"$WORKSPACE_ROOT/reference-repos/vllm-ascend"}
OFFICIAL_VLLM_WORKTREE=${OFFICIAL_VLLM_WORKTREE:-"/tmp/vllm-v0110"}
OFFICIAL_VLLM_ASCEND_WORKTREE=${OFFICIAL_VLLM_ASCEND_WORKTREE:-"/tmp/vllm-ascend-v0110"}
OFFICIAL_VLLM_REF=${OFFICIAL_VLLM_REF:-"v0.11.0"}
OFFICIAL_VLLM_ASCEND_REF=${OFFICIAL_VLLM_ASCEND_REF:-"v0.11.0"}
BENCHMARK_SERVER_PORT=${BENCHMARK_SERVER_PORT:-"8000"}
PREPARE_BENCHMARK_ADMISSION_ONLY=${PREPARE_BENCHMARK_ADMISSION_ONLY:-"0"}
ASCEND_TOOLKIT_SET_ENV=${ASCEND_TOOLKIT_SET_ENV:-"/usr/local/Ascend/ascend-toolkit/set_env.sh"}
ASCEND_ATB_SET_ENV=${ASCEND_ATB_SET_ENV:-"/usr/local/Ascend/nnal/atb/set_env.sh"}
ASCEND_ATB_CXX_ABI=${ASCEND_ATB_CXX_ABI:-"1"}
EXTRA_PYPI_INDEX=${EXTRA_PYPI_INDEX:-"https://mirrors.huaweicloud.com/ascend/repos/pypi"}

export ENV_PREFIX

ensure_worktree() {
  local source_repo=$1
  local target_dir=$2
  local ref_name=$3
  if [[ -f "$target_dir/pyproject.toml" ]]; then
    return 0
  fi
  git -C "$source_repo" worktree add --detach "$target_dir" "$ref_name"
}

run_with_ascend_env() {
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
  "$@"
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

list_benchmark_residual_pids() {
  ps -eo pid=,args= | awk '
    /vllm\.entrypoints\.openai\.api_server|vllm\.entrypoints\.cli\.main bench serve|EngineCore_DP0/ && !/awk/ {
      print $1
    }
  ' | sort -u
}

cleanup_benchmark_residual_processes() {
  if [[ "$PREPARE_BENCHMARK_ADMISSION_ONLY" == "1" ]]; then
    local port_pids
    port_pids=$(list_port_listener_pids "$BENCHMARK_SERVER_PORT")
    if [[ -n "$port_pids" ]]; then
      echo "Port ${BENCHMARK_SERVER_PORT} is already occupied by listening processes: $port_pids" >&2
      return 1
    fi

    if [[ -n "$(list_benchmark_residual_pids)" ]]; then
      echo "Residual benchmark processes still exist during admission check" >&2
      return 1
    fi

    echo "[official-env] benchmark admission preflight passed: no residual benchmark processes"
    return 0
  fi

  local pids
  pids=$(list_benchmark_residual_pids)

  if [[ -n "$pids" ]]; then
    echo "[official-env] cleaning residual benchmark processes: $pids"
    kill $pids 2>/dev/null || true

    local pid
    for pid in $pids; do
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    done
  fi

  local port_pids
  port_pids=$(list_port_listener_pids "$BENCHMARK_SERVER_PORT")
  if [[ -n "$port_pids" ]]; then
    echo "Port ${BENCHMARK_SERVER_PORT} is already occupied by listening processes: $port_pids" >&2
    return 1
  fi

  if [[ -n "$(list_benchmark_residual_pids)" ]]; then
    echo "Residual benchmark processes still exist after cleanup" >&2
    return 1
  fi

  echo "[official-env] benchmark admission preflight passed: no residual benchmark processes"
}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required" >&2
  exit 2
fi

if [[ ! -d "$OFFICIAL_VLLM_REPO/.git" ]]; then
  echo "Official vllm repo not found: $OFFICIAL_VLLM_REPO" >&2
  exit 2
fi

if [[ ! -d "$OFFICIAL_VLLM_ASCEND_REPO/.git" ]]; then
  echo "Official vllm-ascend repo not found: $OFFICIAL_VLLM_ASCEND_REPO" >&2
  exit 2
fi

cleanup_benchmark_residual_processes

if [[ "$PREPARE_BENCHMARK_ADMISSION_ONLY" == "1" ]]; then
  echo "[official-env] admission-only mode completed"
  exit 0
fi

ensure_worktree "$OFFICIAL_VLLM_REPO" "$OFFICIAL_VLLM_WORKTREE" "$OFFICIAL_VLLM_REF"
ensure_worktree "$OFFICIAL_VLLM_ASCEND_REPO" "$OFFICIAL_VLLM_ASCEND_WORKTREE" "$OFFICIAL_VLLM_ASCEND_REF"

if [[ ! -d "$ENV_PREFIX" ]]; then
  conda create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION" pip
fi

run_with_ascend_env conda run -p "$ENV_PREFIX" python -m pip uninstall -y \
  vllm \
  vllm-ascend \
  vllm_ascend \
  vllm-hust \
  vllm-ascend-hust \
  torch \
  torch-npu \
  torch_npu \
  torchvision \
  torchaudio \
  compressed-tensors \
  depyf \
  llguidance \
  xgrammar \
  fastapi \
  numba || true

run_with_ascend_env conda run -p "$ENV_PREFIX" python -m pip install --upgrade --force-reinstall \
  "setuptools>=77.0.3,<80.0.0"

run_with_ascend_env conda run -p "$ENV_PREFIX" python -m pip install --upgrade --force-reinstall \
  --extra-index-url "$EXTRA_PYPI_INDEX" \
  torch==2.7.1 \
  torch-npu==2.7.1 \
  torchvision==0.22.1 \
  torchaudio==2.7.1

run_with_ascend_env conda run -p "$ENV_PREFIX" python -m pip install --upgrade --force-reinstall \
  -r "$OFFICIAL_VLLM_WORKTREE/requirements/common.txt" \
  -r "$OFFICIAL_VLLM_ASCEND_WORKTREE/requirements.txt" \
  -r "$OFFICIAL_VLLM_ASCEND_WORKTREE/benchmarks/requirements-bench.txt" \
  "setuptools>=77.0.3,<80.0.0" \
  torch==2.7.1 \
  torch-npu==2.7.1 \
  torchvision==0.22.1 \
  torchaudio==2.7.1 \
  numpy==1.26.4 \
  transformers==4.57.1 \
  compressed-tensors==0.11.0 \
  depyf==0.19.0 \
  llguidance==0.7.30 \
  xgrammar==0.1.25 \
  fastapi==0.123.10 \
  numba==0.61.2 \
  opencv-python-headless==4.11.0.86

run_with_ascend_env conda run -p "$ENV_PREFIX" python -m pip install --upgrade --force-reinstall --no-deps \
  "setuptools>=77.0.3,<80.0.0" \
  torch==2.7.1 \
  torch-npu==2.7.1 \
  torchvision==0.22.1 \
  torchaudio==2.7.1 \
  numpy==1.26.4 \
  transformers==4.57.1 \
  compressed-tensors==0.11.0 \
  depyf==0.19.0 \
  llguidance==0.7.30 \
  xgrammar==0.1.25 \
  fastapi==0.123.10 \
  numba==0.61.2

PYTHONPATH="$OFFICIAL_VLLM_ASCEND_WORKTREE:$OFFICIAL_VLLM_WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
  run_with_ascend_env conda run -p "$ENV_PREFIX" python - <<'PY'
import os
from importlib import metadata
from importlib.metadata import entry_points

import torch
import torch_npu
import vllm
import vllm_ascend

print(f"env_prefix={os.environ['ENV_PREFIX']}")
print(f"torch={torch.__version__}")
print(f"torch_npu={torch_npu.__version__}")
print(f"vllm_file={vllm.__file__}")
print(f"vllm_ascend_file={vllm_ascend.__file__}")
print(f"setuptools={metadata.version('setuptools')}")
print(f"compressed_tensors={metadata.version('compressed-tensors')}")
print(f"depyf={metadata.version('depyf')}")
print(f"llguidance={metadata.version('llguidance')}")
print(f"xgrammar={metadata.version('xgrammar')}")
print(f"fastapi={metadata.version('fastapi')}")
print(f"numba={metadata.version('numba')}")
print(f"transformers={metadata.version('transformers')}")
print(f"numpy={metadata.version('numpy')}")
print("platform_plugins=" + ",".join(sorted(ep.name for ep in entry_points(group='vllm.platform_plugins'))))
print("general_plugins=" + ",".join(sorted(ep.name for ep in entry_points(group='vllm.general_plugins'))))
PY

echo "Prepared official Ascend baseline env at $ENV_PREFIX"
echo "Pinned runtime source refs: vllm=$OFFICIAL_VLLM_REF vllm-ascend=$OFFICIAL_VLLM_ASCEND_REF"
echo "Use with: GOAL_BASELINE_ENV_PREFIX=$ENV_PREFIX bash $REPO_ROOT/scripts/run-official-ascend-goal-baseline.sh"