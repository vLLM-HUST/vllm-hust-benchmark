#!/usr/bin/env bash
# One-click setup for the official vLLM v0.18.0 + vllm-ascend v0.18.0 baseline.
# Usage: bash scripts/setup-official-v0180-baseline.sh [--skip-model]
#
# This script:
#   1. Creates a conda env (vllm-ascend-official-v0180)
#   2. Sets up git worktrees for vLLM v0.18.0 and vllm-ascend v0.18.0
#   3. Installs torch + torch-npu + all dependencies
#   4. Writes the vllm-ascend plugin metadata
#   5. Downloads Qwen2.5-14B-Instruct model (unless --skip-model)
#   6. Runs a quick health check
#
# Prerequisites:
#   - conda installed at ~/miniconda3
#   - reference-repos/vllm and reference-repos/vllm-ascend cloned with tags fetched
#   - Ascend toolkit at /usr/local/Ascend/ascend-toolkit
#   - Network access to PyPI / Huawei mirrors

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="${VLLM_HUST_WORKSPACE_ROOT:-$(cd -- "$REPO_ROOT/.." && pwd)}"
PREPARE_SCRIPT="$SCRIPT_DIR/prepare-official-ascend-baseline-env.sh"

# ---------- configurable ----------
ENV_NAME="vllm-ascend-official-v0180"
PYTHON_VERSION="3.11"
VLLM_REF="v0.18.0"
VLLM_ASCEND_REF="v0.18.0"
VLLM_REPO="${WORKSPACE_ROOT}/reference-repos/vllm"
VLLM_ASCEND_REPO="${WORKSPACE_ROOT}/reference-repos/vllm-ascend"
VLLM_WORKTREE="/tmp/vllm-v0180"
VLLM_ASCEND_WORKTREE="/tmp/vllm-ascend-v0180"
MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
MODEL_DIR="/data/shared_models/Qwen2.5-14B-Instruct"
HF_MIRROR="https://hf-mirror.com"
TORCH_VERSION="2.9.0"
TORCH_NPU_VERSION="2.9.0"
TORCHVISION_VERSION="0.24.0"
TORCHAUDIO_VERSION="2.9.0"
SETUPTOOLS_VERSION="80.9.0"
SOC_VERSION="${SOC_VERSION:-ascend910b2}"
ASCEND_TOOLKIT_SET_ENV="/usr/local/Ascend/ascend-toolkit/set_env.sh"
ASCEND_ATB_SET_ENV="/usr/local/Ascend/nnal/atb/set_env.sh"
SKIP_MODEL=0
# ---------- end configurable ----------

# Fix conda permission quirk on machines where /root/.cache is not writable.
export HOME="${HOME:-/home/$(whoami)}"
export XDG_CACHE_HOME="$HOME/.cache"
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic

# Detect conda
CONDA_BIN=""
for candidate in \
  "$(command -v conda 2>/dev/null || true)" \
  "$HOME/miniconda3/condabin/conda" \
  "$HOME/anaconda3/condabin/conda" \
  "/opt/conda/condabin/conda"; do
  if [[ -x "$candidate" ]]; then
    CONDA_BIN="$candidate"
    break
  fi
done

if [[ -z "$CONDA_BIN" ]]; then
  echo "ERROR: conda not found. Install miniconda first." >&2
  exit 1
fi

CONDA_BASE="$("$CONDA_BIN" info --base 2>/dev/null)"
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"

# Parse args
for arg in "$@"; do
  case "$arg" in
    --skip-model) SKIP_MODEL=1 ;;
    --help|-h)
      sed -n '2,/^$/s/^# //p' "$0"
      exit 0
      ;;
  esac
done

log() { printf '\033[1;36m[baseline-setup]\033[0m %s\n' "$1"; }
err() { printf '\033[1;31m[baseline-setup]\033[0m %s\n' "$1" >&2; }

source_ascend_env() {
  export ZSH_VERSION=""
  if [[ -f "$ASCEND_TOOLKIT_SET_ENV" ]]; then
    set +u; source "$ASCEND_TOOLKIT_SET_ENV"; set -u
  fi
  if [[ -f "$ASCEND_ATB_SET_ENV" ]]; then
    set +u; source "$ASCEND_ATB_SET_ENV" --cxx_abi=1; set -u
  fi
}

PIP="$ENV_PREFIX/bin/pip"
PYTHON="$ENV_PREFIX/bin/python"

# ============================================================
# Step 1: Prepare official baseline env and worktrees
# ============================================================
log "Step 1/3: Preparing official v0.18.0 baseline env and worktrees..."
if [[ ! -f "$PREPARE_SCRIPT" ]]; then
  err "prepare script not found: $PREPARE_SCRIPT"
  exit 1
fi
ENV_PREFIX="$ENV_PREFIX" \
PYTHON_VERSION="$PYTHON_VERSION" \
OFFICIAL_VLLM_REPO="$VLLM_REPO" \
OFFICIAL_VLLM_ASCEND_REPO="$VLLM_ASCEND_REPO" \
OFFICIAL_VLLM_WORKTREE="$VLLM_WORKTREE" \
OFFICIAL_VLLM_ASCEND_WORKTREE="$VLLM_ASCEND_WORKTREE" \
OFFICIAL_VLLM_REF="$VLLM_REF" \
OFFICIAL_VLLM_ASCEND_REF="$VLLM_ASCEND_REF" \
OFFICIAL_SOC_VERSION="$SOC_VERSION" \
OFFICIAL_SLEEP_MODE_ENABLED="0" \
OFFICIAL_TORCH_VERSION="$TORCH_VERSION" \
OFFICIAL_TORCH_NPU_VERSION="$TORCH_NPU_VERSION" \
OFFICIAL_TORCHVISION_VERSION="$TORCHVISION_VERSION" \
OFFICIAL_TORCHAUDIO_VERSION="$TORCHAUDIO_VERSION" \
OFFICIAL_SETUPTOOLS_VERSION="$SETUPTOOLS_VERSION" \
SKIP_BENCHMARK_RESIDUAL_CLEANUP="1" \
VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
bash "$PREPARE_SCRIPT"

if [[ ! -x "$PIP" ]] || [[ ! -x "$PYTHON" ]]; then
  err "python or pip not found in prepared env at $ENV_PREFIX"
  exit 1
fi

# ============================================================
# Step 2: Download model
# ============================================================
if (( SKIP_MODEL == 0 )); then
  log "Step 2/3: Downloading model $MODEL_NAME..."
  if [[ -d "$MODEL_DIR" ]] && [[ -f "$MODEL_DIR/config.json" ]]; then
    log "  Model already exists at $MODEL_DIR, skipping."
  else
    mkdir -p "$MODEL_DIR"
    if "$PIP" show huggingface-hub >/dev/null 2>&1; then
      HF_ENDPOINT="$HF_MIRROR" "$PYTHON" -m huggingface_hub.commands.huggingface_cli download \
        "$MODEL_NAME" --local-dir "$MODEL_DIR" 2>&1 | tail -10
    else
      "$PIP" install huggingface-hub 2>&1 | tail -3
      HF_ENDPOINT="$HF_MIRROR" "$PYTHON" -m huggingface_hub.commands.huggingface_cli download \
        "$MODEL_NAME" --local-dir "$MODEL_DIR" 2>&1 | tail -10
    fi
    log "  Model downloaded to $MODEL_DIR"
  fi
else
  log "Step 2/3: Skipping model download (--skip-model)"
fi

# ============================================================
# Step 3: Health check
# ============================================================
log "Step 3/3: Running health check (including C extension ops)..."

(
  cd /tmp
  source_ascend_env
  PYTHONPATH="$VLLM_ASCEND_WORKTREE:$VLLM_WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
  VLLM_CACHE_ROOT="$REPO_ROOT/.cache/official-v0180" \
  "$PYTHON" - <<'PY' 2>&1
import importlib.metadata as md
import pkg_resources
import torch
import torch_npu
print(f'  torch:     {torch.__version__}')
print(f'  torch_npu: {torch_npu.__version__}')

import vllm
print(f'  vllm:      {vllm.__version__}  ({vllm.__file__})  dist={md.version("vllm")}')

import vllm_ascend
import vllm_ascend._build_info as build_info
print(f'  vllm_ascend loaded from: {vllm_ascend.__file__}  dist={md.version("vllm-ascend")}')
print(f'  build_info: soc={build_info.__soc_version__} device_type={build_info.__device_type__}')
print(f'  pkg_resources: {pkg_resources.__file__}')

import transformers
print(f'  transformers: {transformers.__version__}')

# Verify C extension ops are properly registered (LTO fix validation)
from vllm_ascend.utils import enable_custom_op
enable_custom_op()
assert hasattr(torch.ops._C_ascend, 'npu_apply_top_k_top_p'), \
    "FATAL: npu_apply_top_k_top_p not registered! C extension built with LTO stripping constructors."
print(f'  C extension: torch.ops._C_ascend.npu_apply_top_k_top_p = OK')
print()
print('  Health check PASSED.')
PY
) || {
  err "Health check failed - see errors above. The PYTHONPATH runtime may need Ascend toolkit sourced."
  err "Try: cd /tmp && source $ASCEND_TOOLKIT_SET_ENV && PYTHONPATH=$VLLM_ASCEND_WORKTREE:$VLLM_WORKTREE:\$PYTHONPATH $PYTHON -c 'import vllm; import vllm_ascend; import pkg_resources'"
  exit 1
}

echo ""
log "============================================================"
log "Setup complete!"
log ""
log "To run benchmarks:"
log "  cd $REPO_ROOT"
log "  GOAL_BASELINE_ENV_PREFIX=$ENV_PREFIX bash scripts/run-official-ascend-goal-baseline.sh"
log ""
log "Or run all 8 scenarios:"
log "  for spec in docs/official-baselines/official-ascend-jan-2026-v0180-*.json; do"
log "    bash scripts/run-official-ascend-goal-baseline.sh \"\$spec\""
log "  done"
log "============================================================"
