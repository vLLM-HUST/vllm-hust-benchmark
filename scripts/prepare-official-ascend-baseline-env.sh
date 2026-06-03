#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=${VLLM_HUST_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}

ENV_PREFIX=${ENV_PREFIX:-""}
PYTHON_VERSION=${PYTHON_VERSION:-"3.11"}
OFFICIAL_VLLM_REPO=${OFFICIAL_VLLM_REPO:-"$WORKSPACE_ROOT/reference-repos/vllm"}
OFFICIAL_VLLM_ASCEND_REPO=${OFFICIAL_VLLM_ASCEND_REPO:-"$WORKSPACE_ROOT/reference-repos/vllm-ascend"}
OFFICIAL_VLLM_WORKTREE=${OFFICIAL_VLLM_WORKTREE:-"/tmp/vllm-v0110"}
OFFICIAL_VLLM_ASCEND_WORKTREE=${OFFICIAL_VLLM_ASCEND_WORKTREE:-"/tmp/vllm-ascend-v0110"}
OFFICIAL_VLLM_REF=${OFFICIAL_VLLM_REF:-"v0.11.0"}
OFFICIAL_VLLM_ASCEND_REF=${OFFICIAL_VLLM_ASCEND_REF:-"v0.11.0"}
OFFICIAL_SOC_VERSION=${OFFICIAL_SOC_VERSION:-${SOC_VERSION:-"ascend910b3"}}
OFFICIAL_SLEEP_MODE_ENABLED=${OFFICIAL_SLEEP_MODE_ENABLED:-${COMPILE_CUSTOM_KERNELS:-"0"}}
BENCHMARK_SERVER_PORT=${BENCHMARK_SERVER_PORT:-"8000"}
PREPARE_BENCHMARK_ADMISSION_ONLY=${PREPARE_BENCHMARK_ADMISSION_ONLY:-"0"}
ASCEND_TOOLKIT_SET_ENV=${ASCEND_TOOLKIT_SET_ENV:-"/usr/local/Ascend/ascend-toolkit/set_env.sh"}
ASCEND_ATB_SET_ENV=${ASCEND_ATB_SET_ENV:-"/usr/local/Ascend/nnal/atb/set_env.sh"}
ASCEND_ATB_CXX_ABI=${ASCEND_ATB_CXX_ABI:-"1"}
EXTRA_PYPI_INDEX=${EXTRA_PYPI_INDEX:-"https://mirrors.huaweicloud.com/ascend/repos/pypi"}
PYTORCH_CPU_INDEX_URL=${PYTORCH_CPU_INDEX_URL:-"https://download.pytorch.org/whl/cpu"}
DEFAULT_OFFICIAL_TORCH_VERSION="2.7.1"
DEFAULT_OFFICIAL_TORCH_NPU_VERSION="2.7.1.post1"
DEFAULT_OFFICIAL_TORCHVISION_VERSION="0.22.1"
DEFAULT_OFFICIAL_TORCHAUDIO_VERSION="2.7.1"
OFFICIAL_TORCH_VERSION=${OFFICIAL_TORCH_VERSION:-"$DEFAULT_OFFICIAL_TORCH_VERSION"}
OFFICIAL_TORCH_NPU_VERSION=${OFFICIAL_TORCH_NPU_VERSION:-"$DEFAULT_OFFICIAL_TORCH_NPU_VERSION"}
OFFICIAL_TORCHVISION_VERSION=${OFFICIAL_TORCHVISION_VERSION:-"$DEFAULT_OFFICIAL_TORCHVISION_VERSION"}
OFFICIAL_TORCHAUDIO_VERSION=${OFFICIAL_TORCHAUDIO_VERSION:-"$DEFAULT_OFFICIAL_TORCHAUDIO_VERSION"}
OFFICIAL_NUMPY_VERSION=${OFFICIAL_NUMPY_VERSION:-"1.26.4"}
OFFICIAL_TRANSFORMERS_VERSION=${OFFICIAL_TRANSFORMERS_VERSION:-"4.57.4"}
OFFICIAL_COMPRESSED_TENSORS_VERSION=${OFFICIAL_COMPRESSED_TENSORS_VERSION:-"0.13.0"}
OFFICIAL_DEPYF_VERSION=${OFFICIAL_DEPYF_VERSION:-"0.20.0"}
OFFICIAL_LLGUIDANCE_VERSION=${OFFICIAL_LLGUIDANCE_VERSION:-"1.3.0"}
OFFICIAL_XGRAMMAR_VERSION=${OFFICIAL_XGRAMMAR_VERSION:-"0.1.32"}
OFFICIAL_FASTAPI_VERSION=${OFFICIAL_FASTAPI_VERSION:-"0.123.10"}
OFFICIAL_NUMBA_VERSION=${OFFICIAL_NUMBA_VERSION:-"0.61.2"}
OFFICIAL_OPENCV_VERSION=${OFFICIAL_OPENCV_VERSION:-"4.11.0.86"}
OFFICIAL_UVLOOP_TARGET=${OFFICIAL_UVLOOP_TARGET:-"uvloop"}
OFFICIAL_TORCH_WHEEL_URL=${OFFICIAL_TORCH_WHEEL_URL:-""}
OFFICIAL_TORCH_NPU_WHEEL_URL=${OFFICIAL_TORCH_NPU_WHEEL_URL:-""}
OFFICIAL_TORCHVISION_WHEEL_URL=${OFFICIAL_TORCHVISION_WHEEL_URL:-""}
OFFICIAL_TORCHAUDIO_WHEEL_URL=${OFFICIAL_TORCHAUDIO_WHEEL_URL:-""}
FORCE_REPAIR_OFFICIAL_ENV=${FORCE_REPAIR_OFFICIAL_ENV:-"0"}
BENCHMARK_RUNTIME_STATE_ROOT=${BENCHMARK_RUNTIME_STATE_ROOT:-"$REPO_ROOT/.benchmarks"}
MANAGED_BENCHMARK_RUNNER_BASENAME=${MANAGED_BENCHMARK_RUNNER_BASENAME:-"run-official-ascend-goal-baseline.sh"}
CURRENT_PREPARE_USER_ID=${CURRENT_PREPARE_USER_ID:-$(id -u 2>/dev/null || true)}
CURRENT_PREPARE_PID_NAMESPACE=${CURRENT_PREPARE_PID_NAMESPACE:-$(readlink /proc/self/ns/pid 2>/dev/null || true)}
CURRENT_PREPARE_MOUNT_NAMESPACE=${CURRENT_PREPARE_MOUNT_NAMESPACE:-$(readlink /proc/self/ns/mnt 2>/dev/null || true)}

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
  "$@"
}

run_in_official_env_python() {
  local pythonpath_prefix=$1
  shift
  local script_file
  local status=0

  script_file=$(mktemp "${TMPDIR:-/tmp}/official-env-python-XXXXXX.py")
  cat > "$script_file"

  if PYTHONPATH="$pythonpath_prefix${PYTHONPATH:+:$PYTHONPATH}" \
    run_with_ascend_env conda run -p "$ENV_PREFIX" "$@" python "$script_file"; then
    status=0
  else
    status=$?
  fi

  rm -f "$script_file"
  return "$status"
}

python_bool_literal() {
  case "${1,,}" in
    1|true|yes|on)
      printf '%s\n' "True"
      ;;
    *)
      printf '%s\n' "False"
      ;;
  esac
}

ensure_vllm_ascend_plugin_metadata() {
  local version=${OFFICIAL_VLLM_ASCEND_REF#v}
  local dist_info_dir
  local build_info_file
  local setup_py
  local line
  local group=""
  local entry=""
  local -a platform_entries=()
  local -a general_entries=()
  local sleep_mode_enabled_literal

  if [[ -z "$version" ]] || [[ "$version" == "$OFFICIAL_VLLM_ASCEND_REF" ]]; then
    version="0.0.0"
  fi

  setup_py="$OFFICIAL_VLLM_ASCEND_WORKTREE/setup.py"
  while IFS= read -r line; do
    case "$line" in
      *'"vllm.platform_plugins": ['*)
        group="platform"
        continue
        ;;
      *'"vllm.general_plugins": ['*)
        group="general"
        continue
        ;;
      *']'*)
        if [[ -n "$group" ]]; then
          group=""
        fi
        ;;
    esac

    if [[ -n "$group" ]]; then
      entry=$(printf '%s\n' "$line" | sed -n 's/.*"\([^"]*\)".*/\1/p')
    else
      entry=""
    fi

    if [[ -n "$entry" ]]; then
      case "$group" in
        platform)
          platform_entries+=("$entry")
          ;;
        general)
          general_entries+=("$entry")
          ;;
      esac
    fi
  done < "$setup_py"

  if [[ ${#platform_entries[@]} -eq 0 ]]; then
    platform_entries+=("ascend = vllm_ascend:register")
  fi

  dist_info_dir="$OFFICIAL_VLLM_ASCEND_WORKTREE/vllm_ascend-${version}.dist-info"
  mkdir -p "$dist_info_dir"

  cat > "$dist_info_dir/METADATA" <<EOF
Metadata-Version: 2.1
Name: vllm-ascend
Version: $version
EOF

  cat > "$dist_info_dir/top_level.txt" <<'EOF'
vllm_ascend
EOF

  {
    printf '%s\n' '[vllm.platform_plugins]'
    printf '%s\n' "${platform_entries[@]}"
    if [[ ${#general_entries[@]} -gt 0 ]]; then
      printf '\n%s\n' '[vllm.general_plugins]'
      printf '%s\n' "${general_entries[@]}"
    fi
  } > "$dist_info_dir/entry_points.txt"

  mkdir -p "$OFFICIAL_VLLM_ASCEND_WORKTREE/vllm_ascend"
  build_info_file="$OFFICIAL_VLLM_ASCEND_WORKTREE/vllm_ascend/_build_info.py"
  sleep_mode_enabled_literal=$(python_bool_literal "$OFFICIAL_SLEEP_MODE_ENABLED")
  cat > "$build_info_file" <<EOF
# Auto-generated file
__soc_version__ = '$OFFICIAL_SOC_VERSION'
__sleep_mode_enabled__ = $sleep_mode_enabled_literal
EOF

  OFFICIAL_EXPECTED_PLATFORM_PLUGINS=$(printf '%s\n' "${platform_entries[@]}" | sed 's/[[:space:]]*=.*$//' | paste -sd, -)
  OFFICIAL_EXPECTED_GENERAL_PLUGINS=$(printf '%s\n' "${general_entries[@]}" | sed 's/[[:space:]]*=.*$//' | paste -sd, -)
}

normalize_arch() {
  case "$1" in
    x86_64|amd64)
      printf '%s\n' "x86_64"
      ;;
    aarch64|arm64)
      printf '%s\n' "aarch64"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

resolve_python_abi_tag() {
  printf '%s\n' "$1" | sed -E 's/^([0-9]+)\.([0-9]+).*$/cp\1\2/'
}

add_pip_extra_index_arg() {
  local index_url=$1
  if [[ -n "$index_url" ]]; then
    PIP_EXTRA_INDEX_ARGS+=(--extra-index-url "$index_url")
  fi
}

resolve_default_package_version() {
  case "$1" in
    torch)
      printf '%s\n' "$DEFAULT_OFFICIAL_TORCH_VERSION"
      ;;
    torch-npu)
      printf '%s\n' "$DEFAULT_OFFICIAL_TORCH_NPU_VERSION"
      ;;
    torchvision)
      printf '%s\n' "$DEFAULT_OFFICIAL_TORCHVISION_VERSION"
      ;;
    torchaudio)
      printf '%s\n' "$DEFAULT_OFFICIAL_TORCHAUDIO_VERSION"
      ;;
    *)
      printf '%s\n' ""
      ;;
  esac
}

resolve_default_archived_wheel_url() {
  local package_name=$1
  local python_abi_tag=$2
  local runtime_arch=$3

  case "$package_name:$python_abi_tag:$runtime_arch" in
    torch:cp311:x86_64)
      printf '%s\n' "https://download-r2.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"
      ;;
    torch:cp311:aarch64)
      printf '%s\n' "https://download-r2.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl"
      ;;
    torch-npu:cp311:x86_64)
      printf '%s\n' "https://mirrors.huaweicloud.com/ascend/repos/pypi/torch-npu/torch_npu-2.7.1.post1-cp311-cp311-manylinux_2_28_x86_64.whl"
      ;;
    torch-npu:cp311:aarch64)
      printf '%s\n' "https://mirrors.huaweicloud.com/ascend/repos/pypi/torch-npu/torch_npu-2.7.1.post1-cp311-cp311-manylinux_2_28_aarch64.whl"
      ;;
    torchvision:cp311:x86_64)
      printf '%s\n' "https://download-r2.pytorch.org/whl/cpu/torchvision-0.22.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"
      ;;
    torchvision:cp311:aarch64)
      printf '%s\n' "https://download-r2.pytorch.org/whl/cpu/torchvision-0.22.1-cp311-cp311-manylinux_2_28_aarch64.whl"
      ;;
    torchaudio:cp311:x86_64)
      printf '%s\n' "https://download-r2.pytorch.org/whl/cpu/torchaudio-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"
      ;;
    torchaudio:cp311:aarch64)
      printf '%s\n' "https://download-r2.pytorch.org/whl/cpu/torchaudio-2.7.1-cp311-cp311-manylinux_2_28_aarch64.whl"
      ;;
    *)
      printf '%s\n' ""
      ;;
  esac
}

create_filtered_requirements_file() {
  local source_file=$1
  local target_file=$2

  awk '
    /^[[:space:]]*(torch|torch-npu|torch_npu|torchvision|torchaudio)([[:space:]].*|[<>=!~].*)?$/ {
      next
    }
    /^[[:space:]]*numpy([[:space:]].*|[<>=!~].*)?$/ {
      next
    }
    /^[[:space:]]*transformers([[:space:]].*|[<>=!~].*)?$/ {
      next
    }
    /^[[:space:]]*(compressed-tensors|compressed_tensors)([[:space:]].*|[<>=!~].*)?$/ {
      next
    }
    /^[[:space:]]*depyf([[:space:]].*|[<>=!~].*)?$/ {
      next
    }
    /^[[:space:]]*llguidance([[:space:]].*|[<>=!~].*)?$/ {
      next
    }
    /^[[:space:]]*xgrammar([[:space:]].*|[<>=!~].*)?$/ {
      next
    }
    /^[[:space:]]*fastapi(\[[^]]+\])?([[:space:]].*|[<>=!~].*)?$/ {
      next
    }
    /^[[:space:]]*numba([[:space:]].*|[<>=!~].*)?$/ {
      next
    }
    /^[[:space:]]*opencv-python-headless([[:space:]].*|[<>=!~].*)?$/ {
      next
    }
    {
      print
    }
  ' "$source_file" > "$target_file"
}

resolve_package_install_target() {
  local package_name=$1
  local package_version=$2
  local wheel_url=$3
  local default_package_version
  local default_archived_wheel_url

  if [[ -n "$wheel_url" ]]; then
    printf '%s\n' "$wheel_url"
    return 0
  fi

  default_package_version=$(resolve_default_package_version "$package_name")
  if [[ -n "$default_package_version" ]] && [[ "$package_version" == "$default_package_version" ]]; then
    default_archived_wheel_url=$(resolve_default_archived_wheel_url "$package_name" "$OFFICIAL_PYTHON_ABI_TAG" "$OFFICIAL_RUNTIME_ARCH")
    if [[ -n "$default_archived_wheel_url" ]]; then
      printf '%s\n' "$default_archived_wheel_url"
      return 0
    fi
  fi

  printf '%s==%s\n' "$package_name" "$package_version"
}

emit_prepared_env_summary() {
  echo "Prepared official Ascend baseline env at $ENV_PREFIX"
  echo "Pinned runtime source refs: vllm=$OFFICIAL_VLLM_REF vllm-ascend=$OFFICIAL_VLLM_ASCEND_REF"
  echo "Resolved runtime arch: $OFFICIAL_RUNTIME_ARCH"
  echo "Resolved python ABI tag: $OFFICIAL_PYTHON_ABI_TAG"
  echo "Resolved package targets: torch=$OFFICIAL_TORCH_INSTALL_TARGET torch-npu=$OFFICIAL_TORCH_NPU_INSTALL_TARGET torchvision=$OFFICIAL_TORCHVISION_INSTALL_TARGET torchaudio=$OFFICIAL_TORCHAUDIO_INSTALL_TARGET"
  echo "Use with: GOAL_BASELINE_ENV_PREFIX=$ENV_PREFIX bash $REPO_ROOT/scripts/run-official-ascend-goal-baseline.sh"
}

verify_official_env() {
  run_in_official_env_python "$OFFICIAL_VLLM_ASCEND_WORKTREE:$OFFICIAL_VLLM_WORKTREE" \
    env \
    OFFICIAL_EXPECTED_PYTHON_VERSION="$PYTHON_VERSION" \
    OFFICIAL_EXPECTED_SETUPTOOLS_SPEC=">=77.0.3,<80.0.0" \
    OFFICIAL_EXPECTED_PLATFORM_PLUGINS="$OFFICIAL_EXPECTED_PLATFORM_PLUGINS" \
    OFFICIAL_EXPECTED_GENERAL_PLUGINS="$OFFICIAL_EXPECTED_GENERAL_PLUGINS" \
    OFFICIAL_EXPECTED_VLLM_WORKTREE="$OFFICIAL_VLLM_WORKTREE" \
    OFFICIAL_EXPECTED_VLLM_ASCEND_WORKTREE="$OFFICIAL_VLLM_ASCEND_WORKTREE" \
    OFFICIAL_EXPECTED_TORCH_TARGET="$OFFICIAL_TORCH_INSTALL_TARGET" \
    OFFICIAL_EXPECTED_TORCH_NPU_TARGET="$OFFICIAL_TORCH_NPU_INSTALL_TARGET" \
    OFFICIAL_EXPECTED_TORCHVISION_TARGET="$OFFICIAL_TORCHVISION_INSTALL_TARGET" \
    OFFICIAL_EXPECTED_TORCHAUDIO_TARGET="$OFFICIAL_TORCHAUDIO_INSTALL_TARGET" \
    OFFICIAL_EXPECTED_NUMPY_VERSION="$OFFICIAL_NUMPY_VERSION" \
    OFFICIAL_EXPECTED_TRANSFORMERS_VERSION="$OFFICIAL_TRANSFORMERS_VERSION" \
    OFFICIAL_EXPECTED_COMPRESSED_TENSORS_VERSION="$OFFICIAL_COMPRESSED_TENSORS_VERSION" \
    OFFICIAL_EXPECTED_DEPYF_VERSION="$OFFICIAL_DEPYF_VERSION" \
    OFFICIAL_EXPECTED_LLGUIDANCE_VERSION="$OFFICIAL_LLGUIDANCE_VERSION" \
    OFFICIAL_EXPECTED_XGRAMMAR_VERSION="$OFFICIAL_XGRAMMAR_VERSION" \
    OFFICIAL_EXPECTED_FASTAPI_VERSION="$OFFICIAL_FASTAPI_VERSION" \
    OFFICIAL_EXPECTED_NUMBA_VERSION="$OFFICIAL_NUMBA_VERSION" \
    OFFICIAL_EXPECTED_OPENCV_VERSION="$OFFICIAL_OPENCV_VERSION" <<'PY'
import os
import sys
from importlib import metadata
from importlib.metadata import entry_points
from pathlib import Path
from urllib.parse import unquote, urlparse


def fail(message: str) -> None:
  print(f"[official-env] health check failed: {message}", file=sys.stderr)
  raise SystemExit(1)


def dist_version(*names: str) -> str:
  for name in names:
    try:
      return metadata.version(name)
    except metadata.PackageNotFoundError:
      continue
  joined = ", ".join(names)
  raise metadata.PackageNotFoundError(f"No package metadata was found for {joined}")


def ensure_absent(*names: str) -> None:
  for name in names:
    try:
      version = metadata.version(name)
    except metadata.PackageNotFoundError:
      continue
    fail(f"unexpected installed distribution {name}=={version}")


def expected_version_from_target(target: str) -> str:
  if not target:
    fail("empty package target")

  if "://" not in target and "==" in target:
    return target.split("==", 1)[1].strip()

  filename = unquote(Path(urlparse(target).path).name)
  if not filename.endswith(".whl"):
    fail(f"cannot parse expected version from target: {target}")

  parts = filename[:-4].split("-")
  if len(parts) < 5:
    fail(f"unexpected wheel filename format: {filename}")

  return parts[1]


def assert_loaded_from(path_value: str, expected_root: str, label: str) -> None:
  resolved_path = str(Path(path_value).resolve())
  resolved_root = str(Path(expected_root).resolve())
  if resolved_path != resolved_root and not resolved_path.startswith(resolved_root + os.sep):
    fail(f"{label} loaded from {resolved_path}, expected under {resolved_root}")


try:
  from packaging.specifiers import SpecifierSet
  from packaging.version import InvalidVersion, Version
except Exception as exc:  # pragma: no cover - repair path handles this.
  fail(f"required health-check dependency packaging is unavailable: {exc}")


def versions_match(actual: str, expected: str) -> bool:
  try:
    actual_version = Version(actual)
    expected_version = Version(expected)
  except InvalidVersion:
    return actual == expected

  if actual_version == expected_version:
    return True

  # CPU wheels commonly report a local version suffix such as +cpu while the
  # workflow input is expressed as torch==2.9.0. Treat that as compatible when
  # the upstream base version matches and the expectation did not pin a local tag.
  if actual_version.base_version == expected_version.base_version and expected_version.local is None:
    return True

  return False

expected_python_version = os.environ["OFFICIAL_EXPECTED_PYTHON_VERSION"]
actual_python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
if actual_python_version != expected_python_version:
  fail(f"python version is {actual_python_version}, expected {expected_python_version}")

import torch
import torch_npu
import uvloop
import vllm
import vllm_ascend

assert_loaded_from(vllm.__file__, os.environ["OFFICIAL_EXPECTED_VLLM_WORKTREE"], "vllm")
assert_loaded_from(vllm_ascend.__file__, os.environ["OFFICIAL_EXPECTED_VLLM_ASCEND_WORKTREE"], "vllm_ascend")

expected_versions = (
  (("torch",), expected_version_from_target(os.environ["OFFICIAL_EXPECTED_TORCH_TARGET"])),
  (("torch-npu", "torch_npu"), expected_version_from_target(os.environ["OFFICIAL_EXPECTED_TORCH_NPU_TARGET"])),
  (("torchvision",), expected_version_from_target(os.environ["OFFICIAL_EXPECTED_TORCHVISION_TARGET"])),
  (("torchaudio",), expected_version_from_target(os.environ["OFFICIAL_EXPECTED_TORCHAUDIO_TARGET"])),
  (("numpy",), os.environ["OFFICIAL_EXPECTED_NUMPY_VERSION"]),
  (("transformers",), os.environ["OFFICIAL_EXPECTED_TRANSFORMERS_VERSION"]),
  (("compressed-tensors",), os.environ["OFFICIAL_EXPECTED_COMPRESSED_TENSORS_VERSION"]),
  (("depyf",), os.environ["OFFICIAL_EXPECTED_DEPYF_VERSION"]),
  (("llguidance",), os.environ["OFFICIAL_EXPECTED_LLGUIDANCE_VERSION"]),
  (("xgrammar",), os.environ["OFFICIAL_EXPECTED_XGRAMMAR_VERSION"]),
  (("fastapi",), os.environ["OFFICIAL_EXPECTED_FASTAPI_VERSION"]),
  (("numba",), os.environ["OFFICIAL_EXPECTED_NUMBA_VERSION"]),
  (("opencv-python-headless",), os.environ["OFFICIAL_EXPECTED_OPENCV_VERSION"]),
)

for names, expected_version in expected_versions:
  actual_version = dist_version(*names)
  if not versions_match(actual_version, expected_version):
    fail(f"{names[0]} version is {actual_version}, expected {expected_version}")

setuptools_version = dist_version("setuptools")
setuptools_spec = SpecifierSet(os.environ["OFFICIAL_EXPECTED_SETUPTOOLS_SPEC"])
if Version(setuptools_version) not in setuptools_spec:
  fail(
    f"setuptools version is {setuptools_version}, expected {os.environ['OFFICIAL_EXPECTED_SETUPTOOLS_SPEC']}"
  )

ensure_absent("vllm", "vllm-hust", "vllm_hust")
ensure_absent("vllm-ascend-hust", "vllm_ascend_hust")

platform_plugins = sorted(ep.name for ep in entry_points(group="vllm.platform_plugins"))
expected_platform_plugins = sorted(filter(None, os.environ["OFFICIAL_EXPECTED_PLATFORM_PLUGINS"].split(",")))
if platform_plugins != expected_platform_plugins:
  fail(
    "platform plugins are "
    f"{','.join(platform_plugins)}, expected {','.join(expected_platform_plugins)}"
  )

general_plugins = sorted(ep.name for ep in entry_points(group="vllm.general_plugins"))
expected_general_plugins = sorted(filter(None, os.environ["OFFICIAL_EXPECTED_GENERAL_PLUGINS"].split(",")))
missing_general_plugins = sorted(set(expected_general_plugins) - set(general_plugins))
if missing_general_plugins:
  fail(
    "general plugins are "
    f"{','.join(general_plugins)}, missing required {','.join(missing_general_plugins)}"
  )

print(f"env_prefix={os.environ['ENV_PREFIX']}")
print(f"torch={torch.__version__}")
print(f"torch_npu={torch_npu.__version__}")
print(f"vllm_file={vllm.__file__}")
print(f"vllm_ascend_file={vllm_ascend.__file__}")
print(f"setuptools={setuptools_version}")
print(f"compressed_tensors={dist_version('compressed-tensors')}")
print(f"depyf={dist_version('depyf')}")
print(f"llguidance={dist_version('llguidance')}")
print(f"xgrammar={dist_version('xgrammar')}")
print(f"fastapi={dist_version('fastapi')}")
print(f"numba={dist_version('numba')}")
print(f"transformers={dist_version('transformers')}")
print(f"numpy={dist_version('numpy')}")
print(f"opencv_python_headless={dist_version('opencv-python-headless')}")
print("platform_plugins=" + ",".join(platform_plugins))
print("general_plugins=" + ",".join(general_plugins))
PY
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

process_args() {
  local pid=$1
  ps -p "$pid" -o args= 2>/dev/null || true
}

process_user_id() {
  local pid=$1
  ps -p "$pid" -o uid= 2>/dev/null | tr -d '[:space:]' || true
}

process_namespace() {
  local pid=$1
  local namespace_name=$2

  readlink "/proc/${pid}/ns/${namespace_name}" 2>/dev/null || true
}

is_benchmark_process() {
  local pid=$1
  local args

  args=$(process_args "$pid")
  [[ -n "$args" ]] && [[ "$args" =~ vllm\.entrypoints\.openai\.api_server|vllm\.entrypoints\.cli\.main\ bench\ serve|run_vllm_cli_compat\.py\ bench\ (serve|throughput|latency)|EngineCore_DP0 ]]
}

is_managed_runner_wrapper_process() {
  local pid=$1
  local args

  args=$(process_args "$pid")
  [[ -n "$args" ]] && [[ "$args" == *"$MANAGED_BENCHMARK_RUNNER_BASENAME"* ]]
}

is_process_in_cleanup_scope() {
  local pid=$1
  local candidate_user_id
  local candidate_pid_namespace
  local candidate_mount_namespace

  if [[ -z "$pid" ]] || [[ "$pid" == "$$" ]]; then
    return 1
  fi

  candidate_user_id=$(process_user_id "$pid")
  if [[ -z "$candidate_user_id" ]] || [[ "$candidate_user_id" != "$CURRENT_PREPARE_USER_ID" ]]; then
    return 1
  fi

  candidate_pid_namespace=$(process_namespace "$pid" pid)
  if [[ -z "$candidate_pid_namespace" ]] || [[ "$candidate_pid_namespace" != "$CURRENT_PREPARE_PID_NAMESPACE" ]]; then
    return 1
  fi

  candidate_mount_namespace=$(process_namespace "$pid" mnt)
  if [[ -z "$candidate_mount_namespace" ]] || [[ "$candidate_mount_namespace" != "$CURRENT_PREPARE_MOUNT_NAMESPACE" ]]; then
    return 1
  fi

  return 0
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

log_process_snapshots() {
  local label=$1
  local pids=$2
  local pid
  local snapshot
  local process_tree

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    snapshot=$(ps -p "$pid" -o pid=,ppid=,stat=,args= 2>/dev/null | sed 's/^[[:space:]]*//')
    if [[ -n "$snapshot" ]]; then
      echo "[official-env] ${label}: ${snapshot}"
      if command -v pstree >/dev/null 2>&1; then
        process_tree=$(pstree -sp "$pid" 2>/dev/null || true)
        if [[ -n "$process_tree" ]]; then
          echo "[official-env] ${label} tree: ${process_tree}"
        fi
      fi
    else
      echo "[official-env] ${label}: pid=${pid} exited before inspection"
    fi
  done <<< "$pids"
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
  tree_list=$(printf '%s\n' "$tree_pids" | tr '\n' ' ')

  echo "[official-env] stopping ${description}: ${tree_list}"
  kill $tree_list 2>/dev/null || true

  for attempt in $(seq 1 5); do
    still_running=0
    while IFS= read -r tree_pid; do
      [[ -z "$tree_pid" ]] && continue
      if is_live_process "$tree_pid"; then
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

  for attempt in $(seq 1 5); do
    still_running=0
    while IFS= read -r tree_pid; do
      [[ -z "$tree_pid" ]] && continue
      if is_live_process "$tree_pid"; then
        still_running=1
        break
      fi
    done <<< "$tree_pids"

    if [[ "$still_running" == "0" ]]; then
      return 0
    fi
    sleep 1
  done

  return 1
}

terminate_benchmark_pid_set() {
  local pids=$1
  local pid

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    terminate_pid_tree "$pid" "benchmark residual process tree rooted at ${pid}" || true
  done <<< "$pids"
}

list_matching_benchmark_pids() {
  ps -eo pid=,args= | awk '
    /vllm\.entrypoints\.openai\.api_server|vllm\.entrypoints\.cli\.main bench serve|run_vllm_cli_compat\.py bench (serve|throughput|latency)|EngineCore_DP0/ && !/awk/ {
      print $1
    }
  ' | sort -u
}

list_managed_runtime_state_dirs() {
  if [[ ! -d "$BENCHMARK_RUNTIME_STATE_ROOT" ]]; then
    return 0
  fi

  find "$BENCHMARK_RUNTIME_STATE_ROOT" -type d -name '.runtime-state' 2>/dev/null | sort || true
}

list_managed_runtime_state_pids() {
  local state_dir

  while IFS= read -r state_dir; do
    [[ -z "$state_dir" ]] && continue
    if [[ -f "$state_dir/server.wrapper.pid" ]]; then
      tr '[:space:]' '\n' < "$state_dir/server.wrapper.pid"
    fi
    if [[ -f "$state_dir/server.listener.pids" ]]; then
      tr '[:space:]' '\n' < "$state_dir/server.listener.pids"
    fi
  done < <(list_managed_runtime_state_dirs) | sed '/^$/d' | sort -u
}

process_state() {
  local pid=$1
  ps -p "$pid" -o stat= 2>/dev/null | tr -d '[:space:]' || true
}

process_parent_pid() {
  local pid=$1
  ps -p "$pid" -o ppid= 2>/dev/null | tr -d '[:space:]' || true
}

is_zombie_process() {
  local pid=$1
  local state

  state=$(process_state "$pid")
  [[ -n "$state" ]] && [[ "$state" == Z* ]]
}

is_live_process() {
  local pid=$1

  if ! kill -0 "$pid" 2>/dev/null; then
    return 1
  fi

  if is_zombie_process "$pid"; then
    return 1
  fi

  return 0
}

list_benchmark_residual_pids() {
  local pid

  {
    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      if is_zombie_process "$pid"; then
        continue
      fi
      if ! is_process_in_cleanup_scope "$pid"; then
        continue
      fi
      if is_benchmark_process "$pid" || is_managed_runner_wrapper_process "$pid"; then
        printf '%s\n' "$pid"
      fi
    done < <(list_managed_runtime_state_pids)

    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      if is_zombie_process "$pid"; then
        continue
      fi
      if ! is_process_in_cleanup_scope "$pid"; then
        continue
      fi
      printf '%s\n' "$pid"
    done < <(list_matching_benchmark_pids)
  } | sort -u
}

list_benchmark_zombie_pids() {
  local pid

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if is_zombie_process "$pid"; then
      if ! is_process_in_cleanup_scope "$pid"; then
        continue
      fi
      printf '%s\n' "$pid"
    fi
  done < <(list_matching_benchmark_pids) | sort -u
}

list_out_of_scope_benchmark_pids() {
  local pid

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if is_zombie_process "$pid"; then
      continue
    fi
    if is_process_in_cleanup_scope "$pid"; then
      continue
    fi
    printf '%s\n' "$pid"
  done < <(list_matching_benchmark_pids) | sort -u
}

reap_benchmark_zombie_parents() {
  local zombie_pids=$1
  local pid
  local parent_pid
  local parent_pids=""

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    parent_pid=$(process_parent_pid "$pid")
    if [[ -z "$parent_pid" ]] || [[ "$parent_pid" == "1" ]] || [[ "$parent_pid" == "$$" ]]; then
      continue
    fi
    parent_pids+="$parent_pid"$'\n'
  done <<< "$zombie_pids"

  parent_pids=$(printf '%s' "$parent_pids" | sed '/^$/d' | sort -u)
  if [[ -z "$parent_pids" ]]; then
    return 0
  fi

  echo "[official-env] reaping zombie benchmark parents: $parent_pids"
  kill $parent_pids 2>/dev/null || true
}

wait_for_no_benchmark_residual_pids() {
  local max_attempts=${1:-10}
  local attempt

  for attempt in $(seq 1 "$max_attempts"); do
    if [[ -z "$(list_benchmark_residual_pids)" ]]; then
      return 0
    fi
    sleep 1
  done

  return 1
}

cleanup_benchmark_residual_processes() {
  if [[ "$PREPARE_BENCHMARK_ADMISSION_ONLY" == "1" ]]; then
    local port_pids
    port_pids=$(list_port_listener_pids "$BENCHMARK_SERVER_PORT")
    if [[ -n "$port_pids" ]]; then
      log_process_snapshots "port listener during admission check" "$port_pids"
      echo "Port ${BENCHMARK_SERVER_PORT} is already occupied by listening processes: $port_pids" >&2
      return 1
    fi

    if [[ -n "$(list_benchmark_residual_pids)" ]]; then
      echo "Residual benchmark processes still exist during admission check" >&2
      return 1
    fi

    local out_of_scope_pids
    out_of_scope_pids=$(list_out_of_scope_benchmark_pids)
    if [[ -n "$out_of_scope_pids" ]]; then
      log_process_snapshots "out-of-scope residual during admission check" "$out_of_scope_pids"
      echo "Out-of-scope benchmark processes exist outside the runner cleanup scope during admission check: $out_of_scope_pids" >&2
      return 1
    fi

    echo "[official-env] benchmark admission preflight passed: no residual benchmark processes"
    return 0
  fi

  local zombie_pids
  zombie_pids=$(list_benchmark_zombie_pids)
  if [[ -n "$zombie_pids" ]]; then
    echo "[official-env] found zombie benchmark processes: $zombie_pids"
    reap_benchmark_zombie_parents "$zombie_pids"
  fi

  local pids
  pids=$(list_benchmark_residual_pids)

  if [[ -n "$pids" ]]; then
    echo "[official-env] cleaning residual benchmark processes: $pids"
    log_process_snapshots "residual before cleanup" "$pids"
    terminate_benchmark_pid_set "$pids"

    if ! wait_for_no_benchmark_residual_pids 5; then
      local remaining_pids
      remaining_pids=$(list_benchmark_residual_pids)
      if [[ -n "$remaining_pids" ]]; then
        echo "[official-env] escalating residual benchmark cleanup: $remaining_pids"
        log_process_snapshots "residual before escalation" "$remaining_pids"
        terminate_benchmark_pid_set "$remaining_pids"
        wait_for_no_benchmark_residual_pids 5 || true
      fi
    fi
  fi

  local port_pids
  port_pids=$(list_port_listener_pids "$BENCHMARK_SERVER_PORT")
  if [[ -n "$port_pids" ]]; then
    log_process_snapshots "port listener after cleanup" "$port_pids"
    echo "Port ${BENCHMARK_SERVER_PORT} is already occupied by listening processes: $port_pids" >&2
    return 1
  fi

  local final_residual_pids
  final_residual_pids=$(list_benchmark_residual_pids)
  if [[ -n "$final_residual_pids" ]]; then
    log_process_snapshots "residual after cleanup" "$final_residual_pids"
    echo "Residual benchmark processes still exist after cleanup: $final_residual_pids" >&2
    return 1
  fi

  local final_out_of_scope_pids
  final_out_of_scope_pids=$(list_out_of_scope_benchmark_pids)
  if [[ -n "$final_out_of_scope_pids" ]]; then
    log_process_snapshots "out-of-scope residual after cleanup" "$final_out_of_scope_pids"
    echo "Out-of-scope benchmark processes remain outside the runner cleanup scope; refusing to kill them automatically: $final_out_of_scope_pids" >&2
    return 1
  fi

  local final_zombie_pids
  final_zombie_pids=$(list_benchmark_zombie_pids)
  if [[ -n "$final_zombie_pids" ]]; then
    echo "[official-env] continuing with zombie benchmark processes that have no listening port: $final_zombie_pids"
  fi

  echo "[official-env] benchmark admission preflight passed: no residual benchmark processes"
}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required" >&2
  exit 2
fi

if [[ -z "$ENV_PREFIX" ]]; then
  ENV_PREFIX="$(conda info --base)/envs/vllm-ascend-official-v0110"
fi

ENV_PREFIX_PARENT=$(dirname "$ENV_PREFIX")
if [[ ! -d "$ENV_PREFIX" ]] && [[ ! -d "$ENV_PREFIX_PARENT" || ! -w "$ENV_PREFIX_PARENT" ]]; then
  echo "ENV_PREFIX parent directory is not writable: $ENV_PREFIX_PARENT" >&2
  echo "Set ENV_PREFIX to a writable conda env path before running this script." >&2
  exit 2
fi

export ENV_PREFIX

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
ensure_vllm_ascend_plugin_metadata

if [[ ! -d "$ENV_PREFIX" ]]; then
  conda create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION" pip
fi

OFFICIAL_RUNTIME_ARCH=$(normalize_arch "$(uname -m)")
OFFICIAL_PYTHON_ABI_TAG=$(resolve_python_abi_tag "$PYTHON_VERSION")
PIP_EXTRA_INDEX_ARGS=()
add_pip_extra_index_arg "$EXTRA_PYPI_INDEX"
add_pip_extra_index_arg "$PYTORCH_CPU_INDEX_URL"

OFFICIAL_TORCH_INSTALL_TARGET=$(resolve_package_install_target torch "$OFFICIAL_TORCH_VERSION" "$OFFICIAL_TORCH_WHEEL_URL")
OFFICIAL_TORCH_NPU_INSTALL_TARGET=$(resolve_package_install_target torch-npu "$OFFICIAL_TORCH_NPU_VERSION" "$OFFICIAL_TORCH_NPU_WHEEL_URL")
OFFICIAL_TORCHVISION_INSTALL_TARGET=$(resolve_package_install_target torchvision "$OFFICIAL_TORCHVISION_VERSION" "$OFFICIAL_TORCHVISION_WHEEL_URL")
OFFICIAL_TORCHAUDIO_INSTALL_TARGET=$(resolve_package_install_target torchaudio "$OFFICIAL_TORCHAUDIO_VERSION" "$OFFICIAL_TORCHAUDIO_WHEEL_URL")

if [[ "$FORCE_REPAIR_OFFICIAL_ENV" != "1" ]]; then
  if health_check_output=$(verify_official_env 2>&1); then
    echo "[official-env] health check passed; existing env matches pinned official baseline"
    printf '%s\n' "$health_check_output"
    emit_prepared_env_summary
    exit 0
  fi

  printf '%s\n' "$health_check_output" >&2
  echo "[official-env] repairing env to match pinned official baseline"
else
  echo "[official-env] FORCE_REPAIR_OFFICIAL_ENV=1; skipping health check"
fi

FILTERED_REQUIREMENTS_DIR=$(mktemp -d)
trap 'rm -rf "$FILTERED_REQUIREMENTS_DIR"' EXIT
FILTERED_COMMON_REQUIREMENTS_FILE="$FILTERED_REQUIREMENTS_DIR/common.txt"
FILTERED_ASCEND_REQUIREMENTS_FILE="$FILTERED_REQUIREMENTS_DIR/requirements.txt"
FILTERED_BENCH_REQUIREMENTS_FILE="$FILTERED_REQUIREMENTS_DIR/requirements-bench.txt"
create_filtered_requirements_file "$OFFICIAL_VLLM_WORKTREE/requirements/common.txt" "$FILTERED_COMMON_REQUIREMENTS_FILE"
create_filtered_requirements_file "$OFFICIAL_VLLM_ASCEND_WORKTREE/requirements.txt" "$FILTERED_ASCEND_REQUIREMENTS_FILE"
create_filtered_requirements_file "$OFFICIAL_VLLM_ASCEND_WORKTREE/benchmarks/requirements-bench.txt" "$FILTERED_BENCH_REQUIREMENTS_FILE"

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
  uvloop \
  numba || true

run_with_ascend_env conda run -p "$ENV_PREFIX" python -m pip install --upgrade --force-reinstall \
  "setuptools>=77.0.3,<80.0.0"

run_with_ascend_env conda run -p "$ENV_PREFIX" python -m pip install --upgrade --force-reinstall \
  "${PIP_EXTRA_INDEX_ARGS[@]}" \
  "$OFFICIAL_TORCH_INSTALL_TARGET" \
  "$OFFICIAL_TORCH_NPU_INSTALL_TARGET" \
  "$OFFICIAL_TORCHVISION_INSTALL_TARGET" \
  "$OFFICIAL_TORCHAUDIO_INSTALL_TARGET"

run_with_ascend_env conda run -p "$ENV_PREFIX" python -m pip install --upgrade --force-reinstall \
  "${PIP_EXTRA_INDEX_ARGS[@]}" \
  -r "$FILTERED_COMMON_REQUIREMENTS_FILE" \
  -r "$FILTERED_ASCEND_REQUIREMENTS_FILE" \
  -r "$FILTERED_BENCH_REQUIREMENTS_FILE" \
  "setuptools>=77.0.3,<80.0.0" \
  "$OFFICIAL_TORCH_INSTALL_TARGET" \
  "$OFFICIAL_TORCH_NPU_INSTALL_TARGET" \
  "$OFFICIAL_TORCHVISION_INSTALL_TARGET" \
  "$OFFICIAL_TORCHAUDIO_INSTALL_TARGET" \
  "numpy==$OFFICIAL_NUMPY_VERSION" \
  "transformers==$OFFICIAL_TRANSFORMERS_VERSION" \
  "compressed-tensors==$OFFICIAL_COMPRESSED_TENSORS_VERSION" \
  "depyf==$OFFICIAL_DEPYF_VERSION" \
  "llguidance==$OFFICIAL_LLGUIDANCE_VERSION" \
  "xgrammar==$OFFICIAL_XGRAMMAR_VERSION" \
  "fastapi==$OFFICIAL_FASTAPI_VERSION" \
  "$OFFICIAL_UVLOOP_TARGET" \
  "numba==$OFFICIAL_NUMBA_VERSION" \
  "opencv-python-headless==$OFFICIAL_OPENCV_VERSION"

run_with_ascend_env conda run -p "$ENV_PREFIX" python -m pip install --upgrade --force-reinstall --no-deps \
  "${PIP_EXTRA_INDEX_ARGS[@]}" \
  "setuptools>=77.0.3,<80.0.0" \
  "$OFFICIAL_TORCH_INSTALL_TARGET" \
  "$OFFICIAL_TORCH_NPU_INSTALL_TARGET" \
  "$OFFICIAL_TORCHVISION_INSTALL_TARGET" \
  "$OFFICIAL_TORCHAUDIO_INSTALL_TARGET" \
  "numpy==$OFFICIAL_NUMPY_VERSION" \
  "transformers==$OFFICIAL_TRANSFORMERS_VERSION" \
  "compressed-tensors==$OFFICIAL_COMPRESSED_TENSORS_VERSION" \
  "depyf==$OFFICIAL_DEPYF_VERSION" \
  "llguidance==$OFFICIAL_LLGUIDANCE_VERSION" \
  "xgrammar==$OFFICIAL_XGRAMMAR_VERSION" \
  "fastapi==$OFFICIAL_FASTAPI_VERSION" \
  "$OFFICIAL_UVLOOP_TARGET" \
  "numba==$OFFICIAL_NUMBA_VERSION"

PYTHONPATH="$OFFICIAL_VLLM_ASCEND_WORKTREE:$OFFICIAL_VLLM_WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
  verify_official_env

emit_prepared_env_summary