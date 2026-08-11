#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# collect-run-artifact.sh — Post-run collection: env manifest, checksums, STATUS
#
# Run this AFTER a benchmark run to collect environment info, compute artifact
# checksums, and write a STATUS file (OK or FAILED) for producer consumption.
#
# Usage:
#   collect-run-artifact.sh <artifact-dir> [--mark-failed <reason>]
#
# The script writes:
#   env-manifest.json   — OS, Python, packages, env vars, CANN info
#   checksums.sha256    — sha256 of every file in the artifact dir
#   STATUS              — "OK" on success, "FAILED: <reason>" on failure
#
# These files are consumed by the trend producer to decide which artifacts
# are valid.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ARTIFACT_DIR="${1:?Usage: collect-run-artifact.sh <artifact-dir> [--mark-failed <reason>]}"
MARK_FAILED=false
FAIL_REASON=""

shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mark-failed)
      MARK_FAILED=true
      FAIL_REASON="${2:-unknown error}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$ARTIFACT_DIR" ]]; then
  echo "Error: artifact directory does not exist: $ARTIFACT_DIR" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ARTIFACT_DIR"
export BENCHMARK_REPO_ROOT

# ─── 1. Environment Manifest ────────────────────────────────────────────────

# Build env-manifest.json via Python for robust JSON encoding (handles
# special characters in env var values, git output, etc.).
python3 - <<'PY' > env-manifest.json
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def _run(cmd, timeout=5):
    """Run a command and return stripped stdout, or empty string on error."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _git_commit(repo):
    if not repo:
        return "not available"
    commit = _run(["git", "-C", repo, "rev-parse", "HEAD"])
    return commit or "not available"


def _runtime_package_versions():
    runtime_python = os.environ.get("CURRENT_RUNTIME_PYTHON") or sys.executable
    script = """
import importlib.metadata
import json

packages = {}
for distribution in ("torch", "torch-npu", "vllm", "vllm-ascend", "vllm-ascend-hust"):
    try:
        packages[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        continue
print(json.dumps(packages, sort_keys=True))
"""
    raw = _run([runtime_python, "-c", script], timeout=20)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return result if isinstance(result, dict) else {}


def _detect_cann_version():
    declared = os.environ.get("CURRENT_CANN_VERSION", "")
    ascend_home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME", ""
    )
    candidates = (
        Path(ascend_home) / "version.cfg" if ascend_home else None,
        Path(ascend_home) / "version.info" if ascend_home else None,
        Path(ascend_home) / "opp/version.info" if ascend_home else None,
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
        Path("/usr/local/Ascend/latest/version.cfg"),
        Path("/opt/hust-ascend-cann/Ascend/ascend-toolkit/latest/version.cfg"),
    )
    detected = ""
    source = ""
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            raw = candidate.read_text(encoding="utf-8", errors="replace").strip()
            match = re.search(r'^Version\s*=\s*["\']?([^"\'\n]+)', raw, re.MULTILINE)
            detected = match.group(1).strip() if match else raw
            source = str(candidate)
            break
    return {"declared": declared, "detected": detected, "source": source}


def _env(name):
    return os.environ.get(name, "")


def _int_env(name):
    try:
        return int(_env(name) or 0)
    except ValueError:
        return 0


observed_core_commit = _git_commit(_env("CURRENT_VLLM_HUST_REPO"))
observed_backend_commit = _git_commit(_env("CURRENT_VLLM_ASCEND_HUST_REPO"))
package_versions = _runtime_package_versions()

official_source_provenance = {}
provenance_path = os.environ.get("OFFICIAL_SOURCE_PROVENANCE_FILE", "")
if provenance_path:
    try:
        candidate = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        candidate = None
    if isinstance(candidate, dict):
        official_source_provenance = candidate

manifest = {
    "manifest_version": "run-env-manifest/v2",
    "collected_at": _run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
    "os": _run(["uname", "-a"]),
    "python_version": _run([sys.executable, "--version"]),
    "hostname": platform.node(),
    "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
    "ascend_toolkit": (
        "detected at /usr/local/Ascend/ascend-toolkit/set_env.sh"
        if Path("/usr/local/Ascend/ascend-toolkit/set_env.sh").exists()
        else "detected at /usr/local/Ascend/latest/set_env.sh"
        if Path("/usr/local/Ascend/latest/set_env.sh").exists()
        else "not detected"
    ),
    "npu_smi": _run(["npu-smi", "info", "-t", "board", "-i", "0"]) or "npu-smi not available",
    "frozen_inputs_required": _env("CAMPAIGN_REQUIRE_FROZEN_INPUTS") == "1",
    "frozen_inputs": {
        "image_id": _env("CURRENT_IMAGE_ID"),
        "model_revision": _env("CURRENT_MODEL_REVISION"),
        "cann": _detect_cann_version(),
        "torch_npu_version": {
            "declared": _env("CURRENT_TORCH_NPU_VERSION"),
            "detected": package_versions.get("torch-npu", ""),
        },
        "topology": _env("CURRENT_TOPOLOGY"),
    },
    "campaign": {
        "campaign_id": _env("CAMPAIGN_ID"),
        "coverage_class": _env("CAMPAIGN_COVERAGE_CLASS"),
        "comparison_id": _env("CAMPAIGN_COMPARISON_ID"),
        "point_role": _env("CAMPAIGN_POINT_ROLE"),
        "load_profile": _env("CAMPAIGN_LOAD_PROFILE"),
        "repeat_index": _int_env("CAMPAIGN_REPEAT_INDEX"),
        "repetitions": _int_env("CAMPAIGN_REPETITIONS"),
    },
    "runtime_packages": package_versions,
    "env_vars": {
        k: _env(k)
        for k in ["PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "HF_HOME",
                   "VLLM_CACHE_ROOT", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"]
    },
    "git_info": {
        "vllm_hust": {
            "declared": _env("CURRENT_GIT_COMMIT"),
            "observed": observed_core_commit,
        },
        "vllm_ascend_hust": {
            "declared": _env("CURRENT_PLUGIN_GIT_COMMIT"),
            "observed": observed_backend_commit,
        },
        "benchmark": _git_commit(_env("BENCHMARK_REPO_ROOT")),
    },
}
if official_source_provenance:
    manifest["official_source_provenance"] = official_source_provenance

# Use the explicit pip-packages.json file rather than embedding a second copy
manifest["pip_packages"] = "see pip-packages.json"

json.dump(manifest, sys.stdout, indent=2, ensure_ascii=False)
PY
echo "[collect] env-manifest.json written"

# ─── 1b. Pip packages (separate file for parsing convenience) ──────────────

PIP_PYTHON="${CURRENT_RUNTIME_PYTHON:-python3}"
"$PIP_PYTHON" -m pip list --format=json 2>/dev/null > pip-packages.json || echo '[]' > pip-packages.json
echo "[collect] pip-packages.json written"

# ─── 2. Checksums ──────────────────────────────────────────────────────────

# Compute SHA256 for every file except checksums.sha256 itself and STATUS
find . -type f \
  ! -name 'checksums.sha256' \
  ! -name 'STATUS' \
  -exec sha256sum {} \; > checksums.sha256
echo "[collect] checksums.sha256 written"

# ─── 3. STATUS File ───────────────────────────────────────────────────────

if [[ "$MARK_FAILED" == "true" ]]; then
  echo "FAILED: ${FAIL_REASON}" > STATUS
  echo "[collect] STATUS = FAILED: ${FAIL_REASON}"
else
  echo "OK" > STATUS
  echo "[collect] STATUS = OK"
fi
