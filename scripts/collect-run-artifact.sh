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

cd "$ARTIFACT_DIR"

# ─── 1. Environment Manifest ────────────────────────────────────────────────

# Build env-manifest.json via Python for robust JSON encoding (handles
# special characters in env var values, git output, etc.).
python3 -c '
import json, os, platform, subprocess, sys
from pathlib import Path

def _run(cmd, timeout=5):
    """Run a command and return stripped stdout, or empty string on error."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""

def _git_commit(repo):
    if not repo or not Path(repo).joinpath(".git").is_dir():
        return "not available"
    return _run(["git", "-C", repo, "rev-parse", "HEAD"])

manifest = {
    "manifest_version": "run-env-manifest/v1",
    "collected_at": os.popen("date -u +%Y-%m-%dT%H:%M:%SZ").read().strip(),
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
    "env_vars": {
        k: os.environ.get(k, "")
        for k in ["PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "HF_HOME",
                   "VLLM_CACHE_ROOT", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"]
    },
    "git_info": {
        "vllm_hust": _git_commit(os.environ.get("CURRENT_VLLM_HUST_REPO")),
        "vllm_ascend_hust": _git_commit(os.environ.get("CURRENT_VLLM_ASCEND_HUST_REPO")),
        "benchmark": _run(["git", "-C", os.environ.get("BENCHMARK_REPO_ROOT", ""),
                          "rev-parse", "HEAD"]),
    },
}

# Use the explicit pip-packages.json file rather than embedding a second copy
manifest["pip_packages"] = "see pip-packages.json"

json.dump(manifest, sys.stdout, indent=2, ensure_ascii=False)
' > env-manifest.json
echo "[collect] env-manifest.json written"

# ─── 1b. Pip packages (separate file for parsing convenience) ──────────────

python3 -m pip list --format=json 2>/dev/null > pip-packages.json || echo '[]' > pip-packages.json
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
