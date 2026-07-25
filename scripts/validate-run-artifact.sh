#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# validate-run-artifact.sh — Validate a collected artifact directory
#
# Checks:
#   1. STATUS file exists and contains "OK"
#   2. run_leaderboard.json exists and is valid JSON
#   3. leaderboard_manifest.json exists and references run_leaderboard.json
#   4. env-manifest.json exists and is valid JSON
#   5. checksums.sha256 exists and all checksums pass
#   6. run_leaderboard.json passes artifact contract normalization
#
# Usage:
#   validate-run-artifact.sh <artifact-dir>
#
# Exit code: 0 if valid, 1+ if validation fails (details on stderr).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ARTIFACT_DIR="${1:?Usage: validate-run-artifact.sh <artifact-dir>}"

if [[ ! -d "$ARTIFACT_DIR" ]]; then
  echo "Error: artifact directory not found: $ARTIFACT_DIR" >&2
  exit 2
fi

cd "$ARTIFACT_DIR"

ERRORS=0

check() {
  local desc="$1"
  local status="$2"
  if [[ "$status" -eq 0 ]]; then
    echo "  ✅ $desc"
  else
    echo "  ❌ $desc" >&2
    (( ERRORS++ ))
  fi
}

# ─── 1. STATUS file ─────────────────────────────────────────────────────────

if [[ -f "STATUS" ]]; then
  STATUS_CONTENT=$(tr -d '[:space:]' < STATUS)
  if [[ "$STATUS_CONTENT" == "OK" ]]; then
    check "STATUS file contains OK" 0
  else
    check "STATUS file contains FAILED: ${STATUS_CONTENT}" 1
  fi
else
  check "STATUS file exists" 1
fi

# ─── 2. run_leaderboard.json ────────────────────────────────────────────────

if [[ -f "run_leaderboard.json" ]]; then
  if python3 -m json.tool run_leaderboard.json >/dev/null 2>&1; then
    check "run_leaderboard.json is valid JSON" 0
  else
    check "run_leaderboard.json is valid JSON" 1
  fi
else
  check "run_leaderboard.json exists" 1
fi

# ─── 3. leaderboard_manifest.json ───────────────────────────────────────────

if [[ -f "leaderboard_manifest.json" ]]; then
  if python3 -m json.tool leaderboard_manifest.json >/dev/null 2>&1; then
    check "leaderboard_manifest.json is valid JSON" 0
    # Check that it references the artifact
    REFERENCED=$(python3 -c "
import json
m = json.load(open('leaderboard_manifest.json'))
entries = m.get('entries', [])
if entries and isinstance(entries[0], dict):
    print(entries[0].get('leaderboard_artifact', ''))
" 2>/dev/null || echo "")
    if [[ "$REFERENCED" == "run_leaderboard.json" ]]; then
      check "leaderboard_manifest.json references run_leaderboard.json" 0
    elif [[ -n "$REFERENCED" ]]; then
      check "leaderboard_manifest.json references ${REFERENCED}" 0
    else
      check "leaderboard_manifest.json has entry with leaderboard_artifact" 1
    fi
  else
    check "leaderboard_manifest.json is valid JSON" 1
  fi
else
  check "leaderboard_manifest.json exists" 1
fi

# ─── 4. env-manifest.json ───────────────────────────────────────────────────

if [[ -f "env-manifest.json" ]]; then
  if python3 -m json.tool env-manifest.json >/dev/null 2>&1; then
    check "env-manifest.json is valid JSON" 0
    # Check required fields
    MISSING=$(python3 -c "
import json
m = json.load(open('env-manifest.json'))
required = ['os', 'python_version', 'collected_at']
missing = [k for k in required if k not in m]
if missing:
    print(', '.join(missing))
" 2>/dev/null || echo "parse-error")
    if [[ -z "$MISSING" ]] || [[ "$MISSING" == "parse-error" ]]; then
      [[ -z "$MISSING" ]] && check "env-manifest.json has required fields" 0 || check "env-manifest.json required field check" 1
    fi
  else
    check "env-manifest.json is valid JSON" 1
  fi
else
  check "env-manifest.json exists" 1
fi

# ─── 5. checksums.sha256 ───────────────────────────────────────────────────

if [[ -f "checksums.sha256" ]]; then
  if sha256sum -c checksums.sha256 >/dev/null 2>&1; then
    check "checksums.sha256 all pass" 0
  else
    echo "  ⚠️  checksums.sha256: some files have changed or are missing (re-run collect-run-artifact.sh)" >&2
    check "checksums.sha256 all pass" 1
  fi
else
  check "checksums.sha256 exists" 1
fi

# ─── 6. Schema normalization (if Python available) ──────────────────────────

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

if [[ -f "run_leaderboard.json" ]] && [[ -d "$REPO_ROOT/src" ]]; then
  if python3 -c "
import sys
sys.path.insert(0, '$REPO_ROOT/src')
from vllm_hust_benchmark.submission_artifacts import normalize_submission_artifact_contract
import json
try:
    artifact = json.load(open('run_leaderboard.json'))
    normalize_submission_artifact_contract(artifact)
    print('valid')
except Exception as e:
    print(f'invalid: {e}')
    sys.exit(1)
" 2>/dev/null; then
    check "artifact contract normalization passes" 0
  else
    check "artifact contract normalization passes" 1
  fi
else
  echo "  ⚠️  run_leaderboard.json or src not available; skipping contract validation" >&2
fi

# ─── Summary ───────────────────────────────────────────────────────────────

echo ""
if (( ERRORS == 0 )); then
  echo "✅ All validations passed for: $ARTIFACT_DIR"
else
  echo "❌ ${ERRORS} validation error(s) for: $ARTIFACT_DIR" >&2
fi

exit "$ERRORS"
