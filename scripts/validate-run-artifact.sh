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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

ARTIFACT_DIR="${1:?Usage: validate-run-artifact.sh <artifact-dir>}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

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
    ERRORS=$((ERRORS + 1))
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
    if [[ -z "$MISSING" ]]; then
      check "env-manifest.json has required fields" 0
    else
      echo "  missing env-manifest fields: $MISSING" >&2
      check "env-manifest.json has required fields" 1
    fi
    FROZEN_INPUT_ERRORS=$(python3 - <<'PY' || echo "parse-error"
import json

manifest = json.load(open("env-manifest.json", encoding="utf-8"))
if not manifest.get("frozen_inputs_required"):
    raise SystemExit(0)

errors = []
inputs = manifest.get("frozen_inputs") or {}
for field in ("image_id", "model_revision", "topology"):
    if not inputs.get(field):
        errors.append(f"frozen_inputs.{field} is empty")

for field in ("cann", "torch_npu_version"):
    value = inputs.get(field) or {}
    if not value.get("declared"):
        errors.append(f"frozen_inputs.{field}.declared is empty")
    if not value.get("detected"):
        errors.append(f"frozen_inputs.{field}.detected is empty")
    elif value.get("declared") != value.get("detected"):
        errors.append(
            f"frozen_inputs.{field} declared {value.get('declared')!r} "
            f"does not match detected {value.get('detected')!r}"
        )

for repo in ("vllm_hust", "vllm_ascend_hust"):
    value = (manifest.get("git_info") or {}).get(repo) or {}
    if not value.get("declared"):
        errors.append(f"git_info.{repo}.declared is empty")
    if value.get("declared") != value.get("observed"):
        errors.append(
            f"git_info.{repo} declared {value.get('declared')!r} "
            f"does not match observed {value.get('observed')!r}"
        )

campaign = manifest.get("campaign") or {}
for field in ("campaign_id", "coverage_class", "load_profile"):
    if not campaign.get(field):
        errors.append(f"campaign.{field} is empty")
coverage_class = campaign.get("coverage_class")
if coverage_class == "full-matrix" and campaign.get("point_role") != "checkpoint":
    errors.append("campaign.point_role must be checkpoint for full-matrix")
elif coverage_class == "targeted-pair":
    if campaign.get("point_role") not in ("baseline", "head"):
        errors.append("campaign.point_role must be baseline or head for targeted-pair")
    if not campaign.get("comparison_id"):
        errors.append("campaign.comparison_id is empty for targeted-pair")
elif coverage_class == "experimental":
    if campaign.get("point_role") or campaign.get("comparison_id"):
        errors.append("experimental campaign declares a point_role or comparison_id")
else:
    errors.append(f"campaign.coverage_class is unsupported: {coverage_class!r}")
if campaign.get("repetitions", 0) < 3:
    errors.append("campaign.repetitions is less than 3")

print("; ".join(errors))
PY
)
    if [[ -z "$FROZEN_INPUT_ERRORS" ]]; then
      check "formal campaign frozen provenance is complete" 0
    else
      echo "  $FROZEN_INPUT_ERRORS" >&2
      check "formal campaign frozen provenance is complete" 1
    fi

    OFFICIAL_SOURCE_ERRORS=$(python3 - <<'PY' || echo "parse-error"
import json

manifest = json.load(open("env-manifest.json", encoding="utf-8"))
payload = manifest.get("official_source_provenance")
if not payload:
    raise SystemExit(0)

errors = []
if payload.get("schema_version") != "official-source-provenance/v1":
    errors.append("unsupported official source provenance schema")
for role in ("engine", "plugin"):
    source = (payload.get("sources") or {}).get(role) or {}
    for field in ("repository", "requested_ref", "observed_commit", "tracked_patch_sha256", "working_tree_sha256", "status"):
        if not source.get(field):
            errors.append(f"official_source_provenance.{role}.{field} is empty")
    if source.get("status") not in ("clean", "modified"):
        errors.append(f"official_source_provenance.{role}.status is invalid")
    if source.get("status") == "modified" and not source.get("tracked_patch_sha256"):
        errors.append(f"official_source_provenance.{role} modified without patch digest")
    if source.get("status") == "modified" and not source.get("working_tree_sha256"):
        errors.append(f"official_source_provenance.{role} modified without tree digest")
print("; ".join(errors))
PY
)
    if [[ -z "$OFFICIAL_SOURCE_ERRORS" ]]; then
      check "official source provenance is complete" 0
    else
      echo "  $OFFICIAL_SOURCE_ERRORS" >&2
      check "official source provenance is complete" 1
    fi

    OFFICIAL_RUNTIME_ERRORS=$(python3 - <<'PY' || echo "parse-error"
import json
from pathlib import Path

env_manifest = json.load(open("env-manifest.json", encoding="utf-8"))
artifact = (
    json.load(open("run_leaderboard.json", encoding="utf-8"))
    if Path("run_leaderboard.json").is_file()
    else {}
)
leaderboard_manifest = (
    json.load(open("leaderboard_manifest.json", encoding="utf-8"))
    if Path("leaderboard_manifest.json").is_file()
    else {}
)
values = {
    "artifact": (artifact.get("metadata") or {}).get("official_runtime_provenance"),
    "leaderboard_manifest": leaderboard_manifest.get("official_runtime_provenance"),
    "env_manifest": env_manifest.get("official_runtime_provenance"),
}
if not any(values.values()):
    raise SystemExit(0)

errors = []
for location, value in values.items():
    if not value:
        errors.append(f"official_runtime_provenance missing from {location}")
if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
    errors.append("official_runtime_provenance differs across artifact and manifests")

payload = next((value for value in values.values() if value), {})
if payload.get("schema_version") != "official-runtime-provenance/v1":
    errors.append("unsupported official runtime provenance schema")
for field in ("python_executable", "python_version"):
    if not payload.get(field):
        errors.append(f"official_runtime_provenance.{field} is empty")
for role in ("engine", "plugin"):
    source = (payload.get("sources") or {}).get(role) or {}
    for field in (
        "module",
        "module_path",
        "module_version",
        "distribution",
        "distribution_version",
        "prepared_worktree",
        "prepared_commit",
        "source_version",
        "extension_policy",
        "source_patch_sha256",
        "source_tree_sha256",
        "source_status",
    ):
        if not source.get(field):
            errors.append(f"official_runtime_provenance.{role}.{field} is empty")
    extensions = source.get("extensions")
    if source.get("extension_policy") not in ("present", "none-discovered"):
        errors.append(f"official_runtime_provenance.{role}.extension_policy is invalid")
    if not isinstance(extensions, list):
        errors.append(f"official_runtime_provenance.{role}.extensions is not a list")
    elif bool(extensions) != (source.get("extension_policy") == "present"):
        errors.append(
            f"official_runtime_provenance.{role}.extension_policy does not match extensions"
        )
    else:
        for index, extension in enumerate(extensions):
            for field in ("module", "status", "path", "sha256"):
                if not (extension or {}).get(field):
                    errors.append(
                        f"official_runtime_provenance.{role}.extensions[{index}].{field} is empty"
                    )
print("; ".join(errors))
PY
)
    if [[ -z "$OFFICIAL_RUNTIME_ERRORS" ]]; then
      check "official runtime provenance is complete and consistent" 0
    else
      echo "  $OFFICIAL_RUNTIME_ERRORS" >&2
      check "official runtime provenance is complete and consistent" 1
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
