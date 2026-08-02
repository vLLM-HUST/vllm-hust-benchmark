#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run-campaign-repetitions.sh — Repetition loop for campaign runs
#
# Runs the same spec N times (default: 3) with independent artifact
# directories. Each repetition uses a sequential RUN_INDEX (1, 2, 3...).
#
# Usage:
#   run-campaign-repetitions.sh <spec-file> [--campaign-prefix <prefix>] [--repetitions N]
#
# Example:
#   export CURRENT_SUBMITTER=full-stack-jul-2026
#   export CURRENT_DATA_SOURCE=full-stack-jul-2026
#   export CURRENT_GIT_COMMIT=5536d0873fb41c4925d0e6e9112a1ea70faeeb3a
#   export CURRENT_PLUGIN_GIT_COMMIT=b42a66b63b73ceda32fb8983edf7de3c69cce516
#   export CURRENT_ENGINE_VERSION=v0.23.1rc0-1327-g5536d0873f
#
#   bash scripts/run-campaign-repetitions.sh \
#     docs/official-baselines/full-stack-jul-2026-random-online-qwen25-14b-2chip-910b2.json \
#     --campaign-prefix full-stack-jul-2026 \
#     --repetitions 3
#
# Output:
#   submissions/<campaign-prefix>-<workload>-<chip>chip-<ts>/  (N times)
#
# The script waits for GPU/NPU memory to be freed between repetitions
# by checking that no vLLM server process remains on the target port.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SINGLE_REPETITION_RUNNER=${SINGLE_REPETITION_RUNNER:-"$SCRIPT_DIR/run-single-repetition.sh"}

SPEC_FILE=""
CAMPAIGN_PREFIX=""
REPETITIONS=3
COOLDOWN_SECONDS=60
MAX_PORT_WAIT_SECONDS=120
CAMPAIGN_REQUIRE_FROZEN_INPUTS=${CAMPAIGN_REQUIRE_FROZEN_INPUTS:-0}
CAMPAIGN_SUMMARY_FILE=${CAMPAIGN_SUMMARY_FILE:-}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-prefix)
      CAMPAIGN_PREFIX="$2"
      shift 2
      ;;
    --repetitions)
      REPETITIONS="$2"
      shift 2
      ;;
    --cooldown)
      COOLDOWN_SECONDS="$2"
      shift 2
      ;;
    *)
      if [[ -z "$SPEC_FILE" ]]; then
        SPEC_FILE="$1"
      else
        echo "Unknown option: $1" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

if [[ -z "$SPEC_FILE" ]]; then
  echo "Usage: run-campaign-repetitions.sh <spec-file> [--campaign-prefix <prefix>] [--repetitions N]" >&2
  exit 2
fi

if [[ ! -f "$SPEC_FILE" ]]; then
  echo "Error: spec file not found: $SPEC_FILE" >&2
  exit 2
fi

if [[ ! -x "$SINGLE_REPETITION_RUNNER" ]]; then
  echo "Error: single-repetition runner is not executable: $SINGLE_REPETITION_RUNNER" >&2
  exit 2
fi

if ! [[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: repetitions must be a positive integer: $REPETITIONS" >&2
  exit 2
fi

if ! [[ "$COOLDOWN_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "Error: cooldown must be a non-negative integer: $COOLDOWN_SECONDS" >&2
  exit 2
fi

require_nonempty() {
  local name=$1
  if [[ -z "${!name:-}" ]]; then
    echo "Error: $name is required when CAMPAIGN_REQUIRE_FROZEN_INPUTS=1" >&2
    exit 2
  fi
}

require_git_commit() {
  local name=$1
  require_nonempty "$name"
  if ! [[ "${!name}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: $name must be a full 40-character Git commit: ${!name}" >&2
    exit 2
  fi
}

assert_repo_commit() {
  local repo_name=$1
  local repo_path=$2
  local expected=$3
  local observed

  if [[ ! -d "$repo_path" ]]; then
    echo "Error: $repo_name repository not found: $repo_path" >&2
    exit 2
  fi
  observed=$(git -C "$repo_path" rev-parse HEAD 2>/dev/null || true)
  if [[ "$observed" != "$expected" ]]; then
    echo "Error: $repo_name HEAD $observed does not match frozen commit $expected" >&2
    exit 2
  fi
}

if [[ "$CAMPAIGN_REQUIRE_FROZEN_INPUTS" == "1" ]]; then
  if ((REPETITIONS < 3)); then
    echo "Error: formal campaigns require at least 3 independent service repetitions" >&2
    exit 2
  fi

  require_git_commit CURRENT_GIT_COMMIT
  require_git_commit CURRENT_PLUGIN_GIT_COMMIT
  require_nonempty CURRENT_IMAGE_ID
  require_nonempty CURRENT_MODEL_REVISION
  require_nonempty CURRENT_CANN_VERSION
  require_nonempty CURRENT_TORCH_NPU_VERSION
  require_nonempty CURRENT_TOPOLOGY
  require_nonempty CAMPAIGN_ID
  require_nonempty CAMPAIGN_COVERAGE_CLASS
  require_nonempty CAMPAIGN_POINT_ROLE
  require_nonempty CAMPAIGN_LOAD_PROFILE
  require_nonempty ASCEND_RT_VISIBLE_DEVICES
  require_nonempty ASCEND_VISIBLE_DEVICES

  if ! [[ "$CURRENT_IMAGE_ID" =~ ^(sha256:)?[0-9a-fA-F]{64}$ ]]; then
    echo "Error: CURRENT_IMAGE_ID must be a full image digest or ID: $CURRENT_IMAGE_ID" >&2
    exit 2
  fi
  if ! [[ "$CURRENT_MODEL_REVISION" =~ ^[0-9a-fA-F]{40,64}$ ]]; then
    echo "Error: CURRENT_MODEL_REVISION must be an immutable 40-64 character revision: $CURRENT_MODEL_REVISION" >&2
    exit 2
  fi
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "$ASCEND_VISIBLE_DEVICES" ]]; then
    echo "Error: ASCEND_RT_VISIBLE_DEVICES and ASCEND_VISIBLE_DEVICES must select the same devices" >&2
    exit 2
  fi
  if ! [[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Error: visible devices must be an explicit comma-separated numeric list: $ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 2
  fi
  EXPECTED_CHIP_COUNT=$(jq -r '.chip_count // 1' "$SPEC_FILE")
  if ! [[ "$EXPECTED_CHIP_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: spec chip_count must be a positive integer: $EXPECTED_CHIP_COUNT" >&2
    exit 2
  fi
  IFS=',' read -r -a SELECTED_DEVICES <<< "$ASCEND_RT_VISIBLE_DEVICES"
  if [[ "${#SELECTED_DEVICES[@]}" -ne "$EXPECTED_CHIP_COUNT" ]]; then
    echo "Error: spec requires $EXPECTED_CHIP_COUNT chip(s), but $ASCEND_RT_VISIBLE_DEVICES selects ${#SELECTED_DEVICES[@]}" >&2
    exit 2
  fi
  if [[ "$(printf '%s\n' "${SELECTED_DEVICES[@]}" | sort -u | wc -l)" -ne "${#SELECTED_DEVICES[@]}" ]]; then
    echo "Error: visible device list contains a duplicate: $ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 2
  fi

  case "$CAMPAIGN_COVERAGE_CLASS" in
    full-matrix)
      if [[ "$CAMPAIGN_POINT_ROLE" != "checkpoint" ]]; then
        echo "Error: full-matrix campaigns require CAMPAIGN_POINT_ROLE=checkpoint" >&2
        exit 2
      fi
      ;;
    targeted-pair)
      require_nonempty CAMPAIGN_COMPARISON_ID
      if [[ "$CAMPAIGN_POINT_ROLE" != "baseline" && "$CAMPAIGN_POINT_ROLE" != "head" ]]; then
        echo "Error: targeted-pair campaigns require CAMPAIGN_POINT_ROLE=baseline or head" >&2
        exit 2
      fi
      ;;
    *)
      echo "Error: unsupported formal CAMPAIGN_COVERAGE_CLASS: $CAMPAIGN_COVERAGE_CLASS" >&2
      exit 2
      ;;
  esac

  if [[ "${PERFGATE_WARMUP_RUNS:-0}" != "0" || "${PERFGATE_MEASURED_RUNS:-1}" != "1" ]]; then
    echo "Error: formal independent-service campaigns require PERFGATE_WARMUP_RUNS=0 and PERFGATE_MEASURED_RUNS=1" >&2
    exit 2
  fi

  require_nonempty CURRENT_VLLM_HUST_REPO
  require_nonempty CURRENT_VLLM_ASCEND_HUST_REPO
  assert_repo_commit "vllm-hust" "$CURRENT_VLLM_HUST_REPO" "$CURRENT_GIT_COMMIT"
  assert_repo_commit "vllm-ascend-hust" "$CURRENT_VLLM_ASCEND_HUST_REPO" "$CURRENT_PLUGIN_GIT_COMMIT"
fi

# Infer campaign prefix from spec filename if not provided
if [[ -z "$CAMPAIGN_PREFIX" ]]; then
  SPEC_BASENAME=$(basename "$SPEC_FILE" .json)
  # Extract prefix: "full-stack-jul-2026-random-online..." -> "full-stack-jul-2026"
  CAMPAIGN_PREFIX=$(echo "$SPEC_BASENAME" | sed -E 's/(full-stack|targeted-pair|upstream-ref)-.*/\1/')
  if [[ -z "$CAMPAIGN_PREFIX" ]] || [[ "$CAMPAIGN_PREFIX" == "$SPEC_BASENAME" ]]; then
    echo "Error: could not infer campaign prefix from spec filename; use --campaign-prefix" >&2
    exit 2
  fi
  echo "[campaign] inferred campaign prefix: $CAMPAIGN_PREFIX"
fi

WORKLOAD_NAME=$(jq -r '.scenario // "unknown"' "$SPEC_FILE")
CHIP_COUNT=$(jq -r '.chip_count // 1' "$SPEC_FILE")

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Campaign Repetition Loop"
echo "  Spec:        $SPEC_FILE"
echo "  Workload:    $WORKLOAD_NAME"
echo "  Chip count:  $CHIP_COUNT"
echo "  Prefix:      $CAMPAIGN_PREFIX"
echo "  Repetitions: $REPETITIONS"
echo "  Cooldown:    ${COOLDOWN_SECONDS}s"
echo "═══════════════════════════════════════════════════════════════════════"

# ─── Wait for port helper ──────────────────────────────────────────────────

# Wait until no process is listening on the benchmark port
wait_for_port_free() {
  local port="$1"
  local waited=0

  # Default port used by run-current-ascend-same-spec.sh
  port="${port:-8001}"

  while (( waited < MAX_PORT_WAIT_SECONDS )); do
    if ! ss -ltnH "( sport = :${port} )" 2>/dev/null | grep -q .; then
      return 0
    fi
    echo "[campaign] waiting for port ${port} to be released (${waited}s/${MAX_PORT_WAIT_SECONDS}s)"
    sleep 5
    (( waited += 5 ))
  done

  echo "[campaign] WARNING: port ${port} still has listeners after ${MAX_PORT_WAIT_SECONDS}s; proceeding anyway" >&2
  return 1
}

# ─── Main repetition loop ──────────────────────────────────────────────────

SUCCESS_COUNT=0
FAIL_COUNT=0
FIRST_FAILURE_EXIT_CODE=0
ARTIFACT_DIRS=()
SUMMARY_ROWS_FILE=$(mktemp)
trap 'rm -f "$SUMMARY_ROWS_FILE"' EXIT

for (( i=1; i<=REPETITIONS; i++ )); do
  echo ""
  echo "───────────────────────────────────────────────────────────────────────"
  echo "  Repetition ${i}/${REPETITIONS}"
  echo "───────────────────────────────────────────────────────────────────────"

  # Wait for port to be free between repetitions (first run skips this)
  if (( i > 1 )); then
    # Also enforce a cooldown to let NPU memory settle
    echo "[campaign] cooldown ${COOLDOWN_SECONDS}s before next repetition..."
    sleep "$COOLDOWN_SECONDS"
    wait_for_port_free "${CURRENT_SERVER_PORT:-8001}" || true
  fi

  # Pin a single timestamp so run-single-repetition.sh uses the same
  # value even for long-running benchmarks that cross a clock-tick boundary.
  CAMPAIGN_RUN_TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
  export CAMPAIGN_RUN_TIMESTAMP
  export CAMPAIGN_REPEAT_INDEX=$((i - 1))
  export CAMPAIGN_REPETITIONS="$REPETITIONS"

  set +e
  bash "$SINGLE_REPETITION_RUNNER" \
    "$SPEC_FILE" \
    "$CAMPAIGN_PREFIX" \
    "$i"
  EXIT_CODE=$?
  set -e

  # Artifact dir is deterministic from the pinned timestamp.
  ARTIFACT_DIR="$REPO_ROOT/submissions/${CAMPAIGN_PREFIX}-${WORKLOAD_NAME}-${CHIP_COUNT}chip-${CAMPAIGN_RUN_TIMESTAMP}"
  printf '%s\t%s\t%s\n' "$CAMPAIGN_REPEAT_INDEX" "$EXIT_CODE" "$ARTIFACT_DIR" >> "$SUMMARY_ROWS_FILE"

  if [[ "$EXIT_CODE" -eq 0 ]]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    ARTIFACT_DIRS+=("$ARTIFACT_DIR")
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    if [[ "$FIRST_FAILURE_EXIT_CODE" -eq 0 ]]; then
      FIRST_FAILURE_EXIT_CODE="$EXIT_CODE"
    fi
    # On failure: decide whether to continue or abort
    if (( FAIL_COUNT >= REPETITIONS )); then
      echo "[campaign] all repetitions failed; aborting" >&2
      break
    fi
    echo "[campaign] repetition ${i} failed (exit ${EXIT_CODE}); continuing..." >&2
  fi
done

if [[ -n "$CAMPAIGN_SUMMARY_FILE" ]]; then
  mkdir -p "$(dirname "$CAMPAIGN_SUMMARY_FILE")"
  env SUMMARY_ROWS_FILE="$SUMMARY_ROWS_FILE" \
    CAMPAIGN_SUMMARY_FILE="$CAMPAIGN_SUMMARY_FILE" \
    CAMPAIGN_SPEC_FILE="$(realpath "$SPEC_FILE")" \
    CAMPAIGN_PREFIX_VALUE="$CAMPAIGN_PREFIX" \
    python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

rows = []
for line in Path(os.environ["SUMMARY_ROWS_FILE"]).read_text(encoding="utf-8").splitlines():
    repeat_index, exit_code, artifact_dir = line.split("\t", 2)
    rows.append(
        {
            "repeat_index": int(repeat_index),
            "exit_code": int(exit_code),
            "status": "ok" if exit_code == "0" else "failed",
            "artifact_dir": artifact_dir,
        }
    )

payload = {
    "schema_version": "independent-service-campaign-summary/v1",
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "campaign_id": os.environ.get("CAMPAIGN_ID", ""),
    "campaign_prefix": os.environ["CAMPAIGN_PREFIX_VALUE"],
    "coverage_class": os.environ.get("CAMPAIGN_COVERAGE_CLASS", ""),
    "comparison_id": os.environ.get("CAMPAIGN_COMPARISON_ID", ""),
    "point_role": os.environ.get("CAMPAIGN_POINT_ROLE", ""),
    "load_profile": os.environ.get("CAMPAIGN_LOAD_PROFILE", ""),
    "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
    "spec_file": os.environ["CAMPAIGN_SPEC_FILE"],
    "frozen_inputs": {
        "core_commit": os.environ.get("CURRENT_GIT_COMMIT", ""),
        "backend_commit": os.environ.get("CURRENT_PLUGIN_GIT_COMMIT", ""),
        "image_id": os.environ.get("CURRENT_IMAGE_ID", ""),
        "model_revision": os.environ.get("CURRENT_MODEL_REVISION", ""),
        "cann_version": os.environ.get("CURRENT_CANN_VERSION", ""),
        "torch_npu_version": os.environ.get("CURRENT_TORCH_NPU_VERSION", ""),
        "topology": os.environ.get("CURRENT_TOPOLOGY", ""),
    },
    "requested_repetitions": int(os.environ.get("CAMPAIGN_REPETITIONS", "0")),
    "successful_repetitions": sum(row["status"] == "ok" for row in rows),
    "runs": rows,
}
Path(os.environ["CAMPAIGN_SUMMARY_FILE"]).write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY
  echo "[campaign] summary: $CAMPAIGN_SUMMARY_FILE"
fi

# ─── Summary ───────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Campaign Repetition Summary"
echo "  Spec:        $SPEC_FILE"
echo "  Repetitions: ${REPETITIONS}"
echo "  Success:     ${SUCCESS_COUNT}"
echo "  Failed:      ${FAIL_COUNT}"
echo "═══════════════════════════════════════════════════════════════════════"

for dir in "${ARTIFACT_DIRS[@]}"; do
  if [[ -f "$dir/STATUS" ]]; then
    echo "  [$(cat "$dir/STATUS")] $dir"
  fi
done
echo ""

if (( FAIL_COUNT > 0 )); then
  echo "[campaign] ❌ ${FAIL_COUNT}/${REPETITIONS} repetitions failed" >&2
  exit "$FIRST_FAILURE_EXIT_CODE"
else
  echo "[campaign] ✅ All ${REPETITIONS} repetitions completed successfully"
fi
