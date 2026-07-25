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

SPEC_FILE=""
CAMPAIGN_PREFIX=""
REPETITIONS=3
COOLDOWN_SECONDS=60
MAX_PORT_WAIT_SECONDS=120

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

  set +e
  bash "$SCRIPT_DIR/run-single-repetition.sh" \
    "$SPEC_FILE" \
    "$CAMPAIGN_PREFIX" \
    "$i"
  EXIT_CODE=$?
  set -e

  # Artifact dir is deterministic from the pinned timestamp.
  ARTIFACT_DIR="$REPO_ROOT/submissions/${CAMPAIGN_PREFIX}-${WORKLOAD_NAME}-${CHIP_COUNT}chip-${CAMPAIGN_RUN_TIMESTAMP}"

  if [[ "$EXIT_CODE" -eq 0 ]]; then
    (( SUCCESS_COUNT++ ))
    ARTIFACT_DIRS+=("$ARTIFACT_DIR")
  else
    (( FAIL_COUNT++ ))
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
