#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run-single-repetition.sh — Single repetition wrapper for campaign runs
#
# Wraps run-current-ascend-same-spec.sh with:
#   - Unique artifact directory per run (no overwrite)
#   - Environment manifest capture
#   - Log preservation (server + client logs)
#   - Checksum computation
#   - Failure marker on error
#   - Pre-flight no-overwrite check
#
# Usage:
#   run-single-repetition.sh <spec-file> <campaign-prefix> <run-index>
#
# Example:
#   export CURRENT_SUBMITTER=full-stack-jul-2026
#   export CURRENT_DATA_SOURCE=full-stack-jul-2026
#   run-single-repetition.sh \
#     docs/official-baselines/full-stack-jul-2026-random-online-qwen25-14b-2chip-910b2.json \
#     full-stack-jul-2026 \
#     1
#
# Environment variables (passed through to run-current-ascend-same-spec.sh):
#   CURRENT_SUBMITTER, CURRENT_DATA_SOURCE, CURRENT_ENGINE_VERSION,
#   CURRENT_GIT_COMMIT, CURRENT_PLUGIN_GIT_COMMIT, etc.
#
# Artifact directory naming (T18 convention):
#   submissions/<campaign-prefix>-<workload-name>-<chip-count>chip-<run-timestamp>/
#
# Output structure:
#   <artifact-dir>/
#     run_leaderboard.json       — main artifact (from export-leaderboard-artifact)
#     leaderboard_manifest.json  — manifest referencing the artifact
#     env-manifest.json          — environment snapshot
#     pip-packages.json          — pip package list
#     checksums.sha256           — sha256 of every file
#     STATUS                     — "OK" or "FAILED: <reason>"
#     server.stdout.log          — server stdout (if serve-type benchmark)
#     raw_benchmark_result.json  — raw benchmark output
#     resolved_same_spec.json    — resolved same-spec file
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
export BENCHMARK_REPO_ROOT="$REPO_ROOT"

SPEC_FILE="${1:?Usage: run-single-repetition.sh <spec-file> <campaign-prefix> <run-index>}"
CAMPAIGN_PREFIX="${2:?Usage: run-single-repetition.sh <spec-file> <campaign-prefix> <run-index>}"
RUN_INDEX="${3:?Usage: run-single-repetition.sh <spec-file> <campaign-prefix> <run-index>}"

if [[ ! -f "$SPEC_FILE" ]]; then
  echo "Error: spec file not found: $SPEC_FILE" >&2
  exit 2
fi

# ─── Parse spec for artifact naming ─────────────────────────────────────────

WORKLOAD_NAME=$(jq -r '.scenario // "unknown"' "$SPEC_FILE")
CHIP_COUNT=$(jq -r '.chip_count // 1' "$SPEC_FILE")

# Timestamp: campaign loop can pin a single timestamp via CAMPAIGN_RUN_TIMESTAMP
# so both the loop and the runner agree on the artifact directory name.
TIMESTAMP="${CAMPAIGN_RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

# Artifact directory name per T18 convention:
#   <campaign-prefix>-<workload-name>-<chip-count>chip-<timestamp>
ARTIFACT_DIR_NAME="${CAMPAIGN_PREFIX}-${WORKLOAD_NAME}-${CHIP_COUNT}chip-${TIMESTAMP}"

# Determine output directories
SUBMISSIONS_DIR="$REPO_ROOT/submissions"
ARTIFACT_DIR="$SUBMISSIONS_DIR/$ARTIFACT_DIR_NAME"
RUNNER_STATE_DIR="$REPO_ROOT/.benchmarks/${CAMPAIGN_PREFIX}/${ARTIFACT_DIR_NAME}"

# ─── Pre-flight: no-overwrite check ────────────────────────────────────────

if [[ -d "$ARTIFACT_DIR" ]]; then
  # Check if artifact already has an "OK" STATUS — refuse to overwrite
  if [[ -f "$ARTIFACT_DIR/STATUS" ]] && grep -qx "OK" "$ARTIFACT_DIR/STATUS" 2>/dev/null; then
    echo "Error: artifact directory already exists with STATUS=OK: $ARTIFACT_DIR" >&2
    echo "  Use a different RUN_INDEX or remove the existing directory." >&2
    exit 3
  fi
  # If STATUS is not OK (failed/incomplete), we can overwrite
  echo "[run-single] WARNING: overwriting incomplete/stale artifact: $ARTIFACT_DIR" >&2
fi

if [[ -d "$RUNNER_STATE_DIR" ]]; then
  echo "[run-single] WARNING: runner state directory exists; cleaning: $RUNNER_STATE_DIR" >&2
  rm -rf "$RUNNER_STATE_DIR"
fi

# ─── Setup: create runtime dirs ─────────────────────────────────────────────

mkdir -p "$RUNNER_STATE_DIR"
mkdir -p "$ARTIFACT_DIR"

# Unique RUN_ID for this repetition
export RUN_ID="${CAMPAIGN_PREFIX}-${WORKLOAD_NAME}-${CHIP_COUNT}chip-${TIMESTAMP}-r${RUN_INDEX}"
export RESULT_DIR="$RUNNER_STATE_DIR"

echo "[run-single] RUN_ID:     $RUN_ID"
echo "[run-single] SPEC:       $SPEC_FILE"
echo "[run-single] ARTIFACT:   $ARTIFACT_DIR"
echo "[run-single] STATE:      $RUNNER_STATE_DIR"

# Redirect server logs to a known location
export SERVER_STDOUT_LOG="$RUNNER_STATE_DIR/server.stdout.log"

# ─── Run: call existing benchmark runner ────────────────────────────────────

set +e
bash "$SCRIPT_DIR/run-current-ascend-same-spec.sh" "$SPEC_FILE"
RUN_EXIT_CODE=$?
set -e

# ─── Collect: copy results, logs, env manifest ─────────────────────────────

# Copy the exported artifact (run_leaderboard.json + leaderboard_manifest.json)
if [[ -d "$RUNNER_STATE_DIR/submission" ]]; then
  cp -a "$RUNNER_STATE_DIR/submission/." "$ARTIFACT_DIR/"
  echo "[run-single] artifact files copied from submission/"
fi

# Copy raw benchmark result
if [[ -f "$RUNNER_STATE_DIR/raw_benchmark_result.json" ]]; then
  cp "$RUNNER_STATE_DIR/raw_benchmark_result.json" "$ARTIFACT_DIR/"
fi

# Copy resolved same-spec
if [[ -f "$RUNNER_STATE_DIR/resolved_same_spec.json" ]]; then
  cp "$RUNNER_STATE_DIR/resolved_same_spec.json" "$ARTIFACT_DIR/"
fi

# Copy server log (serve-type benchmarks)
if [[ -f "$RUNNER_STATE_DIR/server.stdout.log" ]]; then
  cp "$RUNNER_STATE_DIR/server.stdout.log" "$ARTIFACT_DIR/"
fi

# Copy offline graph proof (for throughput/latency benchmarks)
if [[ -f "$RUNNER_STATE_DIR/offline_graph_proof.json" ]]; then
  cp "$RUNNER_STATE_DIR/offline_graph_proof.json" "$ARTIFACT_DIR/"
fi

# ─── Post-process: env manifest, checksums, STATUS ─────────────────────────

COLLECT_SCRIPT="$SCRIPT_DIR/collect-run-artifact.sh"

if [[ "$RUN_EXIT_CODE" -eq 0 ]]; then
  # Verify artifact exists
  if [[ ! -f "$ARTIFACT_DIR/run_leaderboard.json" ]]; then
    echo "[run-single] ERROR: run_leaderboard.json not found despite exit code 0" >&2
    bash "$COLLECT_SCRIPT" "$ARTIFACT_DIR" --mark-failed "run_leaderboard.json missing after successful exit"
    exit 4
  fi
  bash "$COLLECT_SCRIPT" "$ARTIFACT_DIR"
  echo "[run-single] ✅ Repetition ${RUN_INDEX} completed successfully"
  echo "[run-single] Artifact: $ARTIFACT_DIR"
else
  echo "[run-single] ❌ Repetition ${RUN_INDEX} failed (exit code ${RUN_EXIT_CODE})" >&2
  bash "$COLLECT_SCRIPT" "$ARTIFACT_DIR" --mark-failed "benchmark exit code ${RUN_EXIT_CODE}"
fi

exit "$RUN_EXIT_CODE"
