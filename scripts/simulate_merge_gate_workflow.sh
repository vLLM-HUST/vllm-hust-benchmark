#!/usr/bin/env bash
# 本地模拟 .github/workflows/merge-gate.yml 的 mock 模式 job。
# 不需要 GitHub Actions，直接在本地跑全套场景，验证 merge-gate-check 接线正确性。
#
# 用法：
#   bash scripts/simulate_merge_gate_workflow.sh [scenario]
#   scenario 可选，默认 all（跑全部场景）
#
# 退出码：全部场景期望匹配 → 0；任一场景不匹配 → 1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-.venv/bin/python}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/tmp/merge-gate-simulate}"

SCENARIO="${1:-all}"
cd "$REPO_ROOT"

mkdir -p "$ARTIFACTS_DIR"

if [[ "$SCENARIO" == "all" ]]; then
  SCENARIOS="pass fail_config_drift fail_config_drift_head_only fail_data_source fail_unpaired_spec fail_3b_not_14b fail_missing_artifact fail_missing_target_declaration skip_docs_only skip_docs_only_no_approver specialty_valid specialty_no_reason fail_registry_hash_mismatch"
else
  SCENARIOS="$SCENARIO"
fi

PR_REPO="vllm-hust"
PR_NUMBER="193"
PR_HEAD_SHA="abc1234"
PR_BASE_SHA="def5678"

overall_rc=0

for scenario in $SCENARIOS; do
  echo "=== Scenario: $scenario ==="
  mock_dir="$ARTIFACTS_DIR/$scenario"
  rm -rf "$mock_dir"
  "$PYTHON" scripts/generate_mock_merge_gate_artifacts.py \
    --scenario "$scenario" --output-dir "$mock_dir" >/dev/null

  manifest="$mock_dir/scenario_manifest.json"
  expected=$("$PYTHON" -c "import json; print(json.load(open('$manifest'))['expected_disposition'])")
  expected_head_status=$("$PYTHON" -c "import json; m=json.load(open('$manifest')); print(m.get('expected_head_status','accepted'))" 2>/dev/null || echo "accepted")
  labels=$("$PYTHON" -c "import json; m=json.load(open('$manifest')); print(','.join(m.get('pr_labels',[])))" 2>/dev/null || echo "")
  specialty_spec=$("$PYTHON" -c "import json; m=json.load(open('$manifest')); print(m.get('specialty_spec','') or '')" 2>/dev/null || echo "")
  specialty_reason=$("$PYTHON" -c "import json; m=json.load(open('$manifest')); print(m.get('specialty_reason','') or '')" 2>/dev/null || echo "")
  declared_target=$("$PYTHON" -c "import json; m=json.load(open('$manifest')); print(m.get('declared_target_id','') or '')" 2>/dev/null || echo "")
  declared_target_version=$("$PYTHON" -c "import json; m=json.load(open('$manifest')); print(m.get('declared_target_version','') or '')" 2>/dev/null || echo "")
  declared_profile_id=$("$PYTHON" -c "import json; m=json.load(open('$manifest')); print(m.get('declared_profile_id','') or '')" 2>/dev/null || echo "")
  skip_approver=$("$PYTHON" -c "import json; m=json.load(open('$manifest')); print(m.get('skip_approver','') or '')" 2>/dev/null || echo "")

  base_artifact=""
  if [[ -f "$mock_dir/base/run_leaderboard.json" ]]; then
    base_artifact="$mock_dir/base/run_leaderboard.json"
  fi
  head_artifact=""
  if [[ -f "$mock_dir/head/run_leaderboard.json" ]]; then
    head_artifact="$mock_dir/head/run_leaderboard.json"
  fi

  base_status="accepted"
  head_status="accepted"
  if [[ "$expected_head_status" == "missing" ]]; then
    head_status="missing"
  fi

  cli_args=(
    merge-gate-check
    --repo "$PR_REPO"
    --pr-number "$PR_NUMBER"
    --head-sha "$PR_HEAD_SHA"
    --base-sha "$PR_BASE_SHA"
    --base-status "$base_status"
    --head-status "$head_status"
    --decision-output "$mock_dir/merge-gate-decision.json"
  )
  [[ -n "$base_artifact" ]] && cli_args+=(--base-artifact "$base_artifact")
  [[ -n "$head_artifact" ]] && cli_args+=(--head-artifact "$head_artifact")
  [[ -n "$labels" ]] && cli_args+=(--labels "$labels")
  [[ -n "$specialty_spec" ]] && cli_args+=(--specialty-spec "$specialty_spec")
  [[ -n "$specialty_reason" ]] && cli_args+=(--specialty-reason "$specialty_reason")
  [[ -n "$declared_target" ]] && cli_args+=(--declared-target-id "$declared_target")
  [[ -n "$declared_target_version" ]] && cli_args+=(--declared-target-version "$declared_target_version")
  [[ -n "$declared_profile_id" ]] && cli_args+=(--declared-profile-id "$declared_profile_id")
  [[ -n "$skip_approver" ]] && cli_args+=(--skip-approver "$skip_approver")

  set +e
  "$PYTHON" -m vllm_hust_benchmark.cli "${cli_args[@]}" >/dev/null 2>&1
  rc=$?
  set -e

  actual=$("$PYTHON" -c "import json; print(json.load(open('$mock_dir/merge-gate-decision.json'))['disposition'])")
  echo "  expected=$expected actual=$actual exit=$rc"

  if [[ "$expected" == "pass" || "$expected" == "skip" ]]; then
    if [[ $rc -ne 0 ]]; then
      echo "  ❌ FAIL: expected exit 0, got $rc"
      overall_rc=1
    elif [[ "$actual" != "$expected" ]]; then
      echo "  ❌ FAIL: expected disposition=$expected, got $actual"
      overall_rc=1
    else
      echo "  ✅ OK"
    fi
  elif [[ "$expected" == "fail" ]]; then
    if [[ $rc -ne 1 ]]; then
      echo "  ❌ FAIL: expected exit 1 (fail), got $rc"
      overall_rc=1
    elif [[ "$actual" != "fail" ]]; then
      echo "  ❌ FAIL: expected disposition=fail, got $actual"
      overall_rc=1
    else
      echo "  ✅ OK"
    fi
  fi
  echo ""
done

if [[ $overall_rc -eq 0 ]]; then
  echo "=== All scenarios passed ==="
else
  echo "=== Some scenarios failed ==="
fi
exit $overall_rc
