from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.trend_validator import validate_entries


FIXTURES = Path(__file__).parent / "fixtures" / "trend_coverage"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / "valid" / name).read_text(encoding="utf-8"))


def make_repeated_pair(side_count: int = 3) -> list[dict]:
    baseline = fixture("targeted-pair.json")
    baseline["point_role"] = "baseline"
    baseline["repeat_group"] = "pair/v1::baseline"
    baseline["canonical_aggregate"] = {
        "method": "mean", "count": 3, "metrics": {"ttft_ms": {"value": 40}}, "outlier_handling": "none"
    }
    entries = []
    for role, prefix, group in (
        ("baseline", "a", "pair/v1::baseline"),
        ("head", "b", "pair/v1::head"),
    ):
        for index in range(side_count):
            candidate = copy.deepcopy(baseline)
            candidate["entry_id"] = f"{prefix * 8}-{prefix * 4}-4{prefix * 3}-8{prefix * 3}-{prefix * 12}"
            candidate["point_role"] = role
            candidate["repeat_group"] = group
            candidate["repeat_index"] = index
            entries.append(candidate)
    return entries


@pytest.mark.parametrize("throughput", [None, 0])
def test_random_latency_missing_or_zero_throughput_is_blocked_not_invalid(throughput) -> None:
    entry = fixture("invalid.json")
    entry["metrics"]["throughput_tps"] = throughput
    report = validate_entries([entry])
    assert report.decisions[0].status == "blocked"
    assert "LATENCY_THROUGHPUT_NOT_APPLICABLE" in {issue.code for issue in report.issues}


def test_retired_qwen3_bf16_is_excluded() -> None:
    entry = fixture("experimental.json")
    entry["model"].update({"name": "Qwen/Qwen3-8B", "repo_id": "Qwen/Qwen3-8B", "short_name": "Qwen3-8B", "precision": "BF16"})
    report = validate_entries([entry])
    assert report.decisions[0].status == "excluded"


def test_w8a8_is_experimental() -> None:
    report = validate_entries([fixture("experimental.json")])
    assert report.decisions[0].status == "experimental"
    assert "W8A8" in report.decisions[0].reason


def test_blocked_half_pair_has_actionable_reason() -> None:
    entries = [entry for entry in make_repeated_pair() if entry["point_role"] == "head"]
    for entry in entries:
        entry["comparison_id"] = fixture("blocked.json")["comparison_id"]
        entry["repeat_group"] = "blocked/v1::head"
    report = validate_entries(entries)
    assert all(decision.status == "blocked" for decision in report.decisions)
    assert "PAIR_HALF_MISSING" in {issue.code for issue in report.issues}
    assert "comparison_id" in report.decisions[0].reason


def test_full_matrix_requires_raw_entries_for_declared_aggregate() -> None:
    report = validate_entries([fixture("full-matrix.json")])
    assert report.decisions[0].status == "blocked"
    assert "MATRIX_REPEAT_INCOMPLETE" in {issue.code for issue in report.issues}


def test_pair_is_blocked_when_counterpart_has_insufficient_raw_repeats() -> None:
    entries = make_repeated_pair(side_count=3)
    entries = [entry for entry in entries if entry["point_role"] == "baseline"] + [
        entry for entry in entries if entry["point_role"] == "head" and entry["repeat_index"] == 0
    ]
    report = validate_entries(entries)
    baseline_decisions = [decision for decision in report.decisions if decision.entry_id.startswith("a")]
    assert baseline_decisions
    assert all(decision.status == "blocked" for decision in baseline_decisions)
    assert "PAIR_COUNTERPART_REPEAT_INCOMPLETE" in {issue.code for issue in report.issues}


def test_complete_pair_and_full_matrix_are_admitted() -> None:
    pair = make_repeated_pair()
    matrix_template = fixture("full-matrix.json")
    matrix = []
    for index in range(3):
        candidate = copy.deepcopy(matrix_template)
        candidate["entry_id"] = f"c{index + 1:07d}-c00{index + 1}-4c0{index + 1}-8c0{index + 1}-{'c' * 12}"
        candidate["repeat_index"] = index
        matrix.append(candidate)
    report = validate_entries([*pair, *matrix])
    statuses = {decision.entry_id: decision.status for decision in report.decisions}
    assert statuses["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"] == "default"
    assert statuses["bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"] == "default"
    assert all(statuses[entry["entry_id"]] == "default" for entry in matrix)
    assert report.passed


def test_effective_config_contract_failure_is_blocked() -> None:
    entry = fixture("full-matrix.json")
    entry["same_spec"] = {
        "spec_id": "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2",
        "scenario": "random-online",
        "resolved_server_parameters": {},
        "resolved_client_parameters": {},
    }
    entry["metadata"] = {"submitted_at": "2026-07-25T00:00:00Z"}
    report = validate_entries([entry])
    assert report.decisions[0].status == "blocked"
    assert "EFFECTIVE_CONFIG_INVALID" in {issue.code for issue in report.issues}
