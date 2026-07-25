from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.trend_validator import validate_entries


FIXTURES = Path(__file__).parent / "fixtures" / "trend_coverage"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / "valid" / name).read_text(encoding="utf-8"))


def make_repeated_pair() -> list[dict]:
    baseline = fixture("targeted-pair.json")
    baseline["entry_id"] = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    baseline["point_role"] = "baseline"
    baseline["repeat_group"] = "pair/v1::baseline"
    baseline["repeat_index"] = 0
    baseline["canonical_aggregate"] = {
        "method": "mean", "count": 3, "metrics": {"ttft_ms": {"value": 40}}, "outlier_handling": "none"
    }
    head = copy.deepcopy(baseline)
    head["entry_id"] = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    head["point_role"] = "head"
    head["repeat_group"] = "pair/v1::head"
    head["repeat_index"] = 0
    return [baseline, head]


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
    entry = fixture("blocked.json")
    entry["repeat_group"] = "blocked/v1::head"
    entry["canonical_aggregate"] = {
        "method": "mean", "count": 3, "metrics": {"throughput_tps": {"value": 350}}, "outlier_handling": "none"
    }
    entry["repeat_index"] = 0
    report = validate_entries([entry])
    assert report.decisions[0].status == "blocked"
    assert "PAIR_HALF_MISSING" in {issue.code for issue in report.issues}
    assert "comparison_id" in report.decisions[0].reason


def test_complete_pair_and_full_matrix_are_admitted() -> None:
    pair = make_repeated_pair()
    matrix = fixture("full-matrix.json")
    report = validate_entries([*pair, matrix])
    statuses = {decision.entry_id: decision.status for decision in report.decisions}
    assert statuses["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"] == "default"
    assert statuses["bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"] == "default"
    assert statuses[matrix["entry_id"]] == "default"
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
