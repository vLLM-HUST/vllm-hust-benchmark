"""Software-only original file identities and evidence separation contracts."""

from copy import deepcopy

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def test_three_original_variants_are_pinned_without_equating_their_histories():
    spec = source_spec("LONGMEMEVAL")
    expected_revision = "98d7416c24c778c2fee6e6f3006e7a073259d48f"  # pragma: allowlist secret - public publisher revision
    assert spec["revision"] == expected_revision
    assert {f["publisher_variant"] for f in spec["files"]} == {
        "oracle",
        "s_cleaned",
        "m_cleaned",
    }
    assert len({f["sha256"] for f in spec["files"]}) == 3
    for asset in spec["files"]:
        assert asset["bytes"] > 0 and len(asset["sha256"]) == 64
        assert f"/resolve/{spec['revision']}/" in asset["url"]
        assert asset["url"].endswith("/" + asset["path"])


def example():
    return {
        "question_id": "fixture-only",
        "question_type": "knowledge-update",
        "question": "What is current?",
        "question_date": "2024/01/02",
        "haystack_sessions": [
            [{"role": "user", "content": "New value", "has_answer": True}]
        ],
        "haystack_dates": ["2024/01/01"],
        "haystack_session_ids": ["session-1"],
        "answer": "New value",
        "answer_session_ids": ["session-1"],
    }


def test_original_nested_evidence_can_be_restored_but_is_not_model_input():
    row = example()
    original = deepcopy(row)
    record = adapt_row(
        "LONGMEMEVAL",
        row,
        source={"source_id": "LONGMEMEVAL"},
        spec=adapter_spec("LONGMEMEVAL"),
        index=0,
    )
    assert row == original
    assert "has_answer" not in record.task["input"]["haystack_sessions"][0][0]
    outcome = record.outcome_record()
    assert outcome["answer"] == row["answer"]
    restored = deepcopy(record.task["input"]["haystack_sessions"])
    for label in outcome["turn_evidence_labels"]:
        restored[label["session_index"]][label["turn_index"]]["has_answer"] = label[
            "has_answer"
        ]
    assert restored == row["haystack_sessions"]


def test_session_date_relationship_must_not_be_silently_repaired():
    row = example()
    row["haystack_dates"] = []
    with pytest.raises(ValueError, match="unaligned"):
        adapt_row(
            "LONGMEMEVAL", row, source={}, spec=adapter_spec("LONGMEMEVAL"), index=0
        )
