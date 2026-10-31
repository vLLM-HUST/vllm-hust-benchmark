"""Bound the completed source checkbox to the actual original corpus profile."""

import json
from pathlib import Path


def test_openmath2_completed_profile_does_not_claim_independent_quality():
    root = Path(__file__).resolve().parents[1]
    directory = root / "docs/workload-data"
    ledger = json.loads(
        (directory / "dataset-validation-status.json").read_text("utf-8")
    )
    item = ledger["delivery_inventory"]["OPENMATH2"]
    receipt = json.loads((root / item["delivery_receipt"]).read_text("utf-8"))
    historical = json.loads(
        (directory / "validation/openmath2-original-readiness-20261028.json").read_text(
            "utf-8"
        )
    )
    assert item["checked"] and receipt["source_checkbox_eligible"]
    assert not historical["source_checkbox_eligible"]
    assert receipt["acquired_original_shards"] == [0, 15, 31]
    assert receipt["registered_original_shards"] == 32
    assert receipt["original_rows_fully_equal_and_indexed"] == 1309949
    native = receipt["native_profile"]
    assert (
        native["original_cases"] == native["stop_count"] + native["length_count"] == 9
    )
    assert native["length_count"] == 6
    assert not native["teacher_or_expected_answer_sent_to_model"]
    provider = receipt["provider_and_research"]
    assert provider["installed_payloads_byte_equal"] == 56
    assert provider["fixed_current_public_and_research_readbacks"] == 9
    assert provider["model_calls_this_audit"] == 0
    assert not provider["index_rebuilt"]
    for claim in (
        "quality_scored",
        "independent_quality_benchmark_claim",
        "teacher_training_reproduction_claim",
        "whole_32_shard_execution_claim",
        "performance_claim",
    ):
        assert receipt[claim] is False
