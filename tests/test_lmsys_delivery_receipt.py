"""Ledger consistency controls; this test does not execute the native corpus."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lmsys_delivery_keeps_budget_and_corpus_boundaries_explicit():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    source = ledger["delivery_inventory"]["LMSYS_CHAT"]
    receipt = json.loads((ROOT / source["delivery_receipt"]).read_text("utf-8"))
    assert source["checked"] and source["status"] == "AVAILABLE"
    assert receipt["source_id"] == "LMSYS_CHAT"
    assert receipt["source_checkbox_eligible"]
    assert receipt["original_files"] == 6
    assert receipt["all_original_rows_bound_and_footer_counts_verified"] == 1000000
    assert receipt["fixed_original_conversation_positions"] == [0, 1, 2]
    assert receipt["native_complete_original_prefix_requests"] == 8
    assert receipt["finish_reasons"] == {"length": 8}
    assert receipt["completion_tokens"] == [receipt["output_budget"]] * 8
    assert receipt["public_and_unchanged_research_readbacks"] == 3
    assert receipt["actual_research_external_lock_cli_passed"]
    for key in (
        "default_research_lock_changed",
        "quality_scored",
        "live_episode_or_original_api_capture_claimed",
        "raw_wire_status_or_bytes_independently_saved",
        "performance_validated",
        "raw_data_published",
        "agent_accepted_terms",
    ):
        assert receipt[key] is False
