"""Frozen original corpus execution evidence; not a live model test."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = "docs/workload-data/validation/ultrachat-original-continuation-20261026.json"


def test_ultrachat_all_original_positions_not_a_shortened_live_episode():
    receipt = json.loads((ROOT / RECEIPT).read_text("utf-8"))
    native = receipt["native_profile"]
    assert receipt["source_checkbox_eligible"] is True
    assert native["selected_original_rows"] == [0, 1, 2]
    assert native["conversations_completed"] == 3
    assert native["all_original_assistant_positions_requested"] == 7
    assert native["request_original_row_indices"] == [0, 0, 1, 1, 2, 2, 2]
    assert native["request_original_prefix_lengths"] == [1, 3, 1, 3, 1, 3, 5]
    assert native["finish_reasons"] == {"stop": 7}
    assert native["complete_original_history_teacher_forced"] is True
    assert native["target_and_future_messages_excluded_from_requests"] is True
    assert (
        native["saved_requests_responses_usage_and_file_hash_independently_reverified"]
        is True
    )
    assert native["task_edits_or_history_trimming"] is False
    for key in (
        "quality_scored",
        "online_agent_episode_claim",
        "original_api_capture_claim",
        "full_corpus_model_execution_claim",
        "raw_payload_or_prediction_published",
    ):
        assert receipt[key] is False


def test_corpus_full_fields_fixed_install_and_research_consumer_are_separate_from_native():
    receipt = json.loads((ROOT / RECEIPT).read_text("utf-8"))
    consumer = receipt["fixed_consumer"]
    assert (
        consumer["all_original_rows_roles_content_order_prompt_and_ids_preserved"]
        == 28304
    )
    assert consumer["prepared_index_rows"] == 28304
    assert consumer["exact_installed_payload_files"] == 41
    assert consumer["public_and_research_roundtrips_equal"] == 6
    assert consumer["public_and_research_roundtrip_indices"] == [
        0,
        1,
        2,
        28301,
        28302,
        28303,
    ]
    assert consumer["research_top_level_reader_is_public_provider"] is True
    assert not consumer["research_default_lock_changed"]
    assert not consumer["feature_or_observation_executed"]
    assert consumer["new_model_calls"] == 0
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["ULTRACHAT"]
    assert row["status"] == "AVAILABLE" and row["checked"] is True
    assert row["delivery_receipt"] == RECEIPT
