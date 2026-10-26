"""Frozen original corpus execution evidence; not a live model test."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = "docs/workload-data/validation/ultrachat-original-continuation-20261026.json"


def test_openthoughts_sampled_corpus_delivery_does_not_upgrade_budget_exhaustion_to_quality():
    path = "docs/workload-data/validation/openthoughts3-native-delivery-20261027.json"
    receipt = json.loads((ROOT / path).read_text("utf-8"))
    assert receipt["source_checkbox_eligible"] is True
    assert (
        receipt["delivery_kind"]
        == "original_training_corpus_continuation_not_independent_quality_benchmark"
    )
    native = receipt["native_profile"]
    assert native["saved_generations"] == 9
    assert native["declared_output_token_budget"] == 32768
    assert native["all_saved_results_budget_exhausted_not_final_answers"] is True
    assert native["valid_generations_rerun"] == 0
    assert native["earlier_client_timeout_attempts_preserved"] == 1
    assert [
        (p["domain"], p["shard"], p["original_rows"], p["finish_reasons"])
        for p in native["profiles"]
    ] == [
        ("code", 0, [0, 1, 2], {"length": 3}),
        ("math", 25, [0, 1, 2], {"length": 3}),
        ("science", 110, [0, 1, 2], {"length": 3}),
    ]
    consumer = receipt["provider_and_research"]
    assert consumer["full_selected_original_index_rows"] == 30000
    assert consumer["installed_payloads_byte_equal"] == 46
    assert consumer["fixed_research_roundtrips"] == 18
    assert receipt["quality_scored"] is False
    assert receipt["independent_quality_benchmark_claim"] is False
    assert receipt["whole_120_shard_or_1point2M_execution_claim"] is False
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    assert ledger["delivery_inventory"]["OPENTHOUGHTS3"]["checked"] is True
    assert ledger["delivery_inventory"]["OPENTHOUGHTS3"]["delivery_receipt"] == path


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


def test_wildchat_original_positions_dates_and_budget_termination_are_honest():
    path = "docs/workload-data/validation/wildchat-original-continuation-20261026.json"
    receipt = json.loads((ROOT / path).read_text("utf-8"))
    native, consumer = receipt["native_profile"], receipt["fixed_consumer"]
    assert receipt["source_checkbox_eligible"] is True
    assert receipt["asset"]["original_rows_bound_and_indexed"] == 59857
    assert native["selected_original_rows"] == [0, 1, 2]
    assert native["all_original_assistant_positions_requested"] == 4
    assert native["request_original_prefix_lengths"] == [1, 1, 3, 1]
    assert native["request_original_row_indices"] == [0, 1, 1, 2]
    assert native["finish_reasons"] == {"length": 1, "stop": 3}
    assert native["budget_exhausted_response_retained_without_retry"] is True
    assert native["source_datetime_types_preserved_in_reader"] is True
    assert native["target_and_future_messages_excluded_from_requests"] is True
    assert not native["task_edits_or_input_truncation"]
    assert consumer["public_research_roundtrip_indices"] == [0, 1, 2]
    assert consumer["exact_installed_payload_files"] == 41
    assert (
        consumer[
            "original_messages_all_turn_metadata_moderation_outcomes_and_ids_preserved_in_sample"
        ]
        is True
    )
    assert consumer["research_top_level_reader_is_public_provider"] is True
    assert not consumer["research_default_lock_changed"]
    for key in (
        "quality_scored",
        "online_agent_episode_claim",
        "original_api_capture_claim",
        "full_corpus_model_execution_claim",
        "raw_payload_or_prediction_published",
    ):
        assert receipt[key] is False
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["WILDCHAT"]
    assert row["checked"] is True and row["status"] == "AVAILABLE"
    assert row["delivery_receipt"] == path
