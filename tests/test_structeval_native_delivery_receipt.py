"""Declared-protocol evidence consistency, not a replacement for actual native runs."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_nonrender_delivery_has_original_native_custody_and_current_consumer_proof():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    current = ledger["delivery_inventory"]["STRUCTEVAL"]
    receipt = json.loads((ROOT / current["delivery_receipt"]).read_text("utf-8"))
    assert current["checked"] and receipt["source_checkbox_eligible"]
    native = receipt["native_evidence"]
    assert (
        native["complete_original_protocols"]
        == len(set(native["original_protocols"]))
        == 19
    )
    assert native["unique_original_tasks"] == native["tasks_per_protocol"] * 19 == 950
    assert native["output_profile"] == "publisher-null" and native["max_tokens"] is None
    assert native["official_saved_output_rescore_identical"]
    assert native["current_saved_run_directories_rechecked"] == 37
    assert native["current_saved_receipt_inference_evaluation_files_rehashed"] == 111
    assert (
        native["missing_or_changed_saved_files"]
        == native["new_model_calls_for_delivery_review"]
        == 0
    )
    assert all((ROOT / path).is_file() for path in native["historical_receipts_reused"])
    client = receipt["original_asset_and_client_qualification"]
    assert (
        client["public_verify_cli_exit_code"]
        == client["clean_client_pip_check_exit_code"]
        == 0
    )
    assert client["installed_official_source_modules_verified"] == 27
    assert (
        client["existing_originals_reused_by_hardlink"]
        and not client["fresh_client_installation_claim"]
    )
    handoff = receipt["fixed_successful_ci_and_current_research_handoff"]
    assert handoff["both_runs_completed_successfully"] and handoff["artifact_id"]
    assert handoff["git_wheel_installed_payloads_equal"] == 78
    assert handoff["all_original_rows_and_index_rows_verified"] == 2035
    assert handoff["original_fields_per_row"] == 10
    assert (
        handoff["public_unchanged_research_complete_record_roundtrips"]
        == handoff["original_protocol_groups"] * 3
        == 132
    )
    assert handoff["original_input_reference_annotation_separation_preserved"]
    assert (
        not handoff["research_code_modified_or_observation_started"]
        and handoff["new_model_calls"] == 0
    )
    remaining = receipt["remaining_modes"]
    assert not remaining["all44_protocols_or_all2035_native_tasks_complete"]
    assert not remaining["capped_9b_profile_results_pooled_with_null"]
    assert not receipt[
        "raw_questions_references_generations_or_enterprise_data_published"
    ]
