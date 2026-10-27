"""Frozen real-run receipt contracts; these tests never re-execute MBPP."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = "docs/workload-data/validation/mbpp-original-native-sample-20261026.json"


def test_native_original_sample_and_failure_are_retained():
    receipt = json.loads((ROOT / RECEIPT).read_text("utf-8"))
    assert receipt["source_checkbox_eligible"] is True
    generation, scoring = receipt["generation"], receipt["scoring"]
    assert generation["selected_task_ids"] == [11, 12, 13]
    assert generation["demonstration_task_ids"] == [2, 3, 4]
    assert generation["new_native_model_calls"] == 3
    assert scoring["original_reference_checks_passed"] is True
    assert scoring["passed"] == [True, True, False]
    assert scoring["original_failure_retained"] is True
    assert scoring["isolation_checks_passed_before_import_and_execution"] is True
    assert scoring["model_calls_during_scoring"] == 0
    assert not receipt["raw_data_or_completions_published"]


def test_real_exact_artifact_consumer_not_only_reading_completes_source():
    receipt = json.loads((ROOT / RECEIPT).read_text("utf-8"))
    consumer = receipt["fixed_consumer"]
    assert consumer["full_original_rows_and_fields_verified"] == 974
    assert consumer["prepared_rows"] == 974
    assert consumer["public_and_research_roundtrips"] == 14
    assert consumer["exact_installed_payload_files"] == 41
    assert consumer["research_top_level_reader_is_public_provider"] is True
    assert consumer["reference_code_setup_and_challenge_separated_from_input"] is True
    assert consumer["research_default_lock_changed"] is False
    assert consumer["observations_or_features_executed"] is False
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["MBPP"]
    assert row["status"] == "AVAILABLE" and row["checked"] is True
    assert row["delivery_receipt"] == RECEIPT
