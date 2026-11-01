"""Receipt custody/scope checks; these do not replace original NPU execution."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    "docs/workload-data/validation/streamingbench-real-native-delivery-20261101.json"
)


def test_delivery_links_original_native_score_and_current_consumer():
    receipt = json.loads((ROOT / RECEIPT).read_text("utf-8"))
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]["STREAMINGBENCH"]
    assert ledger["checked"] and ledger["status"] == "AVAILABLE"
    assert ledger["delivery_receipt"] == RECEIPT
    assert receipt["source_checkbox_eligible"]
    protocol = receipt["native_protocol"]
    assert protocol["original_group_positions"] == [0, 1, 2]
    assert (
        protocol["complete_original_groups"]
        == protocol["native_npu_generation_calls"]
        == 3
    )
    assert protocol["original_prefix_end_seconds"] == [31, 206, 403]
    assert protocol["original_max_new_tokens"] == 128
    assert protocol["original_scorer_exit_code"] == 0
    assert protocol["original_question_and_reference_fields_unchanged"]
    assert protocol["independent_original_rescore_identical"]
    assert protocol["audit_new_model_calls"] == 0
    handoff = receipt["current_ci_and_research_handoff"]
    assert handoff["full_ci_conclusion"] == handoff["cpu_smoke_conclusion"] == "success"
    assert handoff["installed_git_and_wheel_payloads_verified"] == 78
    assert handoff["original_question_files"] == 8
    assert handoff["original_index_rows"] == 4948
    assert handoff["fixed_complete_public_research_readbacks"] == 24
    assert handoff["original_native_input_groups_matched"] == 3
    assert not handoff["research_modified_or_observation_started"]
    source = (ROOT / receipt["native_entry"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(source).hexdigest() == receipt["native_entry_sha256"]


def test_real_subset_does_not_claim_full_stream_or_audio_coverage():
    receipt = json.loads((ROOT / RECEIPT).read_text("utf-8"))
    assert not receipt["full_real_model_run_claim"]
    assert not receipt["streaming_audio_or_other_modes_completed_claim"]
    assert not receipt["paper_quality_or_engine_performance_claim"]
    assert not receipt["raw_data_or_model_in_git_or_windows_disk"]
    assert any(
        "45" in gap and "audio" in gap for gap in receipt["remaining_optional_modes"]
    )
    restored = receipt["resource_restoration"]
    assert restored["original9b_restored_and_model_identity_ready"]
    assert restored["protected_card6_14b_unchanged"]
    assert restored["exclusive_lease_released"]
