"""Bind real native synthesis to separate artifact/consumer evidence, not CI alone."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "docs/workload-data/validation"


def read(name):
    return json.loads((VALIDATION / name).read_text("utf-8"))


def test_original_native_run_and_saved_audit_are_not_reader_only_evidence():
    native = read("dailytalk-original-fixed29-synthesis-audit-20261101.json")
    assert native["native_exit_code"] == 0
    assert native["original_utterances"] == 29
    assert native["acoustic_forward_batches"] == native["vocoder_forward_batches"] == 4
    assert native["wav_files_full_pcm_decoded"] == native["wav_files_nonzero"] == 29
    assert native["original_guo_feature_files_reverified"] == 29
    assert native["official_model_files_rehashed"] == 2
    assert native["original_requests_and_history_ids_preserved"]
    assert native["old_failures_retained"]
    assert native["protected_services_same_birth"]
    assert native["exclusive_lease_released"]
    assert native["audit_model_calls"] == 0
    assert not native["quality_score_claimed"]
    assert not native["human_quality_evaluation_performed"]


def test_actual_ci_installed_provider_and_research_handoff_have_separate_scope():
    handoff = read("dailytalk-actual-ci-research-handoff-20261101.json")
    assert handoff["run_id"] == 41309442204
    assert handoff["artifact_id"] == 13084712746
    assert (
        handoff["provider_commit"]
        == (
            "703359592b50cd5690dd9f91f1b54c701c766649"  # pragma: allowlist secret - public qualified provider commit
        )
    )
    assert (
        handoff["installed_payload_audit"]["status"]
        == "EXACT_INSTALLED_PROVIDER_PAYLOAD_MATCHED"
    )
    assert handoff["installed_payload_audit"]["payloads"] == 79
    assert handoff["original_rows_indexed"] == 23676
    assert handoff["all_original_rows_reconstructed_byte_identically"]
    assert handoff["fixed_original_public_research_full_record_roundtrips"] == 29
    assert handoff["fixed_original_native_input_row_digests_matched"] == 29
    assert handoff["fixed_original_reference_media_files_exist"] == 29
    assert handoff["original_index_reused"] and handoff["index_unchanged"]
    assert not handoff["torch_imported"]
    assert not handoff["research_code_modified_or_observation_started"]
    assert handoff["model_calls"] == 0
    assert not handoff["source_checkbox_eligible"]


def test_compatibility_and_public_wrapper_execution_evidence_are_disclosed():
    controls = read("dailytalk-gru-npu-equations-20261101.json")
    assert controls["numeric_cases"] == 18
    public = read("dailytalk-public-workspace-entry-20261101.json")
    assert public["native_entry_actions"] == ["plan", "verify"]
    assert public["native_entry_exit_codes"] == [0, 0]
    assert not public["public_execute_path_reexecuted"]
    assert not public["source_checkbox_eligible"]


def test_source_delivery_requires_combined_native_general_and_artifact_evidence():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    source = ledger["delivery_inventory"]["DAILYTALK"]
    assert source["checked"] and source["status"] == "AVAILABLE"
    final = json.loads((ROOT / source["delivery_receipt"]).read_text("utf-8"))
    assert final["source_checkbox_eligible"]
    assert final["native_selection"]["dialogue_ids"] == ["23", "30", "59"]
    assert final["native_selection"]["original_utterances"] == 29
    assert final["native_exit_code"] == 0
    assert final["acoustic_forward_batches"] == final["vocoder_forward_batches"] == 4
    assert final["original_input_rows_indexed_and_reconstructed"] == 23676
    assert final["actual_ci_public_research_full_record_roundtrips"] == 29
    assert final["qualified_general_cpu_ci_run"] == 41312871677
    assert final["qualified_general_full_ci_run"] == 41312868734
    assert final["qualified_ci_conclusions"] == "success"
    assert final["original_native_generation_reused_without_rerun"]
    assert not final["public_execute_wrapper_reexecuted"]
    assert not final["quality_or_engine_performance_claim"]
    for key in (
        "native_evidence",
        "public_saved_output_audit",
        "public_workspace_evidence",
        "general_preparation_evidence",
        "research_handoff_evidence",
        "declared_protocol",
        "generation_entry",
        "preparation_entry",
        "runtime_constraints",
    ):
        assert (ROOT / final[key]).is_file()
