"""Custody/scope regression checks, not a replacement for real native audits."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_delivery_requires_native_audio_and_current_software_handoff():
    path = "docs/workload-data/validation/ljspeech-native-fixed3-20261101.json"
    receipt = json.loads((ROOT / path).read_text("utf-8"))
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]["LJSPEECH"]
    assert ledger["checked"] and ledger["status"] == "AVAILABLE"
    assert ledger["delivery_receipt"] == path
    assert receipt["source_checkbox_eligible"]
    assert receipt["native_npu_model_calls"] == 3
    assert receipt["selected_ids"] == ["LJ001-0001", "LJ001-0002", "LJ001-0003"]
    assert receipt["audit_neural_calls"] == 0
    assert not receipt["output_truncated"]
    wave = receipt["independent_wave_validation"]
    assert wave["samples"] == [218368, 43264, 209920]
    assert wave["input_and_reference_audio_relations"]
    assert wave["riff_sizes_and_chunks"] and wave["finite_nonempty_nonsilent"]
    handoff = receipt["current_ci_and_research_handoff"]
    assert handoff["cpu_smoke_conclusion"] == handoff["full_ci_conclusion"] == "success"
    assert handoff["installed_payloads_byte_identical_to_wheel_and_git"] == 78
    assert handoff["original_index_rows_reused"] == 13100
    assert handoff["public_research_complete_record_roundtrips"] == 3
    assert (
        handoff["index_unchanged"]
        and handoff["native_input_and_reference_relations_verified"]
    )
    assert not handoff["research_code_modified_or_observation_started"]
    assert handoff["new_model_calls"] == 0
    raw = (ROOT / receipt["native_entry"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == receipt["native_entry_sha256"]


def test_fixed3_usability_is_not_quality_or_all_rows_claim():
    receipt = json.loads(
        (
            ROOT / "docs/workload-data/validation/ljspeech-native-fixed3-20261101.json"
        ).read_text("utf-8")
    )
    assert not receipt["all13100_synthesis_claim"]
    assert not receipt["mos_or_semantic_quality_claim"]
    assert not receipt["engine_performance_claim"]
    assert not receipt["raw_data_or_audio_in_git_or_windows"]
    assert not receipt["original_reference_audio_used_as_model_input"]
    assert receipt["protected_services_unchanged"] and receipt["lease_released"]
