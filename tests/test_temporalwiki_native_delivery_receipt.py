"""Scope/custody assertions; not a substitute for actual native model execution."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = "docs/workload-data/validation/temporalwiki-base-probes-native-delivery-20261101.json"


def test_full_original_probe_native_delivery_and_current_consumer():
    receipt = json.loads((ROOT / RECEIPT).read_text("utf-8"))
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]["TEMPORALWIKI"]
    assert ledger["checked"] and ledger["status"] == "AVAILABLE"
    assert ledger["delivery_receipt"] == RECEIPT and receipt["source_checkbox_eligible"]
    protocol = receipt["native_protocol"]
    assert protocol["original_csv_files"] == len(protocol["scores"]) == 8
    assert (
        sum(x["rows"] for x in protocol["scores"]) == protocol["original_rows"] == 35948
    )
    assert (
        sum(x["batches"] for x in protocol["scores"])
        == protocol["native_npu_completed_forwards"]
        == 1126
    )
    assert protocol["original_input_length"] == protocol["original_output_length"] == 25
    assert (
        protocol["original_batch_size"] == 32
        and protocol["original_precision"] == "float32"
    )
    assert protocol["independent_npu_saved_scalar_rescore_exact"]
    assert (
        protocol["audit_new_model_calls"] == 0
        and protocol["prior_failed_real_forward_attempts"] == 1
    )
    handoff = receipt["current_ci_and_research_handoff"]
    assert handoff["full_ci_conclusion"] == handoff["cpu_smoke_conclusion"] == "success"
    assert handoff["installed_git_and_wheel_payloads_verified"] == 78
    assert (
        handoff["complete_original_fields_roundtripped"]
        == handoff["original_index_rows"]
        == 35948
    )
    assert handoff["fixed_complete_public_research_readbacks"] == 24
    assert (
        not handoff["research_modified_or_observation_started"]
        and handoff["new_model_calls"] == 0
    )
    raw = (ROOT / receipt["native_entry"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == receipt["native_entry_sha256"]


def test_full_base_probe_is_not_all_temporal_modes_or_quality_claim():
    receipt = json.loads((ROOT / RECEIPT).read_text("utf-8"))
    assert not receipt["paper_checkpoint_quality_or_engine_performance_claim"]
    assert not receipt["raw_data_or_model_in_git_or_windows_disk"]
    assert receipt["compatibility"][
        "base_probe_not_continually_pretrained_paper_checkpoint"
    ]
    assert receipt["remaining_optional_modes"]
    assert all(
        math.isfinite(x["perplexity"]) and x["perplexity"] > 0
        for x in receipt["native_protocol"]["scores"]
    )
    resource = receipt["resources"]
    assert (
        resource["protected9b_and14b_services_unchanged"]
        and resource["exclusive_lease_released"]
    )
    assert (
        resource["maximum_observed_physical4_hbm_increase_mib"]
        < resource["total_increase_guard_mib"]
    )
