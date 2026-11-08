"""Receipt scope/count consistency only, not a substitute for native execution."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_typst_failures_are_not_reported_as_positive_judge_qualification():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/structeval-typst-full50-audit-20261031.json"
        ).read_text("utf-8")
    )
    assert receipt["source_id"] == "STRUCTEVAL"
    assert (
        receipt["prior_valid_generations_reused"]
        + receipt["new_original_generation_calls"]
        == 50
    )
    assert receipt["new_generation_finish_reasons"] == {"stop": 47}
    assert receipt["all50_render_score_histogram"] == {"0": 50}
    assert receipt["all50_final_score_histogram"] == {"0.1": 48, "0.09": 1, "0.07": 1}
    assert receipt["actual_native_vqa_calls"] == 0
    assert receipt["original_missing_image_skip_rule_preserved"] is True
    assert receipt["positive_native_vqa_path_qualified"] is False
    assert receipt["new_model_calls_in_audit_or_diagnosis"] == 0
    assert receipt["source_checkbox_eligible"] is False
    assert receipt["whole_source_delivery"] is False
    pins = json.loads(
        (ROOT / "configs/workload-native/longmemeval-v2-reader-pins.json").read_text(
            "utf-8"
        )
    )
    assert receipt["model_revision"] == pins["model"]["revision"]
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    current = ledger["delivery_inventory"]["STRUCTEVAL"]
    assert current["verified_unique_native_tasks"] >= 1394
    delivery = json.loads((ROOT / current["delivery_receipt"]).read_text("utf-8"))
    assert (
        current["checked"]
        and current["delivery_kind"] == "original_nonrender_protocols_official_scoring"
    )
    assert delivery["native_evidence"]["complete_original_protocols"] == 19
    assert not delivery["remaining_modes"][
        "all44_protocols_or_all2035_native_tasks_complete"
    ]
    assert all(
        "typst" not in path.lower()
        for path in delivery["native_evidence"]["historical_receipts_reused"]
    )


def test_latex_closed_groups_are_counted_once_without_promoting_source():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]["STRUCTEVAL"]
    progress = json.loads(
        (ROOT / ledger["latex_full_protocol_progress_receipt"]).read_text("utf-8")
    )
    completed = progress["completed_protocols"]
    assert len({p["profile"] for p in completed}) == len(completed)
    assert progress["new_original_tasks_audited"] == sum(
        p["new_original_tasks"] for p in completed
    )
    for profile in completed:
        assert profile["previous_original_tasks_reused"] == 3
        assert profile["previous3_already_in_previous_counter"] is True
        assert (
            profile["all_original_protocol_tasks"] == profile["new_original_tasks"] + 3
        )
        assert profile["new_generation_calls"] == profile["new_original_tasks"]
        assert profile["new_judge_calls"] == profile["new_successfully_rendered_images"]
        assert profile["new_model_calls_in_independent_audit"] == 0
    assert progress["current_audited_unique_task_count"] == (
        progress["previous_audited_unique_task_count"]
        + progress["new_original_tasks_audited"]
    )
    assert (
        progress["current_audited_unique_task_count"]
        == ledger["verified_unique_native_tasks"]
    )
    assert (
        progress["remaining_original_tasks"]
        == 2035 - ledger["verified_unique_native_tasks"]
    )
    assert progress["source_checkbox_eligible"] is False
    delivery = json.loads((ROOT / ledger["delivery_receipt"]).read_text("utf-8"))
    assert ledger["checked"] and delivery["source_checkbox_eligible"]
    assert delivery["native_evidence"]["unique_original_tasks"] == 950
    assert not delivery["remaining_modes"][
        "all44_protocols_or_all2035_native_tasks_complete"
    ]
