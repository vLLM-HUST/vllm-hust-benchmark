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
    assert current["checked"] is False
