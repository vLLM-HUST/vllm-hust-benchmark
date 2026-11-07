"""Receipt consistency controls; these do not execute original GAIA tasks."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gaia_declared_native_delivery_preserves_failures_and_research_boundary():
    receipt_path = (
        "docs/workload-data/validation/gaia-original-native-delivery-20261103.json"
    )
    receipt = json.loads((ROOT / receipt_path).read_text("utf-8"))
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]["GAIA"]
    assert ledger["checked"] and ledger["status"] == "AVAILABLE"
    assert ledger["delivery_receipt"] == receipt_path
    assert receipt["source_checkbox_eligible"]
    assert receipt["original_rows_verified"] == 466
    assert receipt["original_attachment_relations_verified"] == 109
    original = receipt["profiles"]["original_fixed_first3"]
    retry = receipt["profiles"]["failed_second_task_only"]
    assert (original["start"], original["count"]) == (0, 3)
    assert (original["returned_tasks"], original["failed_tasks"]) == (2, 1)
    assert original["official_correct"] == retry["official_correct"] == 0
    assert original["model_http200_responses"] == 33
    assert retry["model_http200_responses"] == 9
    assert (retry["start"], retry["count"], retry["failed_tasks"]) == (1, 1, 0)
    assert not retry["first_and_third_tasks_rerun"]
    assert not receipt["scores_merged_across_profiles"]
    assert not receipt["original_timeout_or_outputs_overwritten"]
    assert receipt["published_research_and_public_readbacks"] == 7
    assert receipt["research_all_four_ci_jobs_passed"]
    assert receipt["actual_research_dependency_lock_cli_verified"]
    assert not receipt["research_default_lock_changed"]
    assert not receipt["research_pr_merged"]
    assert receipt["new_model_calls_during_final_audit"] == 0
    assert not receipt["raw_data_published"]
    assert any(
        "audio and video execution remain unqualified" in x
        for x in receipt["scope_limits"]
    )
    for key in (
        "native_entry",
        "scoring_entry",
        "tool_factory",
        "python_tool",
        "requirements",
    ):
        assert (ROOT / receipt[key]).is_file()
