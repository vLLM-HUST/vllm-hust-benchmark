"""CPU consistency checks, not new tasks, scores or workload acceptance."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_task_variants_receipt_binds_runner_and_preserves_delivery_gap():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/bbh-all-task-variants-original-npu-20261024.json"
        ).read_text("utf-8")
    )
    assert receipt["generation_calls"] == receipt["unique_original_tasks"] == 78
    assert receipt["task_files"] == len(receipt["tasks"]) == 26
    assert len({task["task_id"] for task in receipt["tasks"]}) == 26
    assert all(task["task_id"] != "boolean_expressions" for task in receipt["tasks"])
    for task in receipt["tasks"]:
        assert [sample["original_index"] for sample in task["samples"]] == [0, 1, 2]
        assert all(sample["finish_reason"] == "stop" for sample in task["samples"])
    assert receipt["normal_stop_samples"] == 78
    assert receipt["automatic_retries"] == 0
    assert receipt["reused_boolean_tasks"] == 3
    assert not receipt["task_rewriting"]
    assert not receipt["prompt_truncation"]
    assert not receipt["full_suite_validated"]
    assert not receipt["source_checkbox_eligible"]
    assert not receipt["raw_data_published"]
    script = ROOT / receipt["runner"]
    assert (
        hashlib.sha256(script.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == receipt["formatted_public_runner_sha256"]
    )
    assert receipt["formatting_only_ast_equivalent"] is True
    assert receipt["runner_sha256"] != receipt["formatted_public_runner_sha256"]


def test_bbh_receipt_preserves_zero_score_as_model_outcome_not_pipeline_failure():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/bbh-boolean-original-npu-20261023.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["source_checkbox_eligible"] is False
    assert receipt["unique_model_tasks"] == len(receipt["samples"]) == 3
    assert receipt["total_generation_calls_including_repeat"] == 6
    for sample in receipt["samples"]:
        assert sample["prompt_tokens"] == sample["preflight_tokens"]
        assert sample["finish_reason"] == "stop"
        assert sample["stop_reason"] == "\n\n"
        assert sample["answer_extracted"] is False
        assert sample["harness_exact_match"] == 0
    assert receipt["installed_reader"]["original_rows_checked"] == 250
    assert receipt["installed_reader"]["model_calls"] == 0
    assert receipt["complete_task_answers_validated"] is False
    assert receipt["full_suite_validated"] is False
    script = ROOT / receipt["public_runner"]
    assert (
        hashlib.sha256(script.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == receipt["public_runner_sha256"]
    )
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text(
            encoding="utf-8"
        )
    )
    row = next(row for row in ledger["sources"] if row["source_id"] == "BBH")
    assert row["checked"] is False
    assert row["selected_samples"] == receipt["selected_original_indices"]
    assert row["completed_samples"] == receipt["selected_original_indices"]
    assert row["server_execution"] == "COMPLETED_EXPECTED_STOP"
    assert "VALID_MODEL_OUTCOME" in row["scoring"]
    assert "不是pipeline故障" in row["blocker"]
