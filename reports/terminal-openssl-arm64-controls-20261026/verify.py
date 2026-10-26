"""Check public control aggregates against the independent review, offline."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify():
    raw = (HERE / "summary.json").read_bytes()
    data = json.loads(raw)
    review = json.loads((HERE / "independent-review.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == review["summary_sha256"]
    assert data["evidence_manifest_sha256"] == review["evidence_manifest_sha256"]
    assert data["derived_image"] == review["derived_image"]
    assert data["base_manifest"] == review["original_base_manifest"]
    assert data["platform"] == review["platform"] == "linux/arm64"
    assert data["model_calls"] == data["model_task_attempts"] == 0
    assert data["official_89_task_model_coverage"] == 0
    assert not data["full_dataset_acceptance"]
    assert not data["amd64_image_equivalence_claimed"]
    assert data["original_task_tests_solution_instructions_unchanged"]
    assert review["instruction_tests_oracle_unchanged"]
    assert review["both_owned_containers_absent_at_readback"]
    assert data["changed_files"] == review["task_changes_only"]
    plan_raw = (HERE / "controls-plan.json").read_bytes()
    plan = json.loads(plan_raw)
    index = json.loads((HERE / "restricted-evidence-index.json").read_text())
    pinned = next(x for x in index["files"] if x["evidence"] == "controls-plan.json")
    assert hashlib.sha256(plan_raw).hexdigest() == pinned["sha256"]
    assert len(plan_raw) == pinned["bytes"]
    assert plan["image_id"] == data["derived_image"]
    assert plan["source_revision"] == data["harbor_revision"]
    assert plan["model_requests"] == plan["other_tasks"] == 0
    assert plan["attempts_per_control"] == 1 and plan["no_retries"]
    assert len(plan["task_files"]) == review["task_original_files_verified"] == 8
    assert len(plan["source_pins"]) == review["actual_Harbor_source_pins_verified"] == 6
    assert len(data["controls"]) == len(review["controls"]) == 2
    for control, checked in zip(data["controls"], review["controls"], strict=True):
        assert control["role"] == checked["role"]
        assert control["reward"] == checked["reward"]
        assert control["passed"] + control["failed"] + control["skipped"] == 6
        assert control["passed"] == checked["test_counts"].get("passed", 0)
        assert control["failed"] == checked["test_counts"].get("failed", 0)
        assert control["source_result_sha256"] == checked["source_result_sha256"]
        assert control["scorer_output_sha256"] == checked["ctrf_sha256"]
        assert control["process_exit_code"] == checked["process_exit_code"] == 0
        assert control["all_owned_containers_released"] and control["model_calls"] == 0
    assert [(c["role"], c["reward"]) for c in data["controls"]] == [
        ("nop", 0),
        ("oracle", 1),
    ]
    return {"status": "passed", "control_trials": 2, "model_task_attempts": 0}


if __name__ == "__main__":
    print(json.dumps(verify()))
