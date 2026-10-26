"""Combine preserved prior observations with independently graded new tasks."""

import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = "reports/swe-native018-navidrome-20261027/catalog.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root=ROOT, here=HERE):
    inputs = json.loads((here / "inputs.json").read_text())
    prior_path = root / PRIOR
    assert sha(prior_path) == inputs["prior_catalog_sha256"]
    prior = json.loads(prior_path.read_text())
    entries = copy.deepcopy(prior["entries"])
    for entry in entries:
        assert sha(root / entry["source"]) == entry["source_sha256"]
    for item in inputs["observations"]:
        path = here / item["summary"]
        assert path.parent == here and sha(path) == item["summary_sha256"]
        value = json.loads(path.read_text())
        review_path = here / item["review"]
        assert sha(review_path) == item["review_sha256"]
        review = json.loads(review_path.read_text())
        assert value["vllm"] == value["vllm_ascend"] == "0.18.0"
        assert value["total_dataset_tasks"] == 731
        assert value["new_task_attempts"] == value["recorded_official_denominator"] == 1
        assert value["full_dataset_acceptance"] is False
        assert value["private_task_answers_patch_or_trajectory_included"] is False
        assert value["official_unresolved_grading"] == 0
        agent, grade, release = value["agent"], value["grading"], value["release"]
        assert type(grade["official_result"]) is bool
        assert (
            grade["independently_recomputed_official_rule"] == grade["official_result"]
        )
        assert grade["model_calls"] == 0
        assert grade["patch_modified"] is grade["grading_reexecuted"] is False
        assert grade["patch_application_attested"] is True
        assert type(grade["apply_exit_code"]) is int
        assert grade["patch_applied"] == (grade["apply_exit_code"] == 0)
        assert release["status"] == "completed" and release["service_exit_code"] == 0
        assert all(
            release[key] is True
            for key in ("sessions_drained", "all_devices_released", "seal_allowed")
        )
        assert value["recorded_official_resolved"] == int(grade["official_result"])
        assert agent["patch_sha256"] == value["evidence_sha256"]["submitted_patch"]
        assert 0 < agent["model_calls"] <= value["budget"]["step_limit"] == 100
        assert value["budget"]["wall_budget_seconds"] == 1200
        assert value["budget"]["max_tokens"] == 4096
        assert value["budget"]["automatic_retries"] == 0
        if review.get("test_coverage", {}).get("target_test_build_failed"):
            coverage = review["test_coverage"]
            assert coverage["target_tests_executed"] is False
            assert coverage["target_test_assertions_failed"] is False
            assert coverage["records_with_Test"] == 0
            assert coverage["official_parser_records"] == 0
            assert grade["official_result"] is False
            assert grade["patch_applied"] is True
            assert all(
                part["official_test_patch_uses_method"]
                and not part["submitted_new_file_declares_method"]
                and part["actual_compiler_reports_undefined"]
                for part in coverage["source_mechanism"].values()
            )
        entries.append(
            {
                "run_id": value["run_id"],
                "task_identity_sha256": value["task_identity_sha256"],
                "source": str(path.relative_to(root)),
                "source_sha256": sha(path),
                "official_resolved": int(grade["official_result"]),
                "official_denominator": 1,
                "official_unresolved_grading": 0,
                "model_calls": agent["model_calls"],
                "elapsed_seconds": agent["elapsed_seconds"],
                "status": agent["status"],
                "agent_submitted": agent["agent_submitted"],
                "patch_bytes": agent["patch_bytes"],
                "patch_sha256": agent["patch_sha256"],
                "patch_application_evidence_limited": False,
                "confirmed_patch_application_failure": not grade["patch_applied"],
                "apply_exit_code": grade["apply_exit_code"],
                "patch_applied": grade["patch_applied"],
                "state_capture": grade["state_capture"],
                "parsed_test_counts": grade["parsed_test_counts"],
                "test_count_scope": grade["test_count_scope"],
                "official_test_coverage": grade["official_test_coverage"],
                "raw_test_coverage": review.get("test_coverage"),
            }
        )
    assert len(entries) <= 731
    assert len({entry["task_identity_sha256"] for entry in entries}) == len(entries)
    assert len({entry["run_id"] for entry in entries}) == len(entries)
    result = {
        key: copy.deepcopy(value) for key, value in prior.items() if key != "entries"
    }
    result.update(
        schema="swe-native018-incremental-catalog/v2",
        prior_catalog=PRIOR,
        prior_catalog_sha256=sha(prior_path),
        attempted_unique_tasks=len(entries),
        recorded_official_denominator=len(entries),
        recorded_official_resolved=sum(entry["official_resolved"] for entry in entries),
        application_evidence_limited_tasks=sum(
            entry["patch_application_evidence_limited"] for entry in entries
        ),
        confirmed_patch_application_failure_tasks=sum(
            entry["confirmed_patch_application_failure"] for entry in entries
        ),
        unattempted_tasks=731 - len(entries),
        confirmed_target_compilation_failure_tasks=sum(
            bool(entry.get("raw_test_coverage", {}).get("target_test_build_failed"))
            for entry in entries
            if entry.get("raw_test_coverage")
        ),
        full_dataset_acceptance=False,
        entries=entries,
    )
    return result


if __name__ == "__main__":
    output = HERE / "catalog.json"
    output.write_text(json.dumps(build(), indent=2) + "\n")
