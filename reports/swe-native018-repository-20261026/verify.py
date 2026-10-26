"""Verify public result arithmetic and independent review; no private grading."""

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(p):
    return json.loads(p.read_text())


def verify():
    spec = importlib.util.spec_from_file_location("catalog", HERE / "derive_catalog.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.build()
    assert data == read(HERE / "catalog.json")
    assert (
        data["attempted_unique_tasks"] == 10 and data["recorded_official_resolved"] == 1
    )
    assert data["unattempted_tasks"] == 721
    assert (
        data["application_evidence_limited_tasks"] == 1
        and data["confirmed_patch_application_failure_tasks"] == 2
    )
    summary = read(HERE / "summary-element.json")
    review = read(HERE / "independent-review.json")
    assert summary["agent"]["model_calls"] == review["agent"]["model_calls"] == 28
    assert summary["agent"]["patch_sha256"] == review["agent"]["patch_sha256"]
    assert (
        summary["grading"]["official_result"]
        == review["official_grading"]["resolved"]
        is False
    )
    assert review["official_grading"]["unresolved"] == []
    assert (
        review["patch_application"]["exit_code"]
        == summary["grading"]["apply_exit_code"]
        == 0
    )
    assert review["patch_application"]["after_graded_equal"] is True
    coverage = read(HERE / "test-coverage-review.json")
    assert (
        coverage["raw_jest"]["tests_passed"] == 437
        and coverage["raw_jest"]["tests_failed"] == 1
    )
    assert (
        coverage["official_parser"]["raw_records"] == 437
        and coverage["official_parser"]["unique_names"] == 418
    )
    assert coverage["raw_expected_F2P_test_executed_and_failed"] is True
    assert summary["grading"]["official_test_coverage"]["fail_to_pass"] == {
        "expected": 1,
        "passed": 0,
        "not_passed": 1,
        "missing": 1,
    }
    recovery = summary["grading_environment_recovery"]
    assert (
        recovery["first_exit_code"] == 1
        and recovery["first_official_adapter_launched"] is False
    )
    assert (
        recovery["new_agent_calls"] == 0 and recovery["official_scoring_attempts"] == 1
    )
    assert review["new_model_calls_by_review"] == review["new_grading_executions"] == 0
    release = read(HERE / "independent-release.json")
    assert (
        release["status"] == "passed_released_seal_verified"
        and release["sealed_files_verified"] == 14
    )
    assert release["service_exit_code"] == 0 and release["client_model_calls"] == 28
    custody = read(HERE / "private-custody.json")
    assert (
        custody["files_verified"] == 67
        and custody["private_contents_included"] is False
    )
    for item in read(HERE / "public-files.json"):
        path = Path(item["path"])
        assert not path.is_absolute() and ".." not in path.parts
        raw = (ROOT / path).read_bytes()
        assert (
            len(raw) == item["bytes"]
            and hashlib.sha256(raw).hexdigest() == item["sha256"]
        )
    return {
        "status": "passed",
        "attempted": 10,
        "official_resolved": 1,
        "unattempted": 721,
        "model_calls": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
