"""Verify two original task outcomes and their distinct coverage boundaries."""

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    spec = importlib.util.spec_from_file_location(
        "new_catalog", HERE / "derive_catalog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.build()
    assert result == read(HERE / "catalog.json")
    prior = read(ROOT / module.PRIOR)
    assert result["entries"][:24] == prior["entries"]
    assert (
        result["attempted_unique_tasks"],
        result["recorded_official_resolved"],
        result["unattempted_tasks"],
    ) == (26, 3, 705)
    for name in ("openlibrary", "navidrome"):
        summary = read(HERE / f"summary-{name}.json")
        review = read(HERE / f"review-{name}.json")
        whitelist = read(HERE / f"storage-whitelist-{name}.json")
        custody = read(HERE / f"custody-{name}.json")
        assert sha(HERE / f"review-{name}.json") == whitelist["files"]["summary.json"]
        assert sha(HERE / f"summary-{name}.json") == custody["summary_sha256"]
        assert (
            sha(HERE / f"derive_private_{name}.py.source")
            == summary["evidence_sha256"]["derivation_source"]
        )
        grade = summary["grading"]
        application = (
            review["application"]
            if name == "openlibrary"
            else review["patch_application"]
        )
        apply_exit = (
            application["apply_exit_code"]
            if name == "openlibrary"
            else application["exit_code"]
        )
        assert grade["apply_exit_code"] == apply_exit == 0
        assert grade["patch_application_attested"] is True
        assert grade["after_graded_file_states_equal"] is True
        assert grade["state_capture"] == application["state_capture"] == "complete"
        assert summary["agent"]["patch_sha256"] == application["raw_patch_sha256"]
        assert summary["agent"]["patch_bytes"] == application["raw_patch_bytes"]
        assert (
            grade["official_result"] == review["official_grading"]["resolved"] is False
        )
        assert (
            summary["agent"]["model_calls"]
            == review["request_audit"]["successful_returns"]
        )
        assert (
            review["request_audit"]["attempts"]
            == review["request_audit"]["successful_returns"]
        )
        assert (
            review["request_audit"]["rejected"]
            == review["request_audit"]["http_retries"]
            == 0
        )
        for kind in ("fail_to_pass", "pass_to_pass"):
            left = grade["official_test_coverage"][kind]
            right = review["official_grading"][kind]
            assert all(left[k] == right[k] for k in ("expected", "passed", "missing"))
        if name == "openlibrary":
            evidence = review["evidence"]
            assert sha(HERE / f"summary-{name}.json") == evidence["CPU_summary_sha256"]
            assert custody["manifest_sha256"] == evidence["CPU_seal_sha256"]
            assert custody["files_verified"] == evidence["CPU_files"] == 67
            assert (
                summary["evidence_sha256"]["service_release"]
                == evidence["service_receipt_sha256"]
            )
            assert all(
                review["resource_release"][k]
                for k in ("sessions_drained", "all_devices_released")
            )
        else:
            assert sha(HERE / f"summary-{name}.json") == review["CPU_summary_sha256"]
            assert custody["manifest_sha256"] == review["CPU_seal_sha256"]
            assert custody["files_verified"] == review["CPU_files_verified"] == 65
            assert (
                review["owned_controllers_exited"]
                and review["owned_Agent_container_absent"]
            )
    openlibrary = read(HERE / "review-openlibrary.json")
    raw = openlibrary["raw_test_observation"]
    assert (
        raw["target_tests_executed"] is True
        and raw["not_compile_or_collection_failure"] is True
    )
    assert raw["raw_final_test_counts"] == {"passed": 0, "failed": 3}
    assert raw["original_image_pytest_reruns"]["observed_total_reruns"] == 9
    assert (
        raw["original_image_pytest_reruns"]["additional_Agent_or_scoring_launches"] == 0
    )
    navidrome = result["entries"][-1]
    assert navidrome["raw_test_coverage"]["target_test_build_failed"] is True
    assert navidrome["raw_test_coverage"]["target_tests_executed"] is False
    assert navidrome["raw_test_coverage"]["target_test_assertions_failed"] is False
    assert navidrome["parsed_test_counts"] == {}
    assert navidrome["official_test_coverage"]["fail_to_pass"]["missing"] == 2
    current = read(HERE / "current-aggregate.json")
    assert current["attempted_tasks"] == current["recovery_denominator"] == 26
    assert current["original_official_positives"] == 3
    assert current["unattempted_tasks"] == 705
    recovery = read(ROOT / current["recovery_source"])
    assert current["prior_recovery_snapshot"] == recovery["aggregate_distinction"]
    assert recovery["derived_dependency_recovery_result"]["official"] is True
    delta = (
        current["prior_recovery_snapshot"]["positives_after_disclosed_recovery"]
        - current["prior_recovery_snapshot"]["original_official_positives"]
    )
    assert current["positives_after_disclosed_recovery"] == 3 + delta == 4
    assert current["additional_Agent_attempts_for_recovery"] == 0
    return {
        "status": "passed",
        "attempts": 26,
        "original_positive": 3,
        "same_answer_recovery_positive": 4,
        "unattempted": 705,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
