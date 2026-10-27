"""Recompute follow-up observations without rerunning Agents or official grading."""

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
        "followup_catalog", HERE / "derive_catalog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = module.build()
    assert catalog == read(HERE / "catalog.json")
    prior = read(ROOT / module.PRIOR)
    assert catalog["entries"][: len(prior["entries"])] == prior["entries"]
    inputs = read(HERE / "inputs.json")
    for observation in inputs["observations"]:
        label = observation["summary"].removeprefix("summary-").removesuffix(".json")
        summary = read(HERE / observation["summary"])
        review = read(HERE / observation["review"])
        whitelist = read(HERE / f"storage-whitelist-{label}.json")
        assert sha(HERE / observation["review"]) == whitelist["files"]["summary.json"]
        assert (
            sha(HERE / observation["summary"])
            == review["evidence"]["CPU_summary_sha256"]
        )
        assert (
            sha(HERE / observation["derivation_source"])
            == summary["evidence_sha256"]["derivation_source"]
        )
        agent, grade = summary["agent"], summary["grading"]
        application = review["application"]
        assert grade["official_result"] == review["official_grading"]["resolved"]
        assert agent["model_calls"] == review["request_audit"]["successful_returns"]
        assert (
            review["request_audit"]["attempts"]
            == agent["model_calls"] + review["request_audit"]["rejected"]
        )
        assert review["request_audit"]["http_retries"] == 0
        assert agent["patch_sha256"] == application["raw_patch_sha256"]
        assert agent["patch_bytes"] == application["raw_patch_bytes"]
        assert grade["apply_exit_code"] == application["apply_exit_code"]
        assert grade["state_capture"] == application["state_capture"]
        assert (
            grade["after_graded_file_states_equal"]
            == application["after_equals_graded"]
        )
        for name in ("fail_to_pass", "pass_to_pass"):
            a = grade["official_test_coverage"][name]
            b = review["official_grading"][name]
            assert all(a[key] == b[key] for key in ("expected", "passed", "missing"))
        assert (
            summary["evidence_sha256"]["service_release"]
            == review["evidence"]["service_receipt_sha256"]
        )
        review_calls = (
            review["new_model_or_grader_by_review"]
            if label == "teleport"
            else review["new_review_model_or_grading_calls"]
        )
        assert review_calls == 0
        assert all(
            review["resource_release"][key]
            for key in ("sessions_drained", "all_devices_released")
        )
    flipt = read(HERE / "review-flipt.json")
    assert flipt["raw_test_observation"]["target_tests_executed"] is False
    assert flipt["raw_test_observation"]["parser_entries"] == 0
    assert flipt["official_grading"]["fail_to_pass"]["missing"] == 4
    vuls = read(HERE / "review-vuls.json")
    assert vuls["request_audit"]["attempts"] == 20
    assert vuls["request_audit"]["successful_returns"] == 19
    assert (
        vuls["context_boundary"]["reported_input_tokens_lower_bound"]
        + vuls["context_boundary"]["requested_output_tokens"]
        > 65536
    )
    assert vuls["application"]["patch_application_attested"] is False
    assert vuls["application"]["apply_exit_code"] == 128
    assert vuls["application"]["affected_paths"] is None
    assert vuls["raw_test_observation"]["target_test_execution_observed"] is True
    teleport = read(HERE / "review-teleport.json")
    assert teleport["request_audit"]["attempts"] == 48
    assert teleport["request_audit"]["rejected"] == 0
    assert teleport["application"]["apply_exit_code"] == 0
    assert teleport["raw_test_observation"]["target_tests_executed"] is True
    assert teleport["raw_test_observation"]["not_compile_failure"] is True
    assert teleport["official_grading"]["fail_to_pass"]["failed"] == 1
    assert teleport["official_grading"]["fail_to_pass"]["missing"] == 0
    current = read(HERE / "current-aggregate.json")
    assert current["attempted_tasks"] == catalog["attempted_unique_tasks"]
    assert (
        current["original_official_positives"] == catalog["recorded_official_resolved"]
    )
    assert current["unattempted_tasks"] == catalog["unattempted_tasks"]
    assert current["recovery_denominator"] == current["attempted_tasks"]
    recovery = read(ROOT / current["recovery_source"])
    assert recovery["derived_dependency_recovery_result"]["official"] is True
    assert current["prior_recovery_snapshot"] == recovery["aggregate_distinction"]
    delta = (
        current["prior_recovery_snapshot"]["positives_after_disclosed_recovery"]
        - current["prior_recovery_snapshot"]["original_official_positives"]
    )
    assert (
        current["positives_after_disclosed_recovery"]
        == current["original_official_positives"] + delta
    )
    assert current["additional_Agent_attempts_for_recovery"] == 0
    return {
        "status": "passed",
        "attempts": current["attempted_tasks"],
        "original_positives": current["original_official_positives"],
        "disclosed_recovery_positives": current["positives_after_disclosed_recovery"],
        "unattempted": current["unattempted_tasks"],
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
