"""Offline hashes, original scoring sets and same-attempt recovery arithmetic."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def validate(s, build, receipts, review):
    r = s["derived_dependency_recovery_result"]
    assert r["official"] is True
    assert r["parsed_tests"] == r["passed"] == 3 and r["failed"] == 0
    tests = {v["name"]: v["status"] for v in r["tests"]}
    assert len(tests) == 3 and set(tests.values()) == {"PASSED"}
    for key in ("F2P", "P2P"):
        group = r[key]
        assert group["expected"] == group["passed"] == 1
        assert group["failed"] == group["missing"] == 0
        assert len(group["tests"]) == 1
        for test in group["tests"]:
            assert tests[test["name"]] == test["status"] == "PASSED"
    for key in ("original_result", "test_source_only_recovery_result"):
        assert s[key]["official"] is False and s[key]["parsed_tests"] == 0
        assert s[key]["functional_coverage_available"] is False
    assert (
        s["Agent_attempts_added"]
        == s["model_requests_added"]
        == s["task_attempts_added"]
        == 0
    )
    assert s["verifier_only_recovery_executions"] == 1 and not s["is_new_Agent_run"]
    a = s["aggregate_distinction"]
    assert (
        a["attempted_tasks"]
        == a["original_official_denominator"]
        == a["recovery_denominator"]
        == 17
    )
    assert a["total_tasks"] == 731 and a["unattempted_tasks"] == 731 - 17
    assert a["original_official_positives"] == 2
    assert (
        a["positives_after_disclosed_recovery"]
        == a["original_official_positives"] + int(r["official"])
        == 3
    )
    assert s["original_Agent_patch_reused"] and s["patch_apply_exit"] == 0
    assert s["after_equals_graded"] and s["test_source_unchanged_after_grading"]
    assert s["original_parser_run_script_scoring_rules_and_budgets_unchanged"]
    assert (
        build["canvas"] == "2.11.2"
        and build["Node"] == "18.20.8"
        and build["ABI"] == 108
    )
    assert build["upgrades"] == build["removals"] == build["new_Agent_model_calls"] == 0
    assert build["required_deb_packages_added"] == 91
    assert (
        build["original_source_and_yarn_lock_retained"]
        and build["original_package_versions_retained"]
    )
    assert build["base_image"] != build["derived_image"]
    assert s["grading_process_exit"] == 0 and s["controller_exit"] == 1
    e = receipts["derived_verifier_recovery"]
    assert e["official_record"] is True and e["outer_controller_exit"] == 1
    assert e["grading_process_exit"] == 0 and e["owned_proxy_exit"] == -15
    assert e["release_API_status"] == 404 and e["original_exit_codes_unchanged"]
    for key in (
        "CPU_containers_absent",
        "controller_and_proxy_absent",
        "socket_absent",
        "release_verified_separately",
    ):
        assert e[key]
    assert review["release_independently_verified"] and review["affected_files"] == 8
    assert (
        review["completed_sealed_files_verified"] == 44
        and review["bindings_verified"] == 195
    )
    assert (
        "mail" in s["raw_output_caveat"] and "No tests found" in s["raw_output_caveat"]
    )
    return True


def verify():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    whitelist = json.loads((ROOT / "whitelist.json").read_text())
    for name, pin in whitelist["files"].items():
        raw = (ROOT / name).read_bytes()
        assert (
            len(raw) == pin["bytes"]
            and hashlib.sha256(raw).hexdigest() == pin["sha256"]
        )
    objects = [
        json.loads((ROOT / name).read_text())
        for name in (
            "summary.json",
            "dependency-build.json",
            "receipt-projections.json",
            "review.json",
        )
    ]
    validate(*objects)
    return {
        "status": "passed",
        "scope": "public hashes, unchanged scoring sets and recovery arithmetic; private original execution is not rerun",
        "original_positive": 2,
        "recovered_positive": 3,
        "attempts": 17,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
