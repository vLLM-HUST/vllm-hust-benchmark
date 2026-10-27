"""Recompute the incremental catalog and compare independent public evidence."""

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
    for item in read(HERE / "public-files.json"):
        relative = Path(item["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = ROOT / relative
        assert path.stat().st_size == item["bytes"]
        assert sha(path) == item["sha256"]
    spec = importlib.util.spec_from_file_location(
        "wave2_catalog", HERE / "derive_catalog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = module.build()
    assert catalog == read(HERE / "catalog.json")
    prior = read(ROOT / module.PRIOR)
    assert catalog["entries"][: len(prior["entries"])] == prior["entries"]
    current = read(HERE / "current-aggregate.json")
    assert current["attempted_tasks"] == catalog["attempted_unique_tasks"] == 21
    assert (
        current["original_official_positives"]
        == catalog["recorded_official_resolved"]
        == 3
    )
    assert current["unattempted_tasks"] == catalog["unattempted_tasks"] == 710
    recovery = read(ROOT / current["recovery_source"])
    assert recovery["derived_dependency_recovery_result"]["official"] is True
    assert current["prior_recovery_snapshot"] == recovery["aggregate_distinction"]
    assert current["positives_after_disclosed_recovery"] == 4
    assert current["recovery_denominator"] == 21
    assert current["additional_Agent_attempts_for_recovery"] == 0
    for label, resolved, calls in (
        ("nodebb", False, 54),
        ("ansible", True, 36),
        ("element", False, 96),
    ):
        summary = read(HERE / f"summary-{label}.json")
        review = read(HERE / f"review-{label}.json")
        whitelist = read(HERE / f"storage-whitelist-{label}.json")
        assert sha(HERE / f"review-{label}.json") == whitelist["files"]["summary.json"]
        assert (
            sha(HERE / f"summary-{label}.json")
            == review["evidence"]["public_CPU_summary_sha256"]
        )
        assert (
            sha(HERE / f"derive_private_{label}.py.source")
            == summary["evidence_sha256"]["derivation_source"]
        )
        assert summary["grading"]["official_result"] is resolved
        assert review["official_grading"]["resolved"] is resolved
        assert (
            summary["agent"]["model_calls"]
            == review["request_audit"]["attempts"]
            == calls
        )
        assert review["request_audit"]["successful_returns"] == calls
        assert (
            review["request_audit"]["rejected"]
            == review["request_audit"]["http_retries"]
            == 0
        )
        assert (
            summary["agent"]["patch_sha256"]
            == review["application"]["raw_patch_sha256"]
        )
        assert (
            summary["grading"]["apply_exit_code"]
            == review["application"]["apply_exit_code"]
            == 0
        )
        assert summary["grading"]["after_graded_file_states_equal"] is True
        assert review["application"]["after_equals_graded"] is True
        for name in ("fail_to_pass", "pass_to_pass"):
            a = summary["grading"]["official_test_coverage"][name]
            b = review["official_grading"][name]
            assert all(a[key] == b[key] for key in ("expected", "passed", "missing"))
        assert (
            summary["evidence_sha256"]["service_release"]
            == review["evidence"]["service_receipt_sha256"]
        )
        assert review["new_review_model_or_grading_calls"] == 0
        assert all(
            review["resource_release"][key] is True
            for key in ("sessions_drained", "all_devices_released")
        )
    element = read(HERE / "review-element.json")
    raw = element["raw_test_observation"]
    assert raw["tests"] == 3 and raw["passed"] == 1 and raw["failed"] == 2
    assert raw["snapshot_failed"] == 2 and raw["test_execution_observed"] is True
    assert raw["parser_entries"] == 0
    assert element["official_grading"]["fail_to_pass"]["missing"] == 3
    return {
        "status": "passed",
        "attempts": 21,
        "original_positives": 3,
        "disclosed_recovery_positives": 4,
        "unattempted": 710,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
