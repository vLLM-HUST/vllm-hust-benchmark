"""Keep failed partial-artifact scoring, original history and recovery counts separate."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
b = importlib.import_module("build_midterm_v096")
project_review = importlib.import_module("build_swe_tutanota_docs").project_review


def read(path):
    return (ROOT / path).read_bytes()


def test_current_projection_and_preserved_history():
    value = b.build("WORKTREE", read)
    old = b.previous.build(
        "WORKTREE", lambda p: read(b.OLD_REPORT if p == b.REPORT else p)
    )
    assert value["version"] == "V0.96"
    current = value["swe_incremental_tasks"]
    assert current["attempted_unique_tasks"] == 18
    assert current["recorded_official_resolved"] == 2
    assert current["unattempted_tasks"] == 713
    assert current["entries"][:-1] == old["swe_incremental_tasks"]["entries"]
    a = value["swe_same_answer_recovery_aggregate"]
    assert a["positives_after_disclosed_recovery"] == 3
    assert (
        a["recovery_denominator"] == 18
        and a["additional_Agent_attempts_for_recovery"] == 0
    )
    assert a["prior_recovery_snapshot"]["attempted_tasks"] == 17
    last = current["entries"][-1]
    assert not last["agent_submitted"] and last["status"] == "LimitsExceeded"
    assert last["patch_applied"] and last["apply_exit_code"] == 0
    assert last["raw_test_coverage"]["target_tests_executed"] is False
    assert last["raw_test_coverage"]["target_test_assertions_failed"] is False
    assert last["parsed_test_counts"] == {"ERROR": 1}
    for ident in ("swe-single-agent-observation", "swe-test-source-recovery"):
        section = next(s for s in value["sections"] if s["id"] == ident)
        for key in ("scope", "scope_en"):
            assert "3/18" in section[key] and "2/18" in section[key]
            assert "437.454803" in section[key]
    section = next(
        s for s in value["sections"] if s["id"] == "swe-single-agent-observation"
    )
    assert (
        len(section["tables"][0]["rows"]) == len(section["tables_en"][0]["rows"]) == 18
    )
    for section in old["sections"]:
        if section["id"] not in {
            "swe-single-agent-observation",
            "swe-test-source-recovery",
            "v15-remaining",
        }:
            assert section == next(
                s for s in value["sections"] if s["id"] == section["id"]
            )
    for key in (
        "paired_own_compute",
        "protonmail_dependency_recovery",
        "terminal_cython_single_trace",
    ):
        assert value[key] == old[key]


def test_public_offline_verifier():
    path = ROOT / b.FOLDER / "verify.py"
    spec = importlib.util.spec_from_file_location("tutanota_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.verify()["attempted"] == 18


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("agent", "agent_submitted"), True),
        (("official_grading", "resolved"), True),
        (("official_grading", "test_counts"), {"PASSED": 1}),
        (("official_grading", "fail_to_pass", "missing"), 0),
        (("request_audit", "successful_returns"), 99),
        (("request_audit", "rejected"), 1),
        (("new_review_model_or_grading_calls",), 1),
    ],
)
def test_reject_rewritten_failed_outcome(path, replacement):
    value = json.loads(read(b.FOLDER + "/storage-review.json"))
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(AssertionError):
        project_review(value)


def test_reject_recovery_as_additional_attempt():
    catalog = json.loads(read(b.SOURCE))
    recovery = copy.deepcopy(json.loads(read(b.previous.RECOVERY)))
    recovery["aggregate_distinction"]["attempted_tasks"] = 18
    with pytest.raises(AssertionError):
        b.aggregate(catalog, recovery)
