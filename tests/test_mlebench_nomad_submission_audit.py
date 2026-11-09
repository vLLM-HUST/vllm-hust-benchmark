"""Synthetic row identity controls; no model predictions or labels."""

import hashlib
import json
from pathlib import Path
import runpy

import pytest


MODULE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "scripts/audit_mlebench_nomad_submission.py"
    )
)
AUDIT = MODULE["audit"]


@pytest.mark.parametrize(
    "rows, error",
    [
        ("a,1,2\nb,3,4\n", None),
        ("b,1,2\na,3,4\n", "public test order"),
        ("a,1,2\na,3,4\n", "public test order"),
        ("a,NaN,2\nb,3,4\n", "finite numbers"),
    ],
)
def test_alignment_guard(tmp_path, rows, error):
    test = tmp_path / "test.csv"
    public = b"id,feature\na,1\nb,2\n"
    test.write_bytes(public)
    submission = tmp_path / "submission.csv"
    submission.write_text("id,formation_energy_ev_natom,bandgap_energy_ev\n" + rows)
    args = {"expected_md5": hashlib.md5(public).hexdigest(), "count": 2}
    if error is None:
        result = AUDIT(test, submission, **args)
        assert result["submission_rows"] == 2
        assert not result["grading_executed"]
    else:
        with pytest.raises(ValueError, match=error):
            AUDIT(test, submission, **args)


def terminal_inputs():
    start = {"original_steps": 500, "original_time_seconds": 86400}
    terminal = {
        "status": "NATIVE_AGENT_TERMINAL_AND_SAVED_SUBMISSION_GRADE_AUDIT",
        "agent_exit_code": 0,
        "original_launcher_exit_code": 0,
        "agent_started_at": "2026-01-01T01:00:00.000000001Z",
        "agent_finished_at": "2026-01-01T03:00:00.000000001Z",
        "agent_oom": False,
        "real_model_requests": 3,
        "response_status_counts": {"200": 2},
        "journal_present": True,
        "best_solution_present": True,
        "submission_sha256": "fixture",
        "pregrade_exit_code": 0,
        "grade_exit_code": 0,
        "valid_submission": True,
    }
    submission = {"submission_sha256": "fixture", "source_checkbox_eligible": False}
    grade = {
        "competition_id": MODULE["COMPETITION"],
        "valid_submission": True,
        "submission_exists": True,
        "is_lower_better": True,
        "submission_path": "/submission.csv",
        "score": 0.4,
    }
    return start, terminal, submission, grade


@pytest.mark.parametrize("prefix", ["", "[2026-11-08 07:25:27,551] [cli.py:203] "])
def test_frozen_terminal_grade_audit_keeps_manual_gate(prefix):
    start, terminal, submission, grade = terminal_inputs()
    report = MODULE["audit_terminal"](
        start, terminal, submission, "grader\n" + prefix + json.dumps(grade, indent=2)
    )
    assert report["official_score"] == 0.4
    assert report["agent_wall_seconds"] == 7200
    assert report["manual_delivery_audit_required"]
    assert not report["source_checkbox_eligible"]


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"agent_exit_code": 143, "original_launcher_exit_code": None}, "interrupted"),
        ({"agent_exit_code": 124, "original_launcher_exit_code": 124}, "timing"),
        ({"submission_sha256": "different"}, "identity"),
        ({"journal_present": False}, "artifacts"),
    ],
)
def test_terminal_audit_does_not_promote_partial_or_detached_evidence(changes, error):
    start, terminal, submission, grade = terminal_inputs()
    terminal.update(changes)
    with pytest.raises(ValueError, match=error):
        MODULE["audit_terminal"](start, terminal, submission, json.dumps(grade))


def test_terminal_budget_exit_allowed_after_original_full_wall_time():
    start, terminal, submission, grade = terminal_inputs()
    terminal.update(
        agent_exit_code=124,
        original_launcher_exit_code=124,
        agent_finished_at="2026-01-02T01:00:00.001Z",
    )
    report = MODULE["audit_terminal"](start, terminal, submission, json.dumps(grade))
    assert report["original_launcher_exit_code"] == 124
