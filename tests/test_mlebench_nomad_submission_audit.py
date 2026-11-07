"""Synthetic row identity controls; no model predictions or labels."""

import hashlib
from pathlib import Path
import runpy

import pytest


AUDIT = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "scripts/audit_mlebench_nomad_submission.py"
    )
)["audit"]


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
