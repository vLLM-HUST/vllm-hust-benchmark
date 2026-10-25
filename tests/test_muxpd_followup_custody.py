"""Frozen report custody checks only; no model execution or acceptance waiver."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/muxpd-followup-20261025"
SPEC = importlib.util.spec_from_file_location(
    "muxpd_followup_verify", REPORT / "verify_report.py"
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_terminal_custody_and_bounded_claims():
    result = VERIFY.verify(REPORT)
    assert result["passed"]
    assert result["npy_files"] == 1328
    assert not result["complete_group_metadata_coverage"]
    assert not result["quality_or_performance_qualified"]


def test_changed_copy_is_rejected(tmp_path):
    path = tmp_path / "corrupted.json"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Canonical bytes differ"):
        VERIFY.checked(path, "0" * 64)
