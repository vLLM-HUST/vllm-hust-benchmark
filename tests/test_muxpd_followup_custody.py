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
PUBLIC_SPEC = importlib.util.spec_from_file_location(
    "muxpd_public_readback_verify", REPORT / "verify_public_readback.py"
)
PUBLIC_VERIFY = importlib.util.module_from_spec(PUBLIC_SPEC)
PUBLIC_SPEC.loader.exec_module(PUBLIC_VERIFY)


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


def test_anonymous_public_readback_and_all_capsule_proofs():
    result = PUBLIC_VERIFY.verify(REPORT)
    assert result["passed"] and result["anonymous_access"]
    assert result["public_assets"] == 21 and result["capsules"] == 13
    assert result["members_verified"] == 1882
    assert result["first_last_inventory_identical"]
    assert not result["quality_or_performance_qualified"]


def test_public_readback_rejects_changed_receipt(tmp_path):
    import shutil

    shutil.copytree(REPORT / "public-readback-v2", tmp_path / "public-readback-v2")
    (tmp_path / "public-readback-v2/release-api-after.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="Readback receipt bytes changed"):
        PUBLIC_VERIFY.verify(tmp_path)
