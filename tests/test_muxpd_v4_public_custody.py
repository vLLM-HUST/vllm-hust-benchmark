"""The public receipt must retain all byte-bound metadata and complete members."""

import importlib.util
import shutil
from pathlib import Path

import pytest

DATA = (
    Path(__file__).resolve().parents[1]
    / "reports/muxpd-followup-20261025/performance-lifecycle-v4"
)
SPEC = importlib.util.spec_from_file_location(
    "muxpd_public_v4", DATA / "verify_public_v4.py"
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_public_all_assets_members_and_old39_boundary():
    result = VERIFY.verify(DATA)
    assert (result["assets"], result["capsules"], result["original_members"]) == (
        18,
        13,
        1181,
    )
    assert result["public_inventory"] == 57
    assert not result["old39_bytes_redownloaded"]


def test_member_proof_bytes_cannot_silently_change(tmp_path):
    root = tmp_path / "candidate"
    shutil.copytree(DATA, root)
    proof = next((root / "public-readback-v4/attempt-01/archives").glob("*.json"))
    proof.write_bytes(proof.read_bytes() + b" ")
    with pytest.raises(ValueError, match="readback bytes differ"):
        VERIFY.verify(root)


def test_missing_http_receipt_cannot_pass(tmp_path):
    root = tmp_path / "candidate"
    shutil.copytree(DATA, root)
    next((root / "public-readback-v4/attempt-01/receipts").glob("*.json")).unlink()
    with pytest.raises(FileNotFoundError):
        VERIFY.verify(root)
