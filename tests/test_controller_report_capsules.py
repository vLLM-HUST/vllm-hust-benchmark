"""Frozen bytes and independent verifier failures must remain CI gates."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location(
    "controller_report_capsules",
    Path(__file__).resolve().parents[1]
    / "scripts/verify_controller_report_capsules.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def capsule(tmp_path):
    measured = tmp_path / "sessions.csv"
    measured.write_bytes(b"program,round\r\np,0\r\n")
    (tmp_path / "immutable-files.json").write_text(
        json.dumps(
            {
                "files": {
                    "sessions.csv": hashlib.sha256(measured.read_bytes()).hexdigest()
                }
            }
        )
    )
    (tmp_path / "README.md").write_text("public evidence\n")
    checksum = hashlib.sha256((tmp_path / "README.md").read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{checksum}  README.md\n")
    (tmp_path / "verify.py").write_text("print('verified')\n")
    return tmp_path


def test_valid_capsule_preserves_original_csv_line_endings(capsule):
    before = (capsule / "sessions.csv").read_bytes()
    result = MODULE.verify(capsule)
    assert result["status"] == "PASS" and result["public_verifiers"] == 1
    assert (capsule / "sessions.csv").read_bytes() == before


def test_line_ending_normalization_of_measured_input_is_rejected(capsule):
    (capsule / "sessions.csv").write_bytes(b"program,round\np,0\n")
    with pytest.raises(ValueError, match="Frozen evidence changed"):
        MODULE.verify(capsule)


def test_changed_public_document_requires_updated_checksum(capsule):
    (capsule / "README.md").write_text("changed\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        MODULE.verify(capsule)


def test_checksum_cannot_escape_its_capsule(capsule):
    (capsule / "SHA256SUMS").write_text(f"{'a' * 64}  ../outside\n")
    with pytest.raises(ValueError, match="Unsafe evidence path"):
        MODULE.verify(capsule)


def test_failing_independent_verifier_is_not_reported_as_pass(capsule):
    (capsule / "verify.py").write_text("raise SystemExit(3)\n")
    with pytest.raises(subprocess.CalledProcessError):
        MODULE.verify(capsule)
