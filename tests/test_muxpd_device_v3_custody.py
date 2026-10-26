"""Prevent prepared diagnostic evidence from becoming an unsupported acceptance."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/muxpd-followup-20261025/device-evidence-v3"
SPEC = importlib.util.spec_from_file_location(
    "muxpd_device_v3", DATA / "verify_device_v3.py"
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def changed_copy(directory, name, mutate):
    """Rehash a changed derived copy so the semantic guard is actually exercised."""
    path = directory / name
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value, indent=2) + "\n")
    copied = directory / "canonical-copy-map.json"
    manifest = json.loads(copied.read_text())
    for row in manifest["copies"]:
        if row["canonical_path"] == name:
            row["sha256"] = row["source"]["sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            row["bytes"] = row["source"]["bytes"] = path.stat().st_size
    copied.write_text(json.dumps(manifest))


def test_v3_fixed_terminal_scope_and_previous_bytes():
    result = VERIFY.verify(DATA)
    assert result["terminal_runs"] == 9
    assert result["failed_attempts_preserved"] == 2
    assert result["intentional_manager_exit130"] == 2
    assert result["prior_evidence_files_unchanged"] == 116
    assert not result["paired_quality_or_performance_accepted"]


def test_v3_preparation_cannot_claim_remote_download(tmp_path):
    target = tmp_path / "candidate"
    shutil.copytree(DATA, target)
    path = target / "release-assets-v3.json"
    value = json.loads(path.read_text())
    value["assets"][0]["remote_verified"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Historical remote state"):
        VERIFY.verify(target)


def test_v3_rehashed_metadata_cannot_claim_performance(tmp_path):
    target = tmp_path / "candidate"
    shutil.copytree(DATA, target)
    changed_copy(
        target,
        "archive-index-v3.json",
        lambda value: value.update(current_performance_claim=True),
    )
    with pytest.raises(ValueError, match="Unsupported performance/quality"):
        VERIFY.verify(target)


def test_v3_rehashed_summary_cannot_hide_missing_abort(tmp_path):
    target = tmp_path / "candidate"
    shutil.copytree(DATA, target)
    changed_copy(
        target,
        "analysis/device-input-reverification-v3.json",
        lambda value: value["pressure_suite"].update(
            overall_passed=True, abort_issued=True
        ),
    )
    with pytest.raises(ValueError, match="Failed pressure coverage"):
        VERIFY.verify(target)
