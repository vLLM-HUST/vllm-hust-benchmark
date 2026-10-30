"""Preparation fixtures, never native multi-agent execution claims."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_multiagentbench_original_assets as entry  # noqa: E402 - standalone script path


def test_default_plan_is_inert_and_preserves_five_original_scenarios(tmp_path):
    destination = tmp_path / "absent"
    report = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-I",
                str(ROOT / "scripts/prepare_multiagentbench_original_assets.py"),
                "--data-root",
                str(destination),
            ],
            text=True,
        )
    )
    assert report["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert len(report["files"]) == 5
    assert report["total_bytes"] == 5369142
    assert report["model_calls"] == 0 and not report["source_checkbox_eligible"]
    assert not destination.exists()


def test_preparation_requires_explicit_budget_before_acquisition(tmp_path, monkeypatch):
    def forbidden(*_args):
        pytest.fail("budget rejection must precede acquisition")

    monkeypatch.setattr(entry, "acquire", forbidden)
    with pytest.raises(ValueError, match="max-bytes"):
        entry.prepare(tmp_path / "absent", action="prepare", max_bytes=1)


def test_missing_originals_not_replaced_by_synthetic_prompts(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        entry.prepare(tmp_path, action="verify")
    assert not list(tmp_path.iterdir())


def test_repository_is_not_a_data_destination():
    with pytest.raises(ValueError, match="outside"):
        entry.prepare(ROOT / "not-created")
