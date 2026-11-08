"""Offline entry guards; publisher preparation is verified on the server."""

import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/prepare_mlebench_nomad_author_data.py"
)


def test_nomad_plan_pins_original_competition_and_does_not_import_runtime():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True
    )
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLAN_ONLY"
    assert plan["competition_id"] == "nomad2018-predict-transparent-conductors"
    assert plan["archive_bytes"] == 6957213
    assert len(plan["archive_sha256"]) == 64


def test_mismatched_archive_rejected_without_creating_outputs(tmp_path):
    script = runpy.run_path(str(SCRIPT))
    archive = tmp_path / "wrong.zip"
    archive.write_bytes(b"not the fixed publisher asset")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="fixed identity"):
        script["prepare"](archive, output)
    assert list(output.iterdir()) == []


def test_readback_checks_geometry_links_and_target_isolation(tmp_path, monkeypatch):
    verify = runpy.run_path(str(SCRIPT))["verify_public_structure"]
    monkeypatch.setitem(verify.__globals__, "PREPARED_COUNTS", {"train": 1, "test": 2})
    inputs = {
        "description.md": "Original description\n",
        "train.csv": "id,target\n10,1\n",
        "test.csv": "id,feature\n20,0\n21,1\n",
        "sample_submission.csv": "id,target\n20,0\n21,0\n",
        "train/10/geometry.xyz": "train geometry\n",
        "test/20/geometry.xyz": "test geometry\n",
        "test/21/geometry.xyz": "test geometry\n",
    }
    for name, content in inputs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    result = verify(tmp_path)
    assert result["geometry_references_resolved"] == 3
    assert result["test_and_sample_submission_order_matched"]
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,target\n21,0\n20,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sample submission"):
        verify(tmp_path)
    sample.write_text(inputs["sample_submission.csv"], encoding="utf-8")
    (tmp_path / "test.csv").write_text("id,target\n20,0\n21,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target columns"):
        verify(tmp_path)
    (tmp_path / "test.csv").write_text(inputs["test.csv"], encoding="utf-8")
    (tmp_path / "test/21/geometry.xyz").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty original geometry"):
        verify(tmp_path)
