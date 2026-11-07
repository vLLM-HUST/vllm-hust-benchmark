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
