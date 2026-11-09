"""Public report verifiers must fail closed under altered execution or views."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPORT = (
    Path(__file__).resolve().parents[1] / "reports/pyramidkv-metadata-net-cost-20261028"
)


@pytest.mark.parametrize("series", ["", "batched", "memory-pressure"])
@pytest.mark.parametrize("mode", ["flag", "environment"])
def test_report_verifier_rejects_optimized_python(series, mode):
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    command = [sys.executable]
    if mode == "flag":
        command.append("-O")
    else:
        env["PYTHONOPTIMIZE"] = "1"
    command.append(str(REPORT / series / "verify.py"))
    result = subprocess.run(
        command, check=False, env=env, capture_output=True, text=True
    )
    assert result.returncode != 0, result.stdout
    assert "assertions enabled" in result.stderr
    assert '"verified": true' not in result.stdout


@pytest.mark.parametrize("series", ["", "batched"])
def test_report_verifier_rejects_changed_outer_analysis(tmp_path, series):
    root = tmp_path / "report"
    shutil.copytree(REPORT, root)
    path = root / series / "analysis.json"
    analysis = json.loads(path.read_text())
    analysis["comparisons"]["24576-c4"]["paired_percent"] = [999, 999]
    path.write_text(json.dumps(analysis))
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    result = subprocess.run(
        [sys.executable, str(root / series / "verify.py")],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, result.stdout
    assert "AssertionError" in result.stderr
    assert '"verified": true' not in result.stdout
