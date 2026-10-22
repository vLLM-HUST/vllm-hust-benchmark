"""CPU interface safety only; no benchmark data or model execution."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_help_without_optional_task_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_structeval_original_json.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--all-json" in result.stdout


def test_external_endpoint_refused_before_inputs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_structeval_original_json.py"),
            "--data-file",
            str(tmp_path / "missing"),
            "--official-archive",
            str(tmp_path / "missing.zip"),
            "--output",
            str(tmp_path / "out"),
            "--endpoint",
            "https://example.com/v1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "only local HTTP NPU" in result.stderr
    assert not (tmp_path / "out").exists()
