"""CPU interface/safety checks, never native task execution evidence."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_tau2_help_does_not_load_optional_environment():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_tau2_original_telecom.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--official-archive" in result.stdout


def test_tau2_rejects_remote_model_endpoint_before_dependency_or_data_access(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_tau2_original_telecom.py"),
            "--data-root",
            str(tmp_path / "missing-data"),
            "--official-archive",
            str(tmp_path / "missing.zip"),
            "--output",
            str(tmp_path / "output"),
            "--endpoint",
            "https://api.openai.com/v1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "only permits local HTTP endpoints" in result.stderr
    assert not (tmp_path / "output").exists()
