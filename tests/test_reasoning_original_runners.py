"""CPU CLI checks; not task execution evidence."""

from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name", ["run_gsm8k_incremental_samples.py", "run_mmlu_pro_original_samples.py"]
)
def test_help_without_task_dependencies(name):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--data-root" in result.stdout


def test_incremental_range_rejected_before_data_or_model_access(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_gsm8k_incremental_samples.py"),
            "--data-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--endpoint",
            "http://127.0.0.1:1/v1/chat/completions",
            "--model",
            "unused",
            "--start",
            "1318",
            "--count",
            "6",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "within the 1319" in result.stderr
    assert not (tmp_path / "out").exists()


def test_arc_runner_help_lists_both_original_suites_without_optional_dependencies():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_arc_original_samples.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ARC-Challenge" in result.stdout
    assert "ARC-Easy" in result.stdout
