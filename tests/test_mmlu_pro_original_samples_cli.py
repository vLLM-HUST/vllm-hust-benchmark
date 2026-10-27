"""CLI contract for the pinned MMLU-Pro original-task runner."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runner_exposes_original_subject_configs_without_loading_eval_stack():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_mmlu_pro_original_samples.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--subject" in result.stdout
    assert "chemistry" in result.stdout
    assert "biology" in result.stdout
