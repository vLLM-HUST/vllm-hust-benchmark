"""CI checks the report's evidence and standalone metric regression suite."""

import subprocess
import sys
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[1] / "reports" / "tech-outcomes-20261117" / "muxpd"
)


def test_muxpd_published_metrics_match_original_evidence():
    result = subprocess.run(
        [sys.executable, str(REPORT / "recompute.py"), "--check"],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_muxpd_metric_validation_and_regressions():
    result = subprocess.run(
        [sys.executable, str(REPORT / "test_recompute.py")],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
