"""Keep the complete KV-tiering numeric evidence and boundary checks in CPU CI."""

import subprocess
import sys
from pathlib import Path


def test_kv_tiering_complete_public_evidence():
    report = (
        Path(__file__).resolve().parents[1]
        / "reports/tech-outcomes-20261117/kv-tiering"
    )
    subprocess.run(
        [sys.executable, str(report / "test_public_evidence.py")],
        cwd=report,
        check=True,
        capture_output=True,
        text=True,
    )


def test_kv_tiering_swe_pressure_subset_evidence():
    report = (
        Path(__file__).resolve().parents[1]
        / "reports/tech-outcomes-20261117/kv-tiering/diagnostics/swe-pressure-8turn-20261118"
    )
    subprocess.run(
        [sys.executable, str(report / "test_evidence.py")],
        cwd=report,
        check=True,
        capture_output=True,
        text=True,
    )
