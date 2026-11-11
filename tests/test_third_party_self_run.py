"""CPU admission checks for the test-specific third-party reproduction tools."""

from pathlib import Path
import subprocess
import sys


def test_version_timing_and_orphan_cleanup():
    kit = Path(__file__).resolve().parents[1] / "scripts" / "third_party_self_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "test_version_gate",
            "test_native_timing",
            "test_process_cleanup",
        ],
        cwd=kit,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
