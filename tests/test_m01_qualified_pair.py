"""Keep published 3.4.1 results tied to all source certificates and critical paths."""

from pathlib import Path
import subprocess
import sys


def test_published_pair_recomputes_from_evidence():
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/recompute_m01_qualified_pair.py"),
            "--check",
        ],
        cwd=root,
        check=True,
    )
