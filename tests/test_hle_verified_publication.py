"""Keep the published HLE outcomes tied to their original measurement evidence."""

import hashlib
from pathlib import Path
import runpy


def test_hle_public_evidence_and_file_checksums():
    root = Path(__file__).resolve().parents[1] / "reports/hle-verified-qwen35-20261015"
    runpy.run_path(str(root / "verify_public.py"))["verify"](root)
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
