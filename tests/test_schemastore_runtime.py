"""Offline preparation contracts; no Node download or execution in CI."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ENTRY = Path(__file__).resolve().parents[1] / "scripts/prepare_schemastore_runtime.py"


def test_plan_creates_nothing(tmp_path):
    target = tmp_path / "absent"
    result = subprocess.run(
        [sys.executable, str(ENTRY), "--output", str(target)],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLAN_ONLY" and plan["model_calls"] == 0
    assert "nodejs.org" in plan["official_checksum_source"]
    assert plan["url"].endswith("linux-arm64.tar.xz")
    assert not target.exists()


def test_untrusted_archive_rejected_before_extraction(tmp_path):
    spec = importlib.util.spec_from_file_location("schema_runtime", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / "fake.tar.xz"
    archive.write_bytes(b"synthetic invalid fixture")
    with pytest.raises(ValueError, match="checksum"):
        module.validate_archive(archive)
