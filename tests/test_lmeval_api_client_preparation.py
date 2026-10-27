"""Software path/CLI fixtures only; never creates a real environment or runs pip."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "api_client_prepare", ROOT / "scripts/prepare_lmeval_api_client.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_only_fresh_separate_client_or_readonly_verification(tmp_path):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    for name in runner.VENDOR_NAMES:
        (vendor / name).mkdir()
    client = tmp_path / "client"
    assert runner.plan(client, vendor) == (client, vendor)
    client.mkdir()
    with pytest.raises(ValueError, match="fresh client"):
        runner.plan(client, vendor)
    assert runner.plan(client, vendor, verify_only=True) == (client, vendor)
    with pytest.raises(ValueError, match="separate"):
        runner.plan(vendor / "nested-client", vendor)


def test_cli_help_does_not_install_or_import_a_model():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_lmeval_api_client.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0 and "--verify-only" in result.stdout
