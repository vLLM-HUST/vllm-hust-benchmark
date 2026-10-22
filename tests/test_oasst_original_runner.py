"""Runner contract checks only, no real requests or publisher payloads."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


PATH = (
    Path(__file__).resolve().parents[1] / "scripts/run_oasst_original_continuation.py"
)
SPEC = importlib.util.spec_from_file_location("oasst_original_runner", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/v1",
        "http://example.com/v1",
        "http://user:password@localhost/v1",
        "http://localhost/v1?x=1",
        "http://localhost/v1#fragment",
        "http://localhost/other",
    ],
)
def test_remote_or_credential_endpoints_rejected(url):
    with pytest.raises(ValueError):
        RUNNER.validate_endpoint(url)


def test_local_endpoint_and_explicit_three_original_targets():
    assert RUNNER.validate_endpoint("http://127.0.0.1:19466/v1")
    assert len(set(RUNNER.TARGETS)) == 3


def test_help_needs_no_optional_data_sdk():
    result = subprocess.run(
        [sys.executable, "-I", str(PATH), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--data-file" in result.stdout
