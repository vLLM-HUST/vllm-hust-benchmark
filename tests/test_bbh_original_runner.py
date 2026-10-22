"""CPU-only invented prompt fixtures; never model or task validation evidence."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_bbh_boolean_original.py"
spec = importlib.util.spec_from_file_location("bbh_original_runner_cpu", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_help_without_optional_evaluation_dependencies():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--data-root" in result.stdout


@pytest.mark.parametrize(
    "value",
    [
        "https://remote.example/v1",
        "http://127.0.0.1/v1?token=x",
        "http://user:pass@localhost/v1",  # pragma: allowlist secret - invented rejection-test credentials
        "http://localhost/not-v1",
        "http://localhost/v1#fragment",
    ],
)
def test_remote_or_credentialed_endpoints_rejected(value):
    with pytest.raises(ValueError):
        runner.checked_endpoint(value)


def test_publisher_prefix_unchanged_and_evaluation_target_never_read():
    prefix = "fixture task description\n\n" + "\n\n".join(
        f"Q: demo-{i}\nA: fixture demonstration." for i in range(3)
    )
    actual, demos = runner.publisher_prefix(
        ("fixture header\n-----\n" + prefix).encode()
    )
    assert actual == prefix
    assert demos == ["demo-0", "demo-1", "demo-2"]
    prompt = runner.task_prompt(
        actual, {"input": "  original\ninput ", "target": "REFERENCE_SENTINEL"}
    )
    assert (
        prompt == prefix + "\n\nQ:   original\ninput \nA: Let's think step by step.\n"
    )
    assert "REFERENCE_SENTINEL" not in prompt


@pytest.mark.parametrize(
    "raw", [b"no separator", b"a\n-----\nb\n-----\nc", b"a\n-----\nno demonstrations"]
)
def test_bad_publisher_boundary_rejected(raw):
    with pytest.raises(ValueError):
        runner.publisher_prefix(raw)


def test_bad_asset_identity_rejected(tmp_path):
    source = tmp_path / "fixture"
    source.write_bytes(b"not original")
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.checked_asset(source, "0" * 64)
