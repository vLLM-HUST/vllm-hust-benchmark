"""CPU contract checks only; no original tasks, tokenizer requests or model calls."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


PATH = (
    Path(__file__).resolve().parents[1] / "scripts/run_natural_instructions_original.py"
)
SPEC = importlib.util.spec_from_file_location("ni_original_runner", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_default_selection_and_full_small_scope():
    for name, (_, count) in RUNNER.TASKS.items():
        assert list(RUNNER.selected_indices(name, False)) == [0, 1, 2]
        if count > 250:
            with pytest.raises(ValueError):
                RUNNER.selected_indices(name, True)
        else:
            assert len(RUNNER.selected_indices(name, True)) == count


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost/v1",
        "http://example.com/v1",
        "http://localhost/v1?token=x",
        "http://localhost/other",
    ],
)
def test_nonlocal_or_ambiguous_endpoint_rejected(url):
    with pytest.raises(ValueError):
        RUNNER.validate_endpoint(url)


def test_truncation_is_never_a_fallback():
    tokenizer = RUNNER.OriginalModelTokenizer("http://localhost:1/v1", "fixture")
    with pytest.raises(ValueError, match="truncation"):
        tokenizer.decode([1, 2, 3])


def test_actual_tokenizer_ids_are_forwarded(monkeypatch):
    monkeypatch.setattr(RUNNER, "post", lambda url, body: {"tokens": [7, 8, 9]})
    tokenizer = RUNNER.OriginalModelTokenizer("http://localhost:1/v1", "fixture")
    assert tokenizer("fixture input") == {"input_ids": [7, 8, 9]}


def test_help_does_not_require_original_runtime_dependencies():
    result = subprocess.run(
        [sys.executable, "-I", str(PATH), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--full-small" in result.stdout
