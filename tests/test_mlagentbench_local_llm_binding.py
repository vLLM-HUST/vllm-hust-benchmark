"""Binding fixtures, not original Agent execution or real model requests."""

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "mlagent_binding", ROOT / "scripts/mlagentbench_local_llm_binding.py"
)
BINDING = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BINDING)


def test_explicit_actual_model_and_original_override_forwarding(monkeypatch):
    monkeypatch.setenv("MLAGENTBENCH_LOCAL_API_BASE", "http://127.0.0.1:14001/v1")
    monkeypatch.setenv("MLAGENTBENCH_LOCAL_MODEL", "Qwen/Qwen3-14B")
    calls = []

    def complete(prompt, model, **kwargs):
        calls.append((prompt, model, kwargs))
        return "fixture response"

    monkeypatch.setattr(BINDING, "complete_original_chat", complete)
    assert (
        BINDING.complete_text(
            "raw prompt", "edit.log", "Qwen/Qwen3-14B", max_tokens_to_sample=4000
        )
        == "fixture response"
    )
    assert calls[-1][0:2] == ("raw prompt", "Qwen/Qwen3-14B")
    assert calls[-1][2]["max_tokens_to_sample"] == 4000
    monkeypatch.setattr(BINDING, "FAST_MODEL", "Qwen/Qwen3-14B")
    BINDING.complete_text_fast("raw summary", log_file="summary.log")
    assert calls[-1][2]["temperature"] == 0.01
    assert "max_tokens_to_sample" not in calls[-1][2]


def test_missing_explicit_model_configuration_is_not_faked(monkeypatch):
    monkeypatch.delenv("MLAGENTBENCH_LOCAL_API_BASE", raising=False)
    with pytest.raises(KeyError):
        BINDING.complete_text("raw", "step.log", "Qwen/Qwen3-14B")


def test_interactive_claude_helper_is_not_silently_replaced():
    with pytest.raises(RuntimeError, match="not this local-chat"):
        BINDING.complete_text_claude("invent task name")
