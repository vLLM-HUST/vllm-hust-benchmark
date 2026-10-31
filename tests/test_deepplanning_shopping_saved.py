"""Saved output schema fixtures, not original task or scoring evidence."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "deepplanning_saved", ROOT / "scripts/verify_deepplanning_shopping_saved.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def history():
    return {
        "messages": [
            {"role": "system", "content": "original system fixture"},
            {"role": "user", "content": "original task fixture"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-fixture",
                        "function": {"name": "fixture_tool", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-fixture",
                "content": "fixture result",
            },
            {"role": "assistant", "content": "first phase fixture"},
            {"role": "user", "content": "second phase fixture"},
            {"role": "assistant", "content": "final fixture"},
        ]
    }


def check(payload):
    return MODULE.verify_history(
        payload,
        "original task fixture",
        "original system fixture",
        "second phase fixture",
    )


def test_original_two_phases_and_tool_pairing():
    counts = check(history())
    assert counts == {"model_messages": 3, "tool_calls": 1, "tool_results": 1}


@pytest.mark.parametrize("position", [0, 1, 5])
def test_replaced_prompt_or_missing_original_phase_rejected(position):
    payload = history()
    payload["messages"][position]["content"] = "altered fixture"
    with pytest.raises(ValueError):
        check(payload)


def test_missing_or_wrong_tool_response_rejected():
    payload = history()
    payload["messages"][3]["tool_call_id"] = "other-fixture"
    with pytest.raises(ValueError, match="relation"):
        check(payload)
    payload = history()
    del payload["messages"][3]
    with pytest.raises(ValueError, match="missing"):
        check(payload)


def test_native_budget_end_after_tool_result_not_upgraded_to_completion():
    payload = history()
    payload["messages"][-1] = deepcopy(payload["messages"][2])
    payload["messages"].append(deepcopy(payload["messages"][3]))
    assert check(payload)["tool_results"] == 2
    assert (
        MODULE.score_projection(
            {"case_name": "fixture", "case_score": 0.0, "is_completed": False}
        )["is_completed"]
        is False
    )


def test_live_run_not_audited_as_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "prepare", lambda *args, **kwargs: {})
    (tmp_path / "native-public.json").write_text(
        json.dumps({"status": "RUNNING_ORIGINAL_SHOPPING_TASKS"})
    )
    with pytest.raises(ValueError, match="not terminal"):
        MODULE.audit(tmp_path / "assets", tmp_path)
