"""CPU protocol fixtures only; no dataset, network, generation, or code execution."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "bcb_runner",
    Path(__file__).parents[1] / "scripts/run_bigcodebench_original_generation.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize(
    "value",
    [
        "https://127.0.0.1",
        "http://external.example",
        "http://localhost/v1",
        "http://localhost/?key=fixture",
    ],
)
def test_nonlocal_or_credential_origins_rejected(value):
    with pytest.raises(ValueError):
        runner.origin(value)


def test_original_formatter_receives_unmodified_prompt_and_fixed_defaults():
    called = {}

    def formatter(**kwargs):
        called.update(kwargs)
        return "fixture-rendered-prompt"

    prompt = "\n  original-fixture\n"
    assert (
        runner.format_prompt(SimpleNamespace(make_raw_chat_prompt=formatter), prompt)
        == "fixture-rendered-prompt"
    )
    assert called["task_prompt"] == prompt
    assert called["tokenizer"] is None
    assert called["split"] == "instruct" and called["subset"] == "full"
    assert called["instruction_prefix"] == runner.INSTRUCTION_PREFIX
    assert called["response_prefix"] == runner.RESPONSE_PREFIX


def response():
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": "invented-code-fixture"}}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.mark.parametrize(
    "finish,status",
    [
        ("stop", "GENERATION_TERMINATED"),
        ("length", "TRUNCATED_GENERATION"),
        ("tool_calls", "INCOMPLETE_GENERATION"),
    ],
)
def test_termination_is_not_code_correctness(finish, status):
    value = response()
    value["choices"][0]["finish_reason"] = finish
    before = deepcopy(value)
    result = runner.response_summary(value, 10)
    assert result["status"] == status
    assert result["generated_code_executed"] is False
    assert "pass_at_1" not in result and "solution" not in result
    assert value == before


@pytest.mark.parametrize("failure", ["empty", "count", "total", "budget", "multiple"])
def test_invalid_or_partial_generation_rejected(failure):
    value = response()
    if failure == "empty":
        value["choices"][0]["message"]["content"] = ""
    elif failure == "count":
        value["usage"]["prompt_tokens"] = 9
    elif failure == "total":
        value["usage"]["total_tokens"] = 16
    elif failure == "budget":
        value["usage"]["completion_tokens"] = 1281
    else:
        value["choices"].append(deepcopy(value["choices"][0]))
    with pytest.raises(ValueError):
        runner.response_summary(value, 10)


def test_publisher_identity_checked_before_import(tmp_path):
    path = tmp_path / "not_original.py"
    path.write_text('raise RuntimeError("must not import")', encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.load_original("never_import", path, "fixture-not-matching")


def test_generation_receipt_does_not_claim_functional_score():
    root = Path(__file__).parents[1]
    receipt = json.loads(
        (
            root
            / "docs/workload-data/validation/bigcodebench-original-generation-npu-20261023.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["unique_tasks"] == 3
    assert receipt["selected_samples"] == [r["task_id"] for r in receipt["samples"]]
    assert all(
        r["finish_reason"] == "stop" and r["full_prompt_token_count_matches"]
        for r in receipt["samples"]
    )
    # Frozen historical generation, not a claim that later generic selectors ran.
    assert (
        receipt["public_runner_sha256"]
        == "4d5a6d4850395a17c0c1831e8dbee0f363172f397b804058138fb4e886bad519"  # pragma: allowlist secret - frozen public runner checksum
    )
    assert receipt["generated_code_executed"] is False
    assert receipt["pass_at_1_claim"] is False and receipt["scorer_runs"] == 0
    assert receipt["source_checkbox_eligible"] is False


def test_original_selection_keeps_positions_and_full_suite_available():
    assert runner.select_indices(1140) == [0, 1, 2]
    assert runner.select_indices(1140, indices=[5, 8, 11]) == [5, 8, 11]
    assert runner.select_indices(1140, start=3, count=3) == [3, 4, 5]
    assert runner.select_indices(1140, full=True) == list(range(1140))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"indices": [2, 1, 0]},
        {"indices": [0, 0, 1]},
        {"indices": [0, 1]},
        {"indices": [False, 1, 2]},
        {"indices": [0, 1, 1140]},
        {"full": True, "count": 3},
        {"indices": [0, 1, 2], "start": 0},
        {"start": -1},
        {"start": 1139},
        {"count": 2},
    ],
)
def test_invalid_or_conflicting_selections_rejected(kwargs):
    with pytest.raises(ValueError):
        runner.select_indices(1140, **kwargs)
