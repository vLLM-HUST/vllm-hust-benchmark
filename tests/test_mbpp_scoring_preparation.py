"""Inert generated-text binding fixtures; no scoring or host code execution."""

import importlib.util
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "mbpp_prepare", ROOT / "scripts/prepare_mbpp_scoring_inputs.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
generate = importlib.import_module("run_mbpp_original_generation").generate


def task(i):
    return {
        "task_id": i,
        "text": f"fixture task {i}",
        "code": f"fixture solution {i}",
        "test_list": [f"fixture test {j}" for j in range(3)],
        "test_setup_code": "fixture original setup",
        "challenge_test_list": [],
    }


@pytest.fixture
def frozen_generation(tmp_path):
    rows = [task(i) for i in range(11, 511)]
    demos = [task(i) for i in (2, 3, 4)]
    source = {"revision": "fixture-only-not-publisher-data"}
    texts = iter(["", "invalid syntax fixture", "partial budget output"])

    def transport(url, body):
        if url.endswith("/tokenize"):
            return {"count": 10}
        text = next(texts)
        return {
            "choices": [
                {
                    "text": text,
                    "finish_reason": "length" if text.startswith("partial") else "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1 if text else 0},
        }

    directory = tmp_path / "generation"
    generate(
        list(enumerate(rows[:3])),
        demos,
        output=directory,
        endpoint="http://127.0.0.1:19466/v1",
        model="fixture-model",
        max_tokens=32,
        context=128,
        transport=transport,
        source_identity=source,
    )
    return directory, rows, demos, source


def prepare(frozen_generation, tmp_path, **kwargs):
    directory, rows, demos, source = frozen_generation
    return RUNNER.prepare(
        directory,
        rows=rows,
        examples=demos,
        source_identity=source,
        output=tmp_path / "prepared",
        **kwargs,
    )


def test_empty_invalid_and_budget_outputs_are_all_retained(frozen_generation, tmp_path):
    result = prepare(frozen_generation, tmp_path)
    tasks = json.loads((tmp_path / "prepared/tasks-private.json").read_text("utf-8"))
    assert [item["task_id"] for item in tasks] == [11, 12, 13]
    assert [item["completion"] for item in tasks] == [
        "",
        "invalid syntax fixture",
        "partial budget output",
    ]
    assert all(
        item["original_task"]["test_setup_code"] == "fixture original setup"
        for item in tasks
    )
    assert not result["functional_tests_executed"]
    assert not result["source_checkbox_eligible"]
    assert result["new_model_calls"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "prompt",
        "target-reference",
        "finish",
        "path",
        "selected",
        "source",
        "not-terminal",
        "usage",
        "context",
    ],
)
def test_changed_or_incomplete_generation_never_prepares(
    frozen_generation, tmp_path, mutation
):
    directory, _, _, _ = frozen_generation
    receipt_path = directory / "receipt-private.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    sample_path = directory / "task-11.private.json"
    saved = json.loads(sample_path.read_text("utf-8"))
    if mutation == "prompt":
        saved["request"]["prompt"] += "modified"
    elif mutation == "target-reference":
        saved["original_task"]["code"] = "replaced reference"
    elif mutation == "finish":
        saved["response"]["choices"][0]["finish_reason"] = "content_filter"
    elif mutation == "path":
        receipt["samples"][0]["private_file"] = "../escape.json"
    elif mutation == "selected":
        receipt["selected_original_rows"] = [1, 2, 3]
    elif mutation == "source":
        receipt["source_identity"] = {"revision": "changed"}
    elif mutation == "usage":
        saved["response"]["usage"]["prompt_tokens"] = 11
        receipt["samples"][0]["usage"]["prompt_tokens"] = 11
    elif mutation == "context":
        receipt["context_capacity"] = 1
    else:
        receipt["status"] = "STARTED"
    sample_path.write_text(json.dumps(saved), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        prepare(frozen_generation, tmp_path)
    assert not (tmp_path / "prepared").exists()


def test_final_asset_guard_runs_before_prepared_status(frozen_generation, tmp_path):
    def changed():
        raise ValueError("fixture asset changed")

    with pytest.raises(ValueError, match="asset changed"):
        prepare(frozen_generation, tmp_path, final_verify=changed)
    assert not (tmp_path / "prepared").exists()
