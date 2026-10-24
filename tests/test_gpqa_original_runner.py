"""Transport/software fixtures, not GPQA model results."""

from collections import namedtuple
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "gpqa_runner", Path(__file__).resolve().parents[1] / "scripts/run_gpqa_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize(
    "url",
    ["https://example.com/v1", "http://localhost/other", "http://localhost/v1?key=x"],
)
def test_external_or_ambiguous_endpoint_rejected(url):
    with pytest.raises(ValueError):
        runner.endpoint(url)


def test_source_is_checked_before_import_or_execution(tmp_path):
    path = tmp_path / "baselines/utils.py"
    path.parent.mkdir()
    path.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="source identity"):
        runner.original_components(tmp_path)


@pytest.mark.parametrize("full", [False, True])
def test_transport_preserves_budget_and_publisher_full_index_skip(
    tmp_path, monkeypatch, full
):
    example = namedtuple(
        "Example", "question choice1 choice2 choice3 choice4 correct_index"
    )
    examples = [example("question", "a", "b", "c", "d", 1) for _ in range(70)]

    class Parser:
        LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}

        @staticmethod
        def parse_sampled_answer(answer):
            return "B" if answer == "(B)" else None

    namespace = {
        "load_examples": lambda path, seed: examples,
        "zero_shot_prompt": lambda entry: "original fixture prompt",
        "AnswerPredictor": Parser,
    }

    def original_call(prompt, model, call_type):
        return namespace["client"].chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=1000,
            stop=None,
            messages=[
                {"role": "system", "content": "publisher fixture instruction"},
                {"role": "user", "content": prompt},
            ],
        )

    namespace["select_and_call_model"] = original_call
    monkeypatch.setattr(runner, "original_components", lambda source: (namespace, 1000))
    raw = b"fixture source bytes, not real questions"
    (tmp_path / "gpqa_main.csv").write_bytes(raw)
    monkeypatch.setitem(runner.DATA_SHA256, "main", hashlib.sha256(raw).hexdigest())
    monkeypatch.setitem(runner.ROW_COUNTS, "main", 70)
    calls = []

    def post(url, body):
        if url.endswith("/tokenize"):
            return {"count": 10}
        calls.append(body)
        return {
            "model": "fixture-model",
            "choices": [{"message": {"content": "(B)"}}],
            "usage": {"prompt_tokens": 10},
        }

    monkeypatch.setattr(runner, "post", post)
    args = SimpleNamespace(
        api_base="http://localhost:8000/v1",
        source=tmp_path,
        data_root=tmp_path,
        split="main",
        seed=0,
        full=full,
        model="fixture-model",
        context_window=2048,
        output=tmp_path / "result",
    )
    receipt = runner.run(args)
    count = 69 if full else 3
    assert receipt["completed_generations"] == len(calls) == count
    assert all(
        r["model"] == "fixture-model" and r["max_tokens"] == 1000 and r["stop"] is None
        for r in calls
    )
    assert receipt["correct"] == count and receipt["automatic_retries"] == 0
    if full:
        assert receipt["samples"][69] == {
            "original_offset": 69,
            "publisher_skipped": True,
            "correct": False,
        }
        assert receipt["accuracy"] == 69 / 70
    with pytest.raises(FileExistsError):
        runner.run(args)
    assert len(calls) == count
    assert (
        json.loads((args.output / "receipt.json").read_text())[
            "source_checkbox_eligible"
        ]
        is False
    )
