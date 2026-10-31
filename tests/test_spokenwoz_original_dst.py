"""Invented protocol/transport fixtures, never original datasets or model calls."""

import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
prepare = importlib.import_module("prepare_spokenwoz_original_dst")
runner = importlib.import_module("run_spokenwoz_original_dst")


def prompt():
    return {
        "basename": "fixture-0",
        "content": "fixture original prompt",
        "answer": {"slot": "wanted"},
    }


def test_payload_does_not_invent_sampling_budget_or_include_reference():
    payload = runner.request_body(prompt(), "fixture", False)
    assert set(payload) == {"model", "messages", "chat_template_kwargs"}
    assert payload["messages"] == [{"role": "user", "content": prompt()["content"]}]
    assert "wanted" not in json.dumps(payload)
    assert "chat_template_kwargs" not in runner.request_body(prompt(), "fixture")


def test_wrong_answer_is_valid_zero_not_filtered_failure():
    result = runner.original_state_score([{**prompt(), "result": "{}"}])
    assert result == {"correct_turns": 0, "total_turns": 1, "parse_failures": []}


def test_invalid_json_cannot_reuse_previous_prediction():
    result = runner.original_state_score(
        [
            {**prompt(), "result": '{"slot":"wanted"}'},
            {**prompt(), "basename": "fixture-2", "result": "{invalid}"},
        ]
    )
    assert result["correct_turns"] is None
    assert result["parse_failures"] == ["fixture-2"]


def test_saved_changed_payload_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "selected_originals", lambda _: {})
    monkeypatch.setattr(runner, "expected_prompts", lambda *_: [prompt()])
    bad = runner.request_body(prompt(), "fixture")
    bad["max_tokens"] = 1
    (tmp_path / "000-request-private.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="binding differs"):
        runner.verify_saved(tmp_path, tmp_path, tmp_path, "fixture")


def execution_fixture(tmp_path, monkeypatch, prompts=None):
    author = tmp_path / "author"
    author.mkdir()
    (author / "eval_JGA.py").write_text("print('JGA: 0.0')\n")
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(runner, "verify_author", lambda _: author)
    monkeypatch.setattr(runner, "selected_originals", lambda _: {})
    monkeypatch.setattr(runner, "expected_prompts", lambda *_: prompts or [prompt()])

    def prepare_inputs(data_root, author_root, output):
        folder = output / "author"
        folder.mkdir(parents=True)
        (folder / "prompt_test_output_cleangpt.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in (prompts or [prompt()]))
        )

    monkeypatch.setattr(runner, "prepare_inputs", prepare_inputs)
    return data, author


def test_fixture_execution_saved_verification_and_zero_score(tmp_path, monkeypatch):
    data, author = execution_fixture(tmp_path, monkeypatch)
    calls = []

    def sender(url, body, **kwargs):
        calls.append((url, body))
        return {
            "model": "fixture",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        }

    output = tmp_path / "output"
    report = runner.execute(
        data, author, output, "http://127.0.0.1:19468/v1", "fixture", sender=sender
    )
    assert len(calls) == 1 and report["independent_grade_agrees"] is True
    assert report["official_reported_jga"] == 0
    assert (
        report["actual_model_calls"] == 1
        and report["source_checkbox_eligible"] is False
    )
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    runner.verify_saved(data, author, output, "fixture")
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_all_transport_failures_retained_without_retry_or_replacement(
    tmp_path, monkeypatch
):
    prompts = [prompt(), {**prompt(), "basename": "fixture-2"}]
    data, author = execution_fixture(tmp_path, monkeypatch, prompts)
    calls = []

    def sender(*args, **kwargs):
        calls.append(args)
        raise OSError("fixture unavailable")

    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="no filtered score"):
        runner.execute(
            data, author, output, "http://127.0.0.1:19468/v1", "fixture", sender=sender
        )
    assert len(calls) == 2 and len(list(output.glob("*-failure-private.json"))) == 2
    assert not (output / "score-private.log").exists()


def test_source_identity_required_before_selection(tmp_path):
    (tmp_path / "data.json").write_text("{}")
    with pytest.raises(ValueError, match="complete pinned original"):
        prepare.selected_originals(tmp_path)


def test_changed_prepared_reference_rejected_before_any_model_call(
    tmp_path, monkeypatch
):
    data, author = execution_fixture(tmp_path, monkeypatch)
    prepared = tmp_path / "prepared/author"
    prepared.mkdir(parents=True)
    (prepared / "prompt_test_output_cleangpt.jsonl").write_text(
        json.dumps({**prompt(), "answer": {"slot": "changed"}}) + "\n"
    )
    calls = []
    with pytest.raises(ValueError, match="before model calls"):
        runner.execute(
            data,
            author,
            tmp_path / "output",
            "http://127.0.0.1:19468/v1",
            "fixture",
            prepared_root=prepared.parent,
            sender=lambda *a, **k: calls.append(a),
        )
    assert calls == []
