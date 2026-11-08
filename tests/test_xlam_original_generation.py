"""Invented prompt/card/API controls only; no original dataset or model execution."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_xlam_original_generation as runner


def card_fixture(tmp_path, body=None):
    body = (
        body
        or """def build_prompt(task_instruction: str, format_instruction: str, tools: list, query: str):
    prompt = f"{task_instruction}|{json.dumps(xlam_format_tools)}|{format_instruction}|{query}"
    return prompt
"""
    )
    text = (
        '```python\ntask_instruction = """invented instruction"""\nformat_instruction = """invented format"""\n'
        + body
        + "\n```\n"
    )
    raw = text.encode()
    path = tmp_path / "author.md"
    path.write_bytes(raw)
    return path, {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def records():
    tools = json.dumps(
        [
            {"name": "one", "parameters": {}, "unknown": True},
            {"name": "two", "parameters": {"x": {"type": "string"}}},
        ]
    )
    return [
        SimpleNamespace(
            source={
                "source_id": "XLAM_TOOL",
                "file": "original.json",
                "file_sha256": "fixture",
            },
            task={
                "task_id": str(i),
                "input": {"query": f"original query {i}", "tools": tools},
            },
            outcome={"answers": "TEACHER_ONLY"},
            replay={"source_row_index": i, "annotations": {"id": i}},
        )
        for i in range(3)
    ]


def test_original_tools_and_query_use_author_builder_without_schema_conversion(
    tmp_path,
):
    card, profile = card_fixture(tmp_path)
    render = runner.author_renderer(card, profile)
    row = records()[0]
    assert (
        render(row.task["input"])
        == "invented instruction|"
        + row.task["input"]["tools"]
        + "|invented format|original query 0"
    )
    assert "TEACHER_ONLY" not in render(row.task["input"])


def test_changed_author_card_rejected_before_executing_builder(tmp_path):
    card, profile = card_fixture(tmp_path)
    card.write_text(card.read_text() + "changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.author_renderer(card, profile)


def test_unreviewed_author_code_is_not_executed(tmp_path):
    card, profile = card_fixture(
        tmp_path,
        """def build_prompt(task_instruction: str, format_instruction: str, tools: list, query: str):
    import os
    return query
""",
    )
    with pytest.raises(ValueError, match="reviewed pure"):
        runner.author_renderer(card, profile)


def test_generation_reuses_corpus_executor_and_keeps_answers_separate(tmp_path):
    card, profile = card_fixture(tmp_path)
    calls = []

    def transport(url, payload):
        calls.append((url, payload))
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        if url.endswith("/tokenize"):
            return {"count": 7}
        assert "TEACHER_ONLY" not in json.dumps(payload)
        assert payload["max_tokens"] == 512
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"tool_calls": []}'}}
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5},
        }

    result = runner.execute(
        records(),
        output=tmp_path / "private",
        endpoint="http://127.0.0.1:8000/v1",
        model="explicit-fallback",
        max_tokens=512,
        context=1024,
        transport=transport,
        source_id="XLAM_TOOL",
        expected_count=3,
        input_renderer=runner.author_renderer(card, profile),
        protocol_profile="AUTHOR_FORMAT_CORPUS_NO_SCORE",
        chat_template_kwargs={"enable_thinking": False},
    )
    assert result["completed_count"] == 3 and len(calls) == 6
    assert (
        result["quality_scored"] is False
        and result["source_checkbox_eligible"] is False
    )
    saved = json.loads((tmp_path / "private/request-000000.private.json").read_text())
    assert saved["original_reference"] == {"answers": "TEACHER_ONLY"}
    assert saved["original_input"] == records()[0].task["input"]
    assert calls[0][1]["messages"] == calls[1][1]["messages"]


def test_xlam_cannot_fall_back_to_bare_query_generation(tmp_path):
    with pytest.raises(ValueError, match="author prompt renderer"):
        runner.execute(
            records(),
            output=tmp_path / "not-created",
            endpoint="http://127.0.0.1:8000/v1",
            model="explicit",
            max_tokens=512,
            context=1024,
            source_id="XLAM_TOOL",
        )
    assert not (tmp_path / "not-created").exists()


def test_model_format_deviation_is_retained_without_normalizing_or_scoring(tmp_path):
    card, profile = card_fixture(tmp_path)
    text = '```json\n{"tool_calls": []}\n```'

    def transport(url, payload):
        if url.endswith("/tokenize"):
            return {"count": 7}
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": text}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5},
        }

    receipt = runner.execute(
        records(),
        output=tmp_path / "private",
        endpoint="http://127.0.0.1:8000/v1",
        model="explicit-fallback",
        max_tokens=512,
        context=1024,
        transport=transport,
        source_id="XLAM_TOOL",
        expected_count=3,
        input_renderer=runner.author_renderer(card, profile),
    )
    saved = json.loads((tmp_path / "private/request-000000.private.json").read_text())
    assert saved["response"]["choices"][0]["message"]["content"] == text
    assert receipt["quality_scored"] is False


def test_existing_math_protocol_does_not_accept_prompt_replacement(tmp_path):
    with pytest.raises(ValueError, match="replacement renderer"):
        runner.execute(
            [],
            output=tmp_path / "not-created",
            endpoint="http://127.0.0.1:8000/v1",
            model="explicit",
            max_tokens=512,
            context=1024,
            input_renderer=lambda original: "rewritten",
        )


def test_plan_is_metadata_only(tmp_path, capsys):
    runner.main([])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PLAN_ONLY" and result["model_calls"] == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("role", ["data", "raw_source"])
def test_cli_uses_normalized_registry_data_role_without_losing_original_fields(
    tmp_path,
    monkeypatch,
    capsys,
    role,
):
    import vllm_hust_workload_data as provider
    import vllm_hust_workload_data.assets as assets
    import vllm_hust_workload_data.registry as registry

    card, profile = card_fixture(tmp_path)
    profile.update(original_max_new_tokens=512, recommended_model="author-model")
    manifest = tmp_path / "profile.json"
    manifest.write_text(json.dumps(profile))
    monkeypatch.setattr(runner, "PROFILE", manifest)
    monkeypatch.setattr(
        registry,
        "source_spec",
        lambda source: {"files": [{"path": "original.json", "role": role}]},
    )
    monkeypatch.setattr(assets, "verify_file", lambda *args: None)
    monkeypatch.setattr(
        provider, "iter_workload", lambda *args, **kwargs: iter(records())
    )
    execute = runner.execute

    def transport(url, payload):
        if url.endswith("/tokenize"):
            return {"count": 7}
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"tool_calls": []}'}}
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5},
        }

    monkeypatch.setattr(
        runner,
        "execute",
        lambda *args, **kwargs: execute(*args, **kwargs, transport=transport),
    )
    data = tmp_path / "original"
    data.mkdir()
    output = tmp_path / "private"
    runner.main(
        [
            "--mode",
            "run",
            "--data-root",
            str(data),
            "--author-card",
            str(card),
            "--output",
            str(output),
            "--model",
            "explicit-fallback",
            "--context",
            "1024",
            "--disable-thinking",
        ]
    )
    assert json.loads(capsys.readouterr().out)["completed_count"] == 3
    receipt = json.loads((output / "receipt-private.json").read_text())
    assert receipt["author_model_reproduced"] is False
    assert receipt["author_prompt"] == profile
