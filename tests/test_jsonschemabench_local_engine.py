"""Fake author-interface/transport fixtures; no official dataset/native run."""

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "local_schema_engine", ROOT / "scripts/jsonschemabench_local_engine.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(
    *, overflow=False, terminal="stop", usage=True, endpoint="http://127.0.0.1:19466/v1"
):
    class Base:
        def __init__(self, config):
            self.config = config

    class Config:
        pass

    types = NS(
        Token=lambda **kw: NS(**kw),
        CompileStatus=lambda **kw: NS(**kw),
        DecodingStatus=lambda **kw: NS(**kw),
        CompileStatusCode=NS(OK=0),
        DecodingStatusCode=NS(OK=0),
    )
    calls = []

    def complete(**body):
        calls.append(body)
        return [
            NS(
                usage=None,
                choices=[
                    NS(index=0, finish_reason=None, delta=NS(content='{"fixture":'))
                ],
            ),
            NS(
                usage=None,
                choices=[NS(index=0, finish_reason=terminal, delta=NS(content="1}"))],
            ),
            NS(
                usage=NS(prompt_tokens=10, completion_tokens=5) if usage else None,
                choices=[],
            ),
        ]

    client = NS(
        base_url=endpoint, max_retries=0, chat=NS(completions=NS(create=complete))
    )
    engine = runner.create_engine(
        engine_module=NS(Engine=Base, EngineConfig=Config),
        types_module=types,
        client=client,
        token_count=lambda _: 100 if overflow else 10,
        model="fixture-model",
        max_tokens=50,
        context_limit=100,
    )
    return engine, calls


def output():
    return NS(
        messages=[{"role": "user", "content": "fixture original messages"}],
        schema={
            "properties": {"fixture": {"type": "number"}},
            "additionalProperties": True,
        },
        token_usage=NS(input_tokens=0, output_tokens=0),
        metadata=NS(),
    )


def test_schema_and_author_messages_are_unchanged():
    engine, calls = fixture()
    result = output()
    schema = deepcopy(result.schema)
    assert engine.adapt_schema(engine.adapt_schema(schema)) == schema
    engine._generate(result)
    assert calls[0]["messages"] == result.messages
    assert (
        calls[0]["response_format"]["json_schema"]["schema"] == schema == result.schema
    )
    assert "required" not in schema and "type" not in schema
    assert result.generation == '{"fixture":1}'
    assert result.token_usage.input_tokens == 10
    assert result.token_usage.output_tokens == 5
    assert engine.max_context_length == 100


def test_overflow_never_trims_or_sends_request():
    engine, calls = fixture(overflow=True)
    with pytest.raises(ValueError, match="no trimming"):
        engine._generate(output())
    assert calls == []


@pytest.mark.parametrize(
    "kwargs", [{"terminal": None}, {"terminal": "tool_calls"}, {"usage": False}]
)
def test_incomplete_stream_never_marked_ok(kwargs):
    engine, _ = fixture(**kwargs)
    with pytest.raises(ValueError, match="terminal completion"):
        engine._generate(output())


def test_external_client_rejected_before_requests():
    with pytest.raises(ValueError, match="loopback-only"):
        fixture(endpoint="https://example.invalid/v1")
