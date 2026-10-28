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
    *,
    overflow=False,
    terminal="stop",
    usage=True,
    endpoint="http://127.0.0.1:19466/v1",
    failure=None,
    stream_failure=False,
    partial_stream=False,
    empty=False,
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
        CompileStatusCode=NS(OK=0, UNSUPPORTED_SCHEMA=1),
        DecodingStatusCode=NS(OK=0),
    )
    calls = []

    def complete(**body):
        calls.append(body)
        if failure is not None and not stream_failure:
            raise failure
        if stream_failure:

            def broken():
                if partial_stream:
                    yield NS(
                        usage=None,
                        choices=[
                            NS(index=0, finish_reason=None, delta=NS(content="partial"))
                        ],
                    )
                raise failure

            return broken()
        return [
            NS(
                usage=None,
                choices=[
                    NS(
                        index=0,
                        finish_reason=None,
                        delta=NS(content="" if empty else '{"fixture":'),
                    )
                ],
            ),
            NS(
                usage=None,
                choices=[
                    NS(
                        index=0,
                        finish_reason=terminal,
                        delta=NS(content="" if empty else "1}"),
                    )
                ],
            ),
            NS(
                usage=NS(prompt_tokens=10, completion_tokens=0 if empty else 5)
                if usage
                else None,
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


class APIError(Exception):
    status_code = None

    def __init__(self, body):
        self.body = body
        super().__init__("inert structured compiler error")


@pytest.mark.parametrize(
    "message",
    [
        "Unsupported JSON Schema structure INERT\nMake sure it is valid to the JSON Schema specification and check if it's supported by Outlines.",
        "Grammar error: inert compiler fixture, no dataset payload",
        "The provided JSON schema contains features not supported by xgrammar.",
    ],
)
def test_streamed_explicit_compiler_rejection_is_not_an_arbitrary_api_failure(message):
    body = {
        "type": "BadRequestError",
        "code": 400,
        "message": message,
    }
    engine, _ = fixture(failure=APIError(body), stream_failure=True)
    result = output()
    engine._generate(result)
    assert result.metadata.compile_status.code == 1
    assert result.token_usage.output_tokens == 0
    for changed in [
        {**body, "code": 500},
        {**body, "code": "400"},
        {**body, "type": "InternalServerError"},
        {**body, "message": "unrelated failure"},
        {
            **body,
            "message": "V1 structured outputs only supports one backend per engine.",
        },
    ]:
        engine, _ = fixture(failure=APIError(changed), stream_failure=True)
        with pytest.raises(APIError):
            engine._generate(output())
    engine, _ = fixture(
        failure=APIError(body), stream_failure=True, partial_stream=True
    )
    with pytest.raises(APIError):
        engine._generate(output())


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


def test_terminal_empty_output_is_kept_for_official_quality_evaluation():
    engine, calls = fixture(empty=True)
    result = output()
    engine._generate(result)
    assert len(calls) == 1
    assert result.generation == "" and result.generated_tokens == []
    assert result.token_usage.output_tokens == 0
    assert result.metadata.first_token_arrival_time is None
    assert result.metadata.compile_status.code == 0


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


class FakeAPIError(Exception):
    def __init__(self, status_code, code):
        self.status_code = status_code
        self.body = {"error": {"code": code}}


def test_explicit_original_schema_rejection_is_not_rewritten_or_retried():
    engine, calls = fixture(failure=FakeAPIError(400, "unsupported_json_schema"))
    result = output()
    before = deepcopy(result.schema)
    engine._generate(result)
    assert result.metadata.compile_status.code == 1
    assert result.schema == before
    assert len(calls) == 1
    assert result.token_usage.output_tokens == 0


@pytest.mark.parametrize(
    "status,code",
    [
        (500, "unsupported_json_schema"),
        (400, "model_not_found"),
        (429, "rate_limit_exceeded"),
    ],
)
def test_infrastructure_or_configuration_errors_are_not_schema_failures(status, code):
    engine, calls = fixture(failure=FakeAPIError(status, code))
    with pytest.raises(FakeAPIError):
        engine._generate(output())
    assert len(calls) == 1
