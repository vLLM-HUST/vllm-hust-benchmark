"""Author Engine extension for local OpenAI transport; never adapt schemas.

Pass the verified author's core.engine/core.types modules and a loopback-only
client from a server launcher. This module makes no dataset/runner completion
claim and supplies no alternate formatter, dataset, evaluator or analysis.
"""

from dataclasses import dataclass
from copy import deepcopy
from time import time
from urllib.parse import urlparse


def explicit_schema_rejection(error):
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    detail = body.get("error", body)
    if not isinstance(detail, dict):
        return False
    if getattr(error, "status_code", None) == 400 and detail.get("code") in {
        "invalid_json_schema",
        "unsupported_json_schema",
        "unsupported_schema",
    }:
        return True
    # OpenAI SDK raises APIError during SSE with no status_code; preserve the
    # server's structured 400 compiler verdict, never infer it from arbitrary text.
    message = detail.get("message")
    return (
        type(error).__name__ == "APIError"
        and getattr(error, "status_code", None) is None
        and detail.get("type") == "BadRequestError"
        and type(detail.get("code")) is int
        and detail["code"] == 400
        and isinstance(message, str)
        and (
            (
                message.startswith("Unsupported JSON Schema structure ")
                and "\nMake sure it is valid to the JSON Schema specification and check if it's supported by Outlines."
                in message
            )
            # Publisher Outlines compilation likewise records compiler
            # exceptions as UNSUPPORTED_SCHEMA, not a successful generation.
            or message.startswith("Grammar error:")
            or message
            == "The provided JSON schema contains features not supported by xgrammar."
        )
    )


def create_engine(
    *,
    engine_module,
    types_module,
    client,
    token_count,
    model,
    max_tokens,
    context_limit,
    temperature=0,
):
    route = urlparse(str(client.base_url))
    if getattr(client, "max_retries", None) != 0:
        raise ValueError("disable hidden client retries in explicit local profile")
    if (
        route.scheme != "http"
        or route.hostname not in {"127.0.0.1", "localhost", "::1"}
        or route.path.rstrip("/") != "/v1"
        or route.username
        or route.password
        or route.query
        or route.fragment
    ):
        raise ValueError("loopback-only local client required")
    if (
        type(max_tokens) is not int
        or max_tokens <= 0
        or type(context_limit) is not int
        or context_limit <= max_tokens
    ):
        raise ValueError("explicit positive local profile budgets required")

    @dataclass
    class LocalConfig(engine_module.EngineConfig):
        model_name: str
        max_tokens: int
        context_limit: int
        temperature: float

    class LocalEngine(engine_module.Engine):
        name = "local_original_schema"

        @property
        def max_context_length(self):
            return self.config.context_limit

        def adapt_schema(self, schema):
            # Author bench calls this twice. Identity/deepcopy is idempotent.
            return deepcopy(schema)

        def _generate(self, output):
            messages, schema = deepcopy(output.messages), deepcopy(output.schema)
            count = token_count(messages)
            if (
                type(count) is not int
                or count < 1
                or count + max_tokens > context_limit
            ):
                raise ValueError(
                    "original messages plus generation budget exceed context; no trimming"
                )
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"schema": schema, "name": "json_schema"},
                    },
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream_options={"include_usage": True},
                )
            except Exception as error:
                if not explicit_schema_rejection(error):
                    raise
                output.metadata.compile_status = types_module.CompileStatus(
                    code=types_module.CompileStatusCode.UNSUPPORTED_SCHEMA,
                    message="Server explicitly rejected the unchanged original schema; no replacement.",
                )
                return
            chunks, terminal, usage, first = [], None, None, None
            try:
                for event in response:
                    if event.usage is not None:
                        usage = event.usage
                    for choice in event.choices:
                        if choice.index != 0:
                            raise ValueError(
                                "single original completion stream required"
                            )
                        if choice.finish_reason is not None:
                            terminal = choice.finish_reason
                        text = choice.delta.content
                        if text is not None:
                            if not isinstance(text, str):
                                raise ValueError("invalid original streaming content")
                            if first is None and text:
                                first = time()
                            chunks.append(text)
            except Exception as error:
                if (
                    chunks
                    or terminal is not None
                    or usage is not None
                    or not explicit_schema_rejection(error)
                ):
                    raise
                output.metadata.compile_status = types_module.CompileStatus(
                    code=types_module.CompileStatusCode.UNSUPPORTED_SCHEMA,
                    message=str(error),
                )
                return
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
            if terminal not in {"stop", "length"} or usage is None:
                raise ValueError(
                    "original terminal completion and measured usage required"
                )
            if (
                type(usage.prompt_tokens) is not int
                or type(usage.completion_tokens) is not int
                or usage.prompt_tokens < 1
                or usage.completion_tokens < 0
            ):
                raise ValueError("invalid service token usage")
            if usage.completion_tokens == 0 and any(chunks):
                raise ValueError("nonempty text contradicts zero completion tokens")
            if (
                usage.completion_tokens > max_tokens
                or usage.prompt_tokens + max_tokens > context_limit
            ):
                raise ValueError(
                    "service usage exceeds declared original local profile"
                )
            output.generation = "".join(chunks)
            output.generated_tokens = [
                types_module.Token(id=None, text=text) for text in chunks if text
            ]
            output.token_usage.input_tokens = usage.prompt_tokens
            output.token_usage.output_tokens = usage.completion_tokens
            output.metadata.first_token_arrival_time = first
            output.metadata.compile_status = types_module.CompileStatus(
                code=types_module.CompileStatusCode.OK
            )
            output.metadata.decoding_status = types_module.DecodingStatus(
                code=types_module.DecodingStatusCode.OK
            )

    return LocalEngine(LocalConfig(model, max_tokens, context_limit, temperature))
