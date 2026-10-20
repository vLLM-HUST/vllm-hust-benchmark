"""Local timing for official Agent queries and tools; no backend-specific API."""

import hashlib
import json
from pathlib import Path
import time


def instrument(model, env, folder, program):
    path = Path(folder) / "agent-events-private.jsonl"
    ordinal = 0

    def record(event, **values):
        with path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": event,
                        "epoch": time.time(),
                        "monotonic": time.monotonic(),
                        **values,
                    }
                )
                + "\n"
            )

    original_query, original_execute = model.query, env.execute

    def query(messages, **kwargs):
        nonlocal ordinal
        ordinal += 1
        request_id = program + f":query-{ordinal:03d}"
        headers = dict(model.config.model_kwargs.get("extra_headers", {}))
        headers.update(kwargs.pop("extra_headers", {}))
        headers["X-Delivery-Request-ID"] = request_id
        record("agent_query_start", request_id=request_id, call=ordinal)
        try:
            response = original_query(messages, extra_headers=headers, **kwargs)
            raw = response.get("extra", {}).get("response", {})
            record(
                "agent_query_return",
                request_id=request_id,
                call=ordinal,
                response_id=raw.get("id"),
                usage=raw.get("usage"),
                content_sha256=hashlib.sha256(response["content"].encode()).hexdigest(),
            )
            return response
        except BaseException as exc:
            record(
                "agent_query_exception",
                request_id=request_id,
                call=ordinal,
                error=type(exc).__name__,
                detail=str(exc),
            )
            raise

    def execute(command, *args, **kwargs):
        record("tool_start", call=ordinal, command=command)
        try:
            result = original_execute(command, *args, **kwargs)
            record(
                "tool_return",
                call=ordinal,
                returncode=result.get("returncode"),
                output=result.get("output", ""),
            )
            return result
        except BaseException as exc:
            record(
                "tool_exception",
                call=ordinal,
                error=type(exc).__name__,
                detail=str(exc),
            )
            raise

    model.query, env.execute = query, execute
