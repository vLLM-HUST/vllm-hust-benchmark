"""Explicit single-request transport; never renders tasks or synthesizes inputs.

Native task loops, media renderers and scoring belong to their official runners.
This helper only accepts a source record already declaring request_replay.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .consumer import resolve_input


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward private input or credentials to a different destination.
        return None


def _read_event_stream(response):
    events, data = [], []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            payload = "\n".join(data)
            data = []
            if payload == "[DONE]":
                return {
                    "stream": True,
                    "events": events,
                    "terminated": True,
                    "termination": "[DONE]",
                }
            event = json.loads(payload)
            if not isinstance(event, dict) or event.get("error"):
                raise ValueError("stream returned an invalid event or server error")
            events.append(event)
    raise ValueError("stream ended without the required [DONE] terminal event")


def replay_request(record, *, endpoint, api_key=None, timeout=120.0, model=None):
    """Send original content to an explicitly authorized endpoint.

    Returns a response, not a score or a benchmark performance measurement.
    Streaming requests preserve their original protocol and require [DONE].
    An explicit model binding may change only the model routing field.
    """
    body = record.request_body()  # Refuses static tasks and metadata-only traces.
    if model is not None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("explicit served model must be a nonempty string")
        body["model"] = model
    if not isinstance(body.get("model"), str) or not body["model"].strip():
        raise ValueError(
            "original request has no model; use an explicit source-native runner"
        )
    target = urlsplit(endpoint)
    if (
        target.scheme not in {"http", "https"}
        or not target.netloc
        or target.username is not None
        or target.password is not None
        or target.fragment
        or target.query
    ):
        raise ValueError(
            "endpoint must be an explicit HTTP(S) API URL without credentials or query"
        )
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        if body.get("stream"):
            return _read_event_stream(response)
        return json.loads(response.read().decode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--endpoint", required=True, help="Full chat/completions or completions URL"
    )
    parser.add_argument(
        "--api-key-env", help="Environment variable name; never pass a key in argv"
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--model",
        help="Explicit served-model binding; original messages and other parameters are unchanged",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Fresh private JSON output; may contain sensitive model output",
    )
    args = parser.parse_args(argv)
    # Reserve output before any network call; never overwrite a prior result.
    try:
        key = os.environ[args.api_key_env] if args.api_key_env else None
        record = resolve_input(args.database, args.record_id, data_root=args.data_root)
        with args.output.open("x", encoding="utf-8") as output:
            response = replay_request(
                record,
                endpoint=args.endpoint,
                api_key=key,
                timeout=args.timeout,
                model=args.model,
            )
            json.dump(
                {
                    "schema_version": "source-request-response/v1",
                    "response": response,
                    "scoring": "NOT_RUN",
                    "native_task_execution": "NOT_RUN",
                    "model_binding": {
                        "original_model": record.request_body().get("model"),
                        "served_model": args.model
                        or record.request_body().get("model"),
                        "explicit_override": args.model is not None,
                    },
                },
                output,
            )
            output.write("\n")
    except Exception as error:
        # Transport errors may include private URLs or input. Do not echo them.
        parser.exit(
            1, f"request replay failed ({type(error).__name__}); no score produced\n"
        )
    print(
        json.dumps(
            {"status": "HTTP_RESPONSE_SAVED_NOT_TASK_VALIDATED", "scoring": "NOT_RUN"}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
