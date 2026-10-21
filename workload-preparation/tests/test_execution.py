"""Loopback software fixtures, never model or official task execution evidence."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from io import BytesIO

import pytest

from vllm_hust_workload_data.execution import _read_event_stream, replay_request
from vllm_hust_workload_data.records import WorkloadRecord


def record(body, kind="request_replay"):
    return WorkloadRecord(
        source={"source_id": "SOFTWARE_FIXTURE"},
        task={"input": {"request_body": body}},
        runtime={"kind": kind},
        replay={},
    )


@pytest.mark.parametrize("model_binding", [None, "explicit-local-model"])
def test_loopback_transport_preserves_original_request(model_binding):
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"fixture_response":true}')

        def log_message(self, *args):
            pass

    body = {
        "model": "mock-not-a-model",
        "messages": [{"role": "user", "content": "fixture"}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": 7,
    }
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = replay_request(
            record(body),
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            model=model_binding,
        )
        assert result == {"fixture_response": True}
        assert captured == [{**body, "model": model_binding or body["model"]}]
        assert body["model"] == "mock-not-a-model"
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


@pytest.mark.parametrize("kind", ["source_native_task", "metadata_trace"])
def test_task_or_trace_cannot_be_silently_rendered(kind):
    with pytest.raises(ValueError, match="not a ready model request"):
        replay_request(record({}, kind), endpoint="http://127.0.0.1:1/unused")


@pytest.mark.parametrize("body", [{"messages": []}, {"model": ""}])
def test_no_silent_request_rewrite(body):
    with pytest.raises(ValueError):
        replay_request(record(body), endpoint="http://127.0.0.1:1/unused")


def test_no_credentials_in_endpoint():
    with pytest.raises(ValueError, match="without credentials"):
        replay_request(
            record({"model": "fixture"}),
            endpoint="https://user:secret@example.org/v1",  # pragma: allowlist secret - invented rejected URL
        )


def test_stream_requires_actual_terminal_event():
    event = b'data: {"choices":[{"delta":{"content":"fixture"}}]}\n\n'
    result = _read_event_stream(
        BytesIO(b": heartbeat\n\n" + event + b"data: [DONE]\n\n")
    )
    assert result["terminated"] is True
    assert len(result["events"]) == 1
    with pytest.raises(ValueError, match="without"):
        _read_event_stream(BytesIO(event))
    with pytest.raises(ValueError, match="server error"):
        _read_event_stream(BytesIO(b'data: {"error":{"message":"fixture"}}\n\n'))
