"""Author-protocol fixtures, not real Agent or model execution."""

import json
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mlagentbench_local_messages import (
    original_messages_request,
    complete_original_messages,
    local_endpoint,
)
import mlagentbench_local_messages as messages_transport


def test_isolated_file_queue_preserves_messages_bytes(tmp_path, monkeypatch):
    calls = []

    class Response:
        code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return json.dumps(
                dict(
                    content=[
                        dict(type="thinking", thinking="fixture reasoning"),
                        dict(type="text", text="queued original output"),
                    ]
                )
            ).encode()

    def opener(request, timeout):
        calls.append(request)
        return Response()

    monkeypatch.setattr(messages_transport, "urlopen", opener)

    def worker():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if messages_transport.process_messages_one(
                tmp_path, "http://127.0.0.1:19476/v1", model="Qwen/actual"
            ):
                return
            time.sleep(0.01)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    result = complete_original_messages(
        "original",
        "Qwen/actual",
        api_base="http://127.0.0.1:14001/v1",
        allowed_model="Qwen/actual",
        opener=messages_transport.queue_opener(tmp_path),
        timeout=5,
    )
    thread.join(timeout=5)
    assert (
        not thread.is_alive() and result == "queued original output" and len(calls) == 1
    )
    request = next(tmp_path.glob("*.request.json"))
    forwarded = request.with_name(
        request.name.replace(".request.json", ".forwarded-request.json")
    )
    assert request.read_bytes() == forwarded.read_bytes() == calls[0].data
    assert calls[0].full_url == "http://127.0.0.1:19476/v1/messages"


def test_original_messages_defaults_and_explicit_overrides():
    original = "unmodified prompt"
    body = original_messages_request(original, "Qwen/actual")
    assert body == dict(
        model="Qwen/actual",
        messages=[dict(role="user", content=original)],
        max_tokens=2000,
        temperature=0.5,
        stop_sequences=["\n\nHuman:", "Observation:"],
    )
    messages = [
        dict(role="user", content="original"),
        dict(role="assistant", content="original history"),
    ]
    override = original_messages_request(
        original,
        "Qwen/actual",
        max_tokens_to_sample=4000,
        temperature=0.01,
        messages=messages,
        stop_sequences=[],
    )
    assert override["messages"] is messages
    assert (
        override["max_tokens"] == 4000
        and override["temperature"] == 0.01
        and override["stop_sequences"] == []
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1/v1",
        "http://example.com/v1",
        "http://u:p@localhost/v1",  # pragma: allowlist secret - rejected fixture URL
        "http://localhost/v1?q=1",
    ],
)
def test_no_remote_or_credentialed_route(endpoint):
    with pytest.raises(ValueError):
        local_endpoint(endpoint)


def test_transport_preserves_actual_request_and_original_first_text():
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return json.dumps(
                dict(content=[dict(type="text", text="original output")])
            ).encode()

    def opener(request, timeout):
        calls.append((request, timeout))
        return Response()

    result = complete_original_messages(
        "unchanged",
        "Qwen/actual",
        api_base="http://127.0.0.1:19476/v1",
        allowed_model="Qwen/actual",
        opener=opener,
    )
    assert result == "original output" and len(calls) == 1
    request, timeout = calls[0]
    assert request.full_url == "http://127.0.0.1:19476/v1/messages" and timeout == 600
    assert json.loads(request.data) == original_messages_request(
        "unchanged", "Qwen/actual"
    )
    assert "authorization" not in {k.lower() for k in request.headers}
