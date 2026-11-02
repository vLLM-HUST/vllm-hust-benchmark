"""Invented HTTP fixtures only: no original task, model or NPU execution."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import marble_live_model_transport as transport  # noqa: E402


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:1/v1",
        "http://example.com:1/v1",
        "http://user@127.0.0.1:1/v1",
        "http://127.0.0.1/v1",
        "http://127.0.0.1:1/v1?key=fixture",
    ],
)
def test_reject_endpoint_before_creating_queue(tmp_path, endpoint):
    with pytest.raises(ValueError, match="loopback"):
        with transport.live_model_transport(tmp_path / "queue", endpoint):
            pytest.fail("entered invalid transport")
    assert not list(tmp_path.iterdir())


def test_existing_queue_never_replayed(tmp_path):
    queue = tmp_path / "old-queue"
    queue.mkdir()
    old = queue / ("a" * 32 + ".request.json")
    old.write_text('{"model":"invented fixture"}')
    before = old.read_bytes()
    with pytest.raises(ValueError, match="no orphan replay"):
        with transport.live_model_transport(queue, "http://127.0.0.1:1/v1"):
            pytest.fail("old queue replayed")
    assert old.read_bytes() == before and list(queue.iterdir()) == [old]


def test_backend_cannot_loop_into_the_proxy(tmp_path):
    with pytest.raises(ValueError, match="point back"):
        with transport.live_model_transport(
            tmp_path / "queue", "http://127.0.0.1:14005/v1"
        ):
            pytest.fail("recursive proxy entered")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("status", [200, 500])
def test_live_request_forwarded_and_original_response_preserved(tmp_path, status):
    calls = []
    response = (
        {"choices": [{"message": {"content": "invented fixture"}}]}
        if status == 200
        else {"error": {"type": "fixture_backend_failure"}}
    )

    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            calls.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            raw = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    queue = tmp_path / "fresh-task"
    original = {
        "model": "fixture",
        "messages": [{"role": "user", "content": "invented"}],
        "max_tokens": 17,
    }
    try:
        with transport.live_model_transport(
            queue, f"http://127.0.0.1:{backend.server_port}/v1", port=0, timeout=3
        ) as port:
            request = Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(original).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                result = urlopen(request, timeout=5)
            except HTTPError as error:
                result = error
            with result:
                assert result.status == status and json.load(result) == response
        assert calls == [
            {**original, "chat_template_kwargs": {"enable_thinking": False}}
        ]
        assert len(list(queue.glob("*.claim"))) == 1
        assert json.loads(next(queue.glob("*.request.json")).read_text()) == original
        saved = json.loads(next(queue.glob("*.response.json")).read_text())
        assert saved == {"http_status": status, "response": response}
        time.sleep(0.1)
        assert len(calls) == 1
    finally:
        backend.shutdown()
        backend.server_close()
        thread.join()


def test_worker_failure_is_retained_and_propagated(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ConnectionRefusedError("invented unavailable backend")

    monkeypatch.setattr(transport, "process_one", fail)
    queue = tmp_path / "fresh-task"
    with pytest.raises(RuntimeError, match="retained private error"):
        with transport.live_model_transport(
            queue, "http://127.0.0.1:1/v1", port=0, timeout=2
        ):
            pass
    assert (
        json.loads((queue / "worker-error-private.json").read_text())["type"]
        == "ConnectionRefusedError"
    )
    assert not list(queue.glob("*.response.json"))
