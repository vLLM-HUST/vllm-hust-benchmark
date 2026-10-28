"""Loopback fixture protocol only, no original data, model or generated code."""

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import subprocess
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from appworld_model_transport import model_proxy, process_one


@pytest.mark.parametrize("disable_thinking", [False, True])
def test_offline_queue_roundtrip_preserves_request_budget_and_no_repeat(
    tmp_path, disable_thinking
):
    received = []

    class Fixture(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            received.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            wire = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": "fixture-only"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

    service = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    body = {
        "model": "fixture-model",
        "messages": [{"role": "user", "content": "fixture-only"}],
        "max_tokens": 400,
        "seed": 123,
        "temperature": 0,
    }
    raw = json.dumps(body).encode()
    try:
        with (
            model_proxy(tmp_path, port=0, timeout=10) as port,
            ThreadPoolExecutor() as pool,
        ):

            def request():
                with urlopen(
                    Request(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        data=raw,
                        headers={"Content-Type": "application/json"},
                    ),
                    timeout=10,
                ) as response:
                    return json.loads(response.read())

            future = pool.submit(request)
            deadline = time.monotonic() + 5
            while not list(tmp_path.glob("*.request.json")):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            endpoint = f"http://127.0.0.1:{service.server_port}/v1"
            assert process_one(tmp_path, endpoint, disable_thinking=disable_thinking)
            assert future.result()["choices"][0]["message"]["content"] == "fixture-only"
            assert not process_one(tmp_path, endpoint)
        expected = dict(body)
        if disable_thinking:
            expected["chat_template_kwargs"] = {"enable_thinking": False}
        assert received == [expected]
        assert next(tmp_path.glob("*.request.json")).read_bytes() == raw
        assert (
            json.loads(next(tmp_path.glob("*.forwarded-request.json")).read_bytes())
            == expected
        )
        assert len(list(tmp_path.glob("*.response.raw"))) == 1
    finally:
        service.shutdown()
        service.server_close()
        thread.join()


def test_explicit_thinking_binding_never_overrides_original_conflict(tmp_path):
    identity = "0" * 32
    (tmp_path / (identity + ".request.json")).write_text(
        json.dumps({"chat_template_kwargs": {"enable_thinking": True}})
    )
    with pytest.raises(ValueError, match="conflicting thinking"):
        process_one(tmp_path, "http://127.0.0.1:1/v1", disable_thinking=True)
    assert not list(tmp_path.glob("*.response.json"))


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.com/v1",
        "http://user:password@127.0.0.1/v1",  # pragma: allowlist secret - rejection fixture, not credentials
        "http://192.0.2.1/v1",
    ],
)
def test_worker_requires_owned_loopback_binding(tmp_path, endpoint):
    with pytest.raises(ValueError, match="owned loopback"):
        process_one(tmp_path, endpoint)


def test_bounded_watch_without_jobs_makes_no_model_request(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/appworld_model_transport.py"
            ),
            "--queue",
            str(tmp_path),
            "--api-base",
            "http://127.0.0.1:1/v1",
            "--watch",
            "--idle-timeout",
            "0.01",
            "--max-jobs",
            "2",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(result.stdout)["jobs_returned"] == 0
    assert not list(tmp_path.iterdir())
