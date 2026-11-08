"""Invented HTTP fixtures only: no original task, model or NPU execution."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
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


def saved_fixture(tmp_path):
    root = tmp_path / "saved"
    root.mkdir()
    identity = "a" * 32
    request = {
        "model": "fixture",
        "messages": [{"role": "user", "content": "invented"}],
        "max_tokens": 17,
    }
    response = {"choices": [{"message": {"content": "invented saved response"}}]}
    objects = {
        ".request.json": request,
        ".forwarded-request.json": {
            **request,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        ".response.raw": response,
        ".response.json": {"http_status": 200, "response": response},
    }
    hashes = {}
    for suffix, value in objects.items():
        raw = json.dumps(value).encode()
        (root / (identity + suffix)).write_bytes(raw)
        hashes[suffix] = hashlib.sha256(raw).hexdigest()
    manifest = {
        "source_id": "MULTIAGENTBENCH",
        "model": "fixture",
        "historical_backend_same_api_birth_verified": True,
        "original_nonthinking_binding": True,
        "rows": [{"identity": identity, "sha256": hashes}],
    }
    (root / "provenance-private.json").write_text(json.dumps(manifest))
    return root, request, response


def test_exact_saved_reply_needs_no_backend_call(tmp_path):
    root, request, response = saved_fixture(tmp_path)
    consumed = tmp_path / "consumed"
    consumed.mkdir()
    queue = tmp_path / "new-task"
    with transport.live_model_transport(
        queue,
        "http://127.0.0.1:1/v1",
        port=0,
        timeout=2,
        reuse_root=root,
        expected_model="fixture",
        reuse_state=consumed,
    ) as port:
        with urlopen(
            Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(request).encode(),
            ),
            timeout=4,
        ) as result:
            assert json.load(result) == response
    marker = json.loads(next(queue.glob("*.reuse-private.json")).read_text())
    assert marker["new_model_calls"] == 0
    assert marker["original_identity"] == "a" * 32
    assert len(list(consumed.iterdir())) == 2


def test_one_saved_sample_is_consumed_once_across_task_processes(tmp_path):
    root, request, _ = saved_fixture(tmp_path)
    consumed = tmp_path / "consumed"
    consumed.mkdir()
    for index in range(2):
        queue = tmp_path / str(index)
        queue.mkdir()
        (queue / ("b" * 32 + ".request.json")).write_text(json.dumps(request))
        cache = transport.saved_responses(root, "fixture")
        assert transport.reuse_one(queue, cache, consumed) is (index == 0)
        assert bool(list(queue.glob("*.response.json"))) is (index == 0)


def test_changed_budget_does_not_match_saved_request(tmp_path):
    root, request, _ = saved_fixture(tmp_path)
    queue = tmp_path / "new"
    queue.mkdir()
    consumed = tmp_path / "consumed"
    consumed.mkdir()
    (queue / ("b" * 32 + ".request.json")).write_text(
        json.dumps({**request, "max_tokens": 18})
    )
    assert not transport.reuse_one(
        queue, transport.saved_responses(root, "fixture"), consumed
    )
    assert not list(queue.glob("*.claim")) and not list(consumed.iterdir())


def test_uncertain_previous_consumption_cannot_fall_through_to_model(tmp_path):
    root, request, _ = saved_fixture(tmp_path)
    queue = tmp_path / "new"
    queue.mkdir()
    consumed = tmp_path / "consumed"
    consumed.mkdir()
    (consumed / ("a" * 32 + ".json")).write_text("invented partial reservation")
    (queue / ("b" * 32 + ".request.json")).write_text(json.dumps(request))
    with pytest.raises(ValueError, match="never resubmit"):
        transport.reuse_one(queue, transport.saved_responses(root, "fixture"), consumed)
    assert not list(queue.glob("*.claim")) and not list(queue.glob("*.response.json"))


def test_tampered_or_wrong_model_saved_reply_rejected(tmp_path):
    root, _, _ = saved_fixture(tmp_path)
    with pytest.raises(ValueError, match="model binding"):
        transport.saved_responses(root, "different-model")
    (root / ("a" * 32 + ".response.raw")).write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        transport.saved_responses(root, "fixture")


def test_known_unanswered_claim_is_never_automatically_forwarded(tmp_path):
    root, request, _ = saved_fixture(tmp_path)
    path = root / "provenance-private.json"
    manifest = json.loads(path.read_text())
    manifest["excluded_claim_without_saved_response"] = 1
    path.write_text(json.dumps(manifest))
    unknown = {**request, "max_tokens": 19}
    digest = hashlib.sha256(transport.canonical_request(unknown).encode()).hexdigest()
    (root / "uncertain-requests-private.json").write_text(
        json.dumps([{"canonical_request_sha256": digest}])
    )
    queue = tmp_path / "new"
    queue.mkdir()
    (queue / ("b" * 32 + ".request.json")).write_text(json.dumps(unknown))
    with pytest.raises(ValueError, match="known unanswered original claim"):
        transport.reuse_one(
            queue,
            transport.saved_responses(root, "fixture"),
            tmp_path,
            transport.uncertain_requests(root),
        )
    assert not list(queue.glob("*.claim"))


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
