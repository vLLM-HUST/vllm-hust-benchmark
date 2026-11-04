"""Exercise the saved-output contract without upstream dependencies."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/score_api_bank_saved.py"
FILES = (
    "AddAgenda-AddAlarm-GetUserToken-level-2-1.jsonl",
    "AddAgenda-AddMeeting-GetUserToken-level-2-2.jsonl",
    "AddAgenda-AddMeeting-GetUserToken-level-2-3.jsonl",
)


@pytest.mark.parametrize(
    "mode",
    ["complete", "missing", "duplicate", "unexpected", "non_object", "no_output"],
)
def test_fixed_scoring_positions(tmp_path, mode):
    inputs = tmp_path / "lv1-lv2-samples/level-1-given-desc"
    inputs.mkdir(parents=True)
    (tmp_path / "api_call_extraction.py").write_text(
        "def get_api_call(output): return output\n"
        "def parse_api_call(call): return call, {}\n",
        encoding="utf-8",
    )
    (tmp_path / "evaluator_by_json.py").write_text(
        "from types import SimpleNamespace\n"
        "class Sample:\n"
        "    @staticmethod\n"
        "    def from_chat_history(rows):\n"
        "        return [SimpleNamespace(ground_truth=r) for r in rows]\n"
        "class Evaluator:\n"
        "    def __init__(self, samples): pass\n"
        "    def get_model_input(self, i): return 'tools', [{'role': 'User', 'text': str(i)}]\n"
        "    def evaluate(self, i, output):\n"
        "        return (True,) if i == 0 else 0.5\n",
        encoding="utf-8",
    )
    predictions = []
    for name in FILES:
        rows = [{"role": "API", "api_name": "AddAgenda"}, {"role": "AI"}]
        (inputs / name).write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
        predictions.extend(
            {"file": name, "id": i, "output": output}
            for i, output in enumerate(("AddAgenda", "reply"))
        )
    predictions[2]["output"] = "UnlistedTool"
    predictions[3]["output"] = ""
    if mode == "missing":
        predictions.pop()
    elif mode == "duplicate":
        predictions.append(predictions[0])
    elif mode == "unexpected":
        predictions[-1]["id"] = 10
    elif mode == "non_object":
        predictions[-1] = []
    elif mode == "no_output":
        predictions[-1].pop("output")
    saved = tmp_path / "saved.jsonl"
    saved.write_text("\n".join(map(json.dumps, predictions)), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--author-root",
            str(tmp_path),
            "--predictions",
            str(saved),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if mode == "complete":
        (tmp_path / "evaluator.py").write_text(
            "api_call_prompt = 'call:'\nresponse_prompt = 'reply:'\n",
            encoding="utf-8",
        )
        messages = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--author-root",
                str(tmp_path),
                "--export-messages",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        rows = [json.loads(line) for line in messages.stdout.splitlines()]
        assert len(rows) == 6
        assert [row["messages"][0]["content"] for row in rows] == [
            "call:tools",
            "reply:tools",
        ] * 3
        assert result.returncode == 0, result.stderr
        scored = json.loads(result.stdout)
        assert [row["score"] for row in scored] == [
            True,
            0.5,
            False,
            0,
            True,
            0.5,
        ]
        exported = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--author-root",
                str(tmp_path),
                "--export-inputs",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        requests = [json.loads(line) for line in exported.stdout.splitlines()]
        assert [(r["file"], r["id"]) for r in requests] == [
            (r["file"], r["id"]) for r in scored
        ]
        assert all(
            set(r) == {"file", "id", "api_descriptions", "chat_history"}
            for r in requests
        )
    else:
        assert result.returncode == 2
        assert not result.stdout


def test_author_messages_preserves_tool_history():
    import runpy

    author_messages = runpy.run_path(str(SCRIPT))["author_messages"]
    assert author_messages(
        "prompt",
        "tools",
        [
            {
                "role": "API",
                "api_name": "AddAgenda",
                "param_dict": {"id": 7},
                "result": {"output": "ok"},
            },
        ],
    ) == [
        {"role": "system", "content": "prompttools"},
        {"role": "system", "content": "[AddAgenda(id='7')] Response: ok"},
    ]


def test_generate_rejects_external_endpoint(tmp_path):
    import runpy

    generate = runpy.run_path(str(SCRIPT))["generate"]
    output = tmp_path / "private.jsonl"
    with pytest.raises(ValueError, match="loopback"):
        generate("https://example.org/v1", "fixture", [], output)
    assert not output.exists()


def test_generate_preserves_completed_rows_on_http_failure(tmp_path):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import runpy
    from threading import Thread

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                (
                    self.path,
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
            if len(received) == 2:
                self.send_error(500)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b'{"choices":[{"message":{"content":"reply"},"finish_reason":"stop"}],'
                b'"usage":{"completion_tokens":1}}'
            )

    generate = runpy.run_path(str(SCRIPT))["generate"]
    messages = [{"role": "user", "content": "fixture"}]
    requests = [{"file": FILES[0], "id": i, "messages": messages} for i in range(3)]
    output = tmp_path / "private.jsonl"
    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}/v1"
            with pytest.raises(RuntimeError, match="HTTP 500"):
                generate(endpoint, "fixture", requests, output)
            with pytest.raises(FileExistsError):
                generate(endpoint, "fixture", requests, output)
        finally:
            server.shutdown()
            worker.join()
    assert (
        received
        == [("/v1/chat/completions", {"model": "fixture", "messages": messages})] * 2
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "file": FILES[0],
        "id": 0,
        "output": "reply",
        "finish_reason": "stop",
        "usage": {"completion_tokens": 1},
    }
