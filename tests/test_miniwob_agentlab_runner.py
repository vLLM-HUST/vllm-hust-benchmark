"""Check the MiniWoB++ launcher without a browser or model dependency."""

import hashlib
import json
import runpy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_metadata_capture_preserves_response_and_restores_client(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_miniwob_agentlab.py"
    capture = runpy.run_path(str(script))["response_metadata"]
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="length", message=SimpleNamespace(content=None)
            ),
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content="private")
            ),
        ],
        usage=SimpleNamespace(completion_tokens=1024),
    )

    class Client:
        def create(self, **kwargs):
            assert kwargs == {"messages": ["private prompt"]}
            return response

    original = Client.create
    path = tmp_path / "metadata.jsonl"
    with pytest.raises(RuntimeError, match="parser failed"):
        with capture(Client, path):
            assert Client().create(messages=["private prompt"]) is response
            raise RuntimeError("parser failed")
    assert Client.create is original
    raw = path.read_text("utf-8")
    assert "private" not in raw
    assert [json.loads(line) for line in raw.splitlines()] == [
        {"finish_reason": "length", "content_is_null": True, "completion_tokens": 1024},
        {"finish_reason": "stop", "content_is_null": False, "completion_tokens": 1024},
    ]


@pytest.mark.parametrize(
    ("terminated", "truncated", "reward", "error", "valid"),
    [
        (True, False, 0.0, None, True),
        (False, True, 0.0, None, True),
        (False, False, 1.0, None, False),
        (True, False, "1.0", None, False),
        (True, False, 1.0, "browser failed", False),
    ],
)
def test_terminal_summary(tmp_path, terminated, truncated, reward, error, valid):
    script = Path(__file__).resolve().parents[1] / "scripts/run_miniwob_agentlab.py"
    read_summary = runpy.run_path(str(script))["terminal_summary"]
    (tmp_path / "summary_info.json").write_text(
        json.dumps(
            {
                "n_steps": 1,
                "cum_reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "err_msg": error,
            }
        ),
        encoding="utf-8",
    )
    if not valid:
        with pytest.raises(ValueError, match="invalid BrowserGym terminal summary"):
            read_summary(tmp_path)
        return
    assert read_summary(tmp_path) == {
        "steps": 1,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }


def run_cli(tmp_path, endpoint):
    script = Path(__file__).resolve().parents[1] / "scripts/run_miniwob_agentlab.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--task",
            "miniwob.click-dialog",
            "--site",
            "http://127.0.0.1:8001/miniwob/",
            "--endpoint",
            endpoint,
            "--model",
            "fixture",
            "--output",
            str(tmp_path / "result"),
            "--max-steps",
            "5",
            "--max-output-tokens",
            "64",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_default_plan_creates_no_result_directory(tmp_path):
    result = run_cli(tmp_path, "http://127.0.0.1:8002/v1")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "PLAN_ONLY"
    assert not (tmp_path / "result").exists()


def test_non_loopback_endpoint_is_rejected(tmp_path):
    result = run_cli(tmp_path, "http://192.0.2.1:8002/v1")
    assert result.returncode == 2
    assert "loopback" in result.stderr


def test_saved_summary_accepts_pinned_bytes_and_rejects_mutation(
    tmp_path, monkeypatch, capsys
):
    script = Path(__file__).resolve().parents[1] / "scripts/run_miniwob_agentlab.py"
    namespace = runpy.run_path(str(script))
    main = namespace["main"]
    task = "miniwob.multi-layouts"
    path = tmp_path / "summary_info.json"
    raw = b'{"n_steps":20,"cum_reward":0,"terminated":false,"truncated":true}'
    path.write_bytes(raw)
    monkeypatch.setitem(
        main.__globals__["SAVED_PROFILES"],
        task,
        (hashlib.sha256(raw).hexdigest(), 4096, (20, 0, False, True)),
    )
    monkeypatch.setattr(
        sys, "argv", [str(script), "--task", task, "--verify-saved", str(tmp_path)]
    )
    main()
    assert json.loads(capsys.readouterr().out)["status"] == "SAVED_SUMMARY_VERIFIED"
    path.write_bytes(raw + b"\n")
    with pytest.raises(SystemExit, match="2"):
        main()
    assert "does not match" in capsys.readouterr().err
