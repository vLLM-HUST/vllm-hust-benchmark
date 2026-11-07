"""Original budget and queue fixtures; never execute code or call models."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "discovery_transport", ROOT / "scripts/discoverybench_tool_transport.py"
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def test_original_one_call_limit_even_after_infrastructure_failure():
    calls = []

    def fail(code):
        calls.append(code)
        raise RuntimeError("fixture infrastructure failure")

    class Tool:
        max_turns = 0
        run = entry.budgeted_python_run(fail)

    tool = Tool()
    with pytest.raises(RuntimeError):
        tool.run("inert fixture")
    assert tool.run("different inert fixture") == entry.LIMIT_MESSAGE
    assert calls == ["inert fixture"]


def test_bound_original_counter_and_exact_output():
    class Tool:
        max_turns = 0
        run = entry.budgeted_python_run(lambda code: "exact output\n")

    tool = Tool()
    assert tool.run("inert") == "exact output\n"
    assert tool.max_turns == 1
    assert tool.run("inert") == entry.LIMIT_MESSAGE


def test_queue_response_bound_to_original_code_and_no_second_submit(tmp_path):
    code = "inert not executed"
    response = tmp_path / "original-python.response.json"

    def reply():
        deadline = time.monotonic() + 5
        while not (tmp_path / "original-python.request.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        entry.save_new(
            response,
            json.dumps(
                {
                    "status": "ORIGINAL_TOOL_RETURNED",
                    "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                    "tool_output": "original fixture output",
                }
            ).encode(),
        )

    worker = threading.Thread(target=reply)
    worker.start()
    try:
        assert (
            entry.request_once(tmp_path, code, timeout=5) == "original fixture output"
        )
    finally:
        worker.join(timeout=5)
    with pytest.raises(RuntimeError, match="never resubmit"):
        entry.request_once(tmp_path, code)


def test_observation_timeout_preserves_request_and_prevents_retry(tmp_path):
    with pytest.raises(TimeoutError):
        entry.request_once(tmp_path, "inert", timeout=0.01)
    assert (tmp_path / "original-python.request.json").is_file()
    with pytest.raises(RuntimeError, match="never resubmit"):
        entry.request_once(tmp_path, "inert")


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"code_sha256": "wrong"}, ValueError),
        ({"status": "INFRASTRUCTURE_FAILED"}, RuntimeError),
        ({"tool_output": None}, ValueError),
    ],
)
def test_wrong_or_failed_response_never_becomes_success(
    tmp_path, monkeypatch, change, error
):
    code = "inert"
    payload = {
        "status": "ORIGINAL_TOOL_RETURNED",
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "tool_output": "fixture",
        **change,
    }

    def reply(_):
        entry.save_new(
            tmp_path / "original-python.response.json", json.dumps(payload).encode()
        )

    monkeypatch.setattr(entry.time, "sleep", reply)
    with pytest.raises(error):
        entry.request_once(tmp_path, code)
