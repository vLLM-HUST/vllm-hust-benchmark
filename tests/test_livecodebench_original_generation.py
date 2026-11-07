"""Inert wire fixtures only; no real model, task execution or scores."""

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "lcb_generation", ROOT / "scripts/run_livecodebench_original_generation.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
worker_spec = importlib.util.spec_from_file_location(
    "lcb_worker", ROOT / "scripts/livecodebench_container_worker.py"
)
worker = importlib.util.module_from_spec(worker_spec)
worker_spec.loader.exec_module(worker)
scoring_spec = importlib.util.spec_from_file_location(
    "lcb_scoring", ROOT / "scripts/run_livecodebench_container_scoring.py"
)
scoring = importlib.util.module_from_spec(scoring_spec)
scoring_spec.loader.exec_module(scoring)


def original_request():
    return {
        "question_id": "inert-original-id",
        "request": {
            "model": "inert-local-model",
            "messages": [{"role": "user", "content": "inert original prompt"}],
            "n": 10,
            "max_tokens": 2000,
            "temperature": 0.2,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
    }


def test_keeps_original_body_and_all_ten_candidates(tmp_path):
    row = original_request()
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([row]))
    calls = []
    reply = {
        "choices": [
            {
                "index": i,
                "message": {"content": "inert output"},
                "finish_reason": "length" if i == 0 else "stop",
            }
            for i in range(10)
        ]
    }

    def transport(endpoint, body, timeout):
        calls.append((endpoint, body, timeout))
        return reply

    output = tmp_path / "output"
    receipt = tool.run(
        prompts, output, endpoint="http://127.0.0.1:19466/v1", transport=transport
    )
    assert len(calls) == 1 and calls[0][1] == row["request"]
    assert calls[0][2] == 90
    saved = json.loads((output / "0.request-response.private.json").read_text())
    assert saved["response"] == reply
    assert receipt["completed_question_ids"] == [row["question_id"]]
    assert not receipt["source_checkbox_eligible"]
    assert not receipt["native_scoring_complete"]


def test_missing_candidates_retained_no_retry_or_replacement(tmp_path):
    row = original_request()
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([row, {**row, "question_id": "second"}]))
    calls = []

    def transport(*args):
        calls.append(args)
        return {"choices": []}

    output = tmp_path / "output"
    with pytest.raises(ValueError, match="n=10"):
        tool.run(
            prompts, output, endpoint="http://localhost:19466/v1", transport=transport
        )
    assert len(calls) == 1
    assert (output / "0.request-response.private.json").is_file()
    receipt = json.loads((output / "receipt-private.json").read_text())
    assert receipt["status"] == "STOP_RETAINED_NEEDS_TRIAGE"
    assert receipt["attempted_requests"] == 1
    assert not receipt["completed_question_ids"]


@pytest.mark.parametrize(
    "field,value", [("n", 1), ("max_tokens", 100), ("temperature", 0.0)]
)
def test_reduced_or_changed_native_budget_rejected(tmp_path, field, value):
    row = original_request()
    row["request"][field] = value
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="defaults"):
        tool.requests_from(path)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.openai.com/v1",
        "http://remote/v1",
        "http://user:password@localhost/v1",  # pragma: allowlist secret - inert rejected URL fixture
    ],
)
def test_paid_or_nonlocal_endpoints_rejected(endpoint):
    with pytest.raises(ValueError, match="loopback"):
        tool.local_endpoint(endpoint)


def test_original_candidate_index_order_restored_for_scoring():
    row = original_request()
    saved = {
        **row,
        "response": {
            "choices": [
                {"index": i, "message": {"content": f"inert candidate {i}"}}
                for i in reversed(range(10))
            ]
        },
    }
    result = worker.outputs_from_saved(
        saved,
        question_id=row["question_id"],
        messages=row["request"]["messages"],
        model=row["request"]["model"],
    )
    assert result == [f"inert candidate {i}" for i in range(10)]


def test_original_prompt_mismatch_blocks_scoring():
    row = original_request()
    with pytest.raises(ValueError, match="prompt or generation budget"):
        worker.outputs_from_saved(
            row,
            question_id=row["question_id"],
            messages=[{"role": "user", "content": "changed"}],
            model=row["request"]["model"],
        )


def test_native_worker_refuses_host_execution(monkeypatch):
    monkeypatch.setattr(worker.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="no host fallback"):
        worker.require_isolated_worker()


def test_network_guard_uses_actual_interfaces_not_sysfs_bonding_control_file():
    table = "Inter-| Receive | Transmit\n face |bytes packets\n lo: 0 0\n"
    assert worker.loopback_only_network(table)
    assert not worker.loopback_only_network(table + "eth0: 0 0\n")
    assert not worker.loopback_only_network("")


def test_scoring_container_uses_offline_nonroot_readonly_isolation():
    name = "hust-lcb-" + "a" * 32
    command = scoring.worker_command("sha256:" + "b" * 64, name, "inert-model")
    assert "--network=none" in command and "--read-only" in command
    assert "--pull=never" in command and "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges:true" in command
    assert command[command.index("--user") + 1] == "65534:65534"
    assert not any(
        "docker.sock" in part or "type=bind" in part or "--privileged" in part
        for part in command
    )


@pytest.mark.parametrize(
    "image,name,uid",
    [
        ("latest", "hust-lcb-" + "a" * 32, 65534),
        ("sha256:" + "b" * 64, "other-owner", 65534),
        ("sha256:" + "b" * 64, "hust-lcb-" + "a" * 32, 0),
    ],
)
def test_unknown_image_owner_or_root_worker_rejected(image, name, uid):
    with pytest.raises(ValueError):
        scoring.worker_command(image, name, "inert-model", uid)


@pytest.mark.skipif(os.name == "nt", reason="POSIX sandbox traversal mode contract")
def test_copied_input_directory_traversal_survives_private_operator_umask(tmp_path):
    previous = os.umask(0o077)
    try:
        path = scoring.staging_directory(tmp_path / "inputs")
        assert path.stat().st_mode & 0o777 == 0o755
    finally:
        os.umask(previous)
