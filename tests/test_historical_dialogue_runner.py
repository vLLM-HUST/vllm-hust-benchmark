"""Mock local transport only; not native NPU or publisher data evidence."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "workload-preparation/src"))
SPEC = importlib.util.spec_from_file_location(
    "historical_runner", ROOT / "scripts/run_historical_dialogue_original.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def original():
    from vllm_hust_workload_data.records import WorkloadRecord

    return WorkloadRecord(
        source={"source_id": "WILDCHAT", "file": "fixture.jsonl"},
        task={
            "task_id": "fixture",
            "input": {
                "conversation": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "held out"},
                    {"role": "user", "content": "follow up"},
                ]
            },
        },
        runtime={"kind": "historical_dialogue"},
        replay={"source_row_index": 0},
    )


def invoke(tmp_path, transport, **kwargs):
    return RUNNER.run_records(
        [original()],
        output=tmp_path / "output",
        endpoint="http://127.0.0.1:19466/v1",
        model="fixture-local-model",
        max_tokens=32,
        context=128,
        transport=transport,
        **kwargs,
    )


def response():
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": "new"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
    }


def test_reasoning_only_length_is_preserved_but_empty_final_stop_is_not_success():
    choice = {
        "finish_reason": "length",
        "message": {
            "content": None,
            "reasoning_content": "unfinished synthetic reasoning",
        },
    }
    assert RUNNER.has_generated_content(choice)
    choice["finish_reason"] = "stop"
    assert not RUNNER.has_generated_content(choice)
    choice["finish_reason"] = "length"
    choice["message"]["reasoning_content"] = ""
    assert not RUNNER.has_generated_content(choice)


def test_explicit_timeout_does_not_change_native_requests_or_budget(tmp_path):
    calls = []

    def transport(url, payload):
        calls.append(payload)
        return {"count": 10} if url.endswith("/tokenize") else response()

    receipt = invoke(tmp_path, transport, request_timeout_seconds=1200)
    assert receipt["client_timeout_seconds"] == 1200
    assert receipt["max_tokens"] == 32
    assert all(payload.get("max_tokens", 32) == 32 for payload in calls)
    assert receipt["quality_scored"] is False


@pytest.mark.parametrize("key", ["reasoning", "reasoning_content"])
def test_legal_reasoning_budget_end_keeps_raw_response_and_quality_unverified(
    tmp_path, key
):
    raw = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": None, key: "synthetic incomplete reasoning"},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 32},
    }

    def transport(url, payload):
        return {"count": 10} if url.endswith("/tokenize") else raw

    receipt = invoke(tmp_path, transport)
    assert all(
        item["status"] == "BUDGET_EXHAUSTED_QUALITY_UNVERIFIED"
        for item in receipt["requests"]
    )
    saved = json.loads(
        (tmp_path / "output/request-000000.private.json").read_text("utf-8")
    )
    assert saved["response"] == raw
    assert receipt["quality_scored"] is False


def test_rejected_original_row_is_saved_before_any_model_call(tmp_path):
    item = original()
    item.task["input"]["conversation"][-1]["role"] = "tool"
    calls = []
    with pytest.raises(ValueError, match="unsupported"):
        RUNNER.run_records(
            [item],
            output=tmp_path / "output",
            endpoint="http://127.0.0.1:19466/v1",
            model="fixture",
            max_tokens=32,
            context=128,
            transport=lambda *args: calls.append(args),
        )
    assert calls == []
    saved = json.loads(
        (tmp_path / "output/conversation-000000.private.json").read_text("utf-8")
    )
    assert saved["task"]["input"]["conversation"][-1]["role"] == "tool"


def test_all_original_requests_without_target_leakage(tmp_path):
    calls = []

    def transport(url, body):
        calls.append((url, body))
        return {"count": 10} if url.endswith("/tokenize") else response()

    result = invoke(tmp_path, transport, expected_count=1)
    requests = [body for url, body in calls if url.endswith("/chat/completions")]
    assert len(requests) == 2
    assert requests[0]["messages"] == [{"role": "user", "content": "question"}]
    assert requests[1]["messages"][1]["content"] == "held out"
    assert result["conversations"] == 1
    assert not result["source_checkbox_eligible"]
    assert not result["quality_scored"]


def test_arrow_datetime_annotations_use_existing_portable_form_not_api_fields(tmp_path):
    item = original()
    stamp = datetime(2026, 1, 2, 3, 4, 5, 123456, timezone.utc)
    item.replay["annotations"] = {"timestamp": stamp}
    item.task["input"]["conversation"][0]["timestamp"] = stamp
    item.task["input"]["conversation"][1]["timestamp"] = stamp
    calls = []

    def transport(url, body):
        calls.append((url, body))
        return {"count": 10} if url.endswith("/tokenize") else response()

    result = RUNNER.run_records(
        [item],
        output=tmp_path / "dated",
        endpoint="http://127.0.0.1:19466/v1",
        model="fixture",
        max_tokens=32,
        context=128,
        transport=transport,
        expected_count=1,
    )
    assert result["status"] == "ORIGINAL_HISTORY_GENERATED_NO_QUALITY_SCORE"
    saved = json.loads(
        (tmp_path / "dated/conversation-000000.private.json").read_text("utf-8")
    )
    assert saved["replay"]["annotations"]["timestamp"] == stamp.isoformat()
    assert saved["task"]["input"]["conversation"][1]["timestamp"] == stamp.isoformat()
    request = json.loads(
        (tmp_path / "dated/request-000000.private.json").read_text("utf-8")
    )
    assert request["view"]["non_api_prefix_metadata"] == [
        {"timestamp": stamp.isoformat()}
    ]
    assert (
        request["view"]["reference"]["native_message"]["timestamp"] == stamp.isoformat()
    )
    assert all(
        "timestamp" not in turn
        for _, body in calls
        for turn in body.get("messages", [])
    )
    assert item.replay["annotations"]["timestamp"] is stamp


@pytest.mark.parametrize(
    "failure", ["context", "response", "transport", "final-identity", "count"]
)
def test_failure_is_retained_without_retry_or_replacement(tmp_path, failure):
    calls = []

    def transport(url, body):
        calls.append(url)
        if failure == "transport":
            raise RuntimeError("fixture failure")
        if url.endswith("/tokenize"):
            return {"count": 120 if failure == "context" else 10}
        result = response()
        if failure == "response":
            result["choices"] = []
        return result

    def verify():
        if failure == "final-identity":
            raise ValueError("fixture changed asset")

    with pytest.raises((ValueError, RuntimeError)):
        invoke(
            tmp_path,
            transport,
            final_verify=verify,
            expected_count=2 if failure == "count" else 1,
        )
    receipt = json.loads((tmp_path / "output/receipt-private.json").read_text("utf-8"))
    assert receipt["status"] == "FAILED_NOT_COMPLETE"
    assert not receipt["source_checkbox_eligible"]
    if failure in {"context", "transport"}:
        assert len(calls) == 1
    if failure == "response":
        saved = json.loads(
            (tmp_path / "output/request-000000.private.json").read_text("utf-8")
        )
        assert saved["response"]["choices"] == []
        assert saved["status"] == "FAILED_NOT_REPLACED"


@pytest.mark.parametrize(
    "sid,field,id_field",
    [
        ("WILDCHAT", "conversation", "conversation_hash"),
        ("ULTRACHAT", "messages", "prompt_id"),
        ("LMSYS_CHAT", "conversation", "conversation_id"),
    ],
)
@pytest.mark.parametrize("mode", ["sample", "full", "changed-file", "changed-revision"])
def test_cli_uses_actual_original_adapter_and_identity_gate(
    tmp_path, monkeypatch, capsys, sid, field, id_field, mode
):
    from copy import deepcopy
    from vllm_hust_workload_data import registry

    data = tmp_path / "data"
    data.mkdir()
    rows = [
        {field: original().task["input"]["conversation"], id_field: str(i)}
        for i in range(4)
    ]
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (data / "fixture.jsonl").write_bytes(raw)
    upstream = deepcopy(registry.source_spec(sid))
    upstream["files"] = [
        {
            "path": "fixture.jsonl",
            "role": "raw_source",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    ]
    manifest = deepcopy(upstream)
    manifest.update(source_id=sid, upstream_revision_verified=True)
    manifest["files"][0].update(role="data", rows=4)
    if mode == "changed-revision":
        manifest["revision"] = "unreviewed revision"
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "source-manifest.private.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    if mode == "changed-file":
        (data / "fixture.jsonl").write_bytes(raw + b"\n")
    monkeypatch.setattr(registry, "source_spec", lambda source_id: upstream)
    actual_run = RUNNER.run_records
    calls = []

    def transport(url, body):
        calls.append(url)
        return {"count": 10} if url.endswith("/tokenize") else response()

    def with_transport(*args, **kwargs):
        return actual_run(*args, **kwargs, transport=transport)

    monkeypatch.setattr(RUNNER, "run_records", with_transport)
    argv = [
        "runner",
        sid,
        "--prepared",
        str(prepared),
        "--data-root",
        str(data),
        "--output",
        str(tmp_path / "out"),
        "--model",
        "fixture-local",
        "--max-tokens",
        "32",
        "--context-capacity",
        "128",
    ]
    if mode == "full":
        argv.append("--full")
    monkeypatch.setattr(sys, "argv", argv)
    if mode.startswith("changed"):
        with pytest.raises(ValueError, match="identity|manifest"):
            RUNNER.main()
        assert calls == []
        return
    RUNNER.main()
    result = json.loads(capsys.readouterr().out)
    expected = 4 if mode == "full" else 3
    assert result["conversations"] == expected
    assert result["requests"] == expected * 2
    assert not result["source_checkbox_eligible"]
