"""Mock local transport only; not native NPU or publisher data evidence."""

import importlib.util
import json
from pathlib import Path
import sys

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
