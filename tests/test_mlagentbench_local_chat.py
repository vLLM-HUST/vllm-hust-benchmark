"""Local transport protocol fixtures; no real model requests or Agent runs."""

from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mlagentbench_local_chat import complete_original_chat, original_chat_request

MODEL = "Qwen/Qwen3-14B"


def test_original_prompt_and_default_generation_contract():
    prompt = "original\n完整任务\nObservation from previous step"
    body = original_chat_request(prompt, MODEL)
    assert body == {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.2,
        "stop": ["Observation:"],
    }
    assert (
        original_chat_request(prompt, MODEL, max_tokens_to_sample=4000)["max_tokens"]
        == 4000
    )
    assert original_chat_request(prompt, MODEL, temperature=0.01)["temperature"] == 0.01


def test_private_actual_response_and_usage_saved_without_price(tmp_path):
    calls = []
    result = {
        "choices": [{"message": {"content": "fixture completion"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }

    def opener(request, timeout):
        calls.append((request, timeout))
        return BytesIO(json.dumps(result).encode())

    log = tmp_path / "step.log"
    completion = complete_original_chat(
        "unchanged",
        MODEL,
        api_base="http://127.0.0.1:14001/v1",
        allowed_model=MODEL,
        log_file=log,
        opener=opener,
    )
    assert completion == "fixture completion"
    assert len(calls) == 1
    assert json.loads(calls[0][0].data)["messages"][0]["content"] == "unchanged"
    receipt = json.loads(Path(str(log) + ".local-transport.jsonl").read_text("utf-8"))
    assert receipt["response"]["usage"] == result["usage"]
    assert receipt["monetary_cost"] is None


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://external.example/v1",
        "http://127.0.0.1:14001/v1?key=x",
        "http://name:pass@localhost/v1",  # pragma: allowlist secret - rejected fixture credentials, not a real account
        "http://localhost/v2",
    ],
)
def test_external_or_ambiguous_endpoint_rejected(endpoint):
    with pytest.raises(ValueError, match="loopback"):
        complete_original_chat("x", MODEL, api_base=endpoint, allowed_model=MODEL)


def test_fake_model_alias_rejected_without_request():
    with pytest.raises(ValueError, match="actual model"):
        complete_original_chat(
            "x", "gpt-4", api_base="http://localhost/v1", allowed_model=MODEL
        )


def test_transport_failure_is_not_silently_retried():
    calls = []

    def opener(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, 500, "fixture failure", {}, None)

    with pytest.raises(HTTPError):
        complete_original_chat(
            "x",
            MODEL,
            api_base="http://localhost/v1",
            allowed_model=MODEL,
            opener=opener,
        )
    assert len(calls) == 1


def test_actual_baseline_receipt_binds_original_source_not_agent_delivery():
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (
            root
            / "docs/workload-data/validation/mlagentbench-cifar-original-baseline-20261101.json"
        ).read_text("utf-8")
    )
    pins = json.loads(
        (root / "configs/workload-native/mlagentbench-author-source.json").read_text(
            "utf-8"
        )
    )
    sources = {item["path"]: item["sha256"] for item in pins["files"]}
    for relative, digest in receipt["unchanged_original_task_files_sha256"].items():
        assert sources["MLAgentBench/benchmarks/cifar10/" + relative] == digest
    assert receipt["exit_code"] == 0
    assert receipt["original_epochs"] == 5
    assert receipt["prediction_rows"] == receipt["original_test_images"] == 10000
    assert receipt["native_agent_tasks_run"] == 0
    assert receipt["source_checkbox_eligible"] is False
