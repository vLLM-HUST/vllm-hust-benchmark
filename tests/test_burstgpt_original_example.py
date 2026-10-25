"""Synthetic software fixtures only; no data download, NPU or performance tests."""

import importlib.util
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "burstgpt_runner", ROOT / "scripts/run_burstgpt_original_example.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def original_body():
    return {
        "prompt": "synthetic software fixture",
        "stream": False,
        "ignore_eos": False,
        "max_tokens": 1024,
        "temperature": 0,
    }


def response():
    return {
        "choices": [{"text": " fixture suffix", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


def test_native_transport_preserves_raw_prompt_and_original_budget():
    body = original_body()
    routed = runner.original_request(body, "fixture-model")
    assert routed == {**body, "model": "fixture-model"}
    assert "model" not in body
    assert "messages" not in routed
    assert runner.legacy_response(body, response()) == {
        "text": [body["prompt"] + " fixture suffix"]
    }


def test_legal_empty_stop_is_not_replaced_or_called_transport_failure():
    result = response()
    result["choices"][0]["text"] = ""
    result["usage"]["completion_tokens"] = 0
    assert runner.legacy_response(original_body(), result) == {
        "text": [original_body()["prompt"]]
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tokens", 128),
        ("max_tokens", True),
        ("stream", True),
        ("ignore_eos", True),
        ("temperature", 1),
        ("prompt", ""),
    ],
)
def test_changed_official_profile_rejected(field, value):
    body = original_body()
    body[field] = value
    with pytest.raises(ValueError):
        runner.original_request(body, "fixture")


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1",
        "http://localhost/",
        "http://localhost/v1?key=x",
        "http://localhost/v1#fragment",
        "http://user:pass@localhost/v1",  # pragma: allowlist secret - rejected synthetic credential fixture
    ],
)
def test_endpoint_fails_closed(url):
    with pytest.raises(ValueError):
        runner.endpoint(url)


@pytest.mark.parametrize(
    "count,reason", [(-1, "stop"), (1025, "length"), (3, "error"), (True, "stop")]
)
def test_bad_native_response_is_not_a_success(count, reason):
    result = response()
    result["usage"]["completion_tokens"] = count
    result["choices"][0]["finish_reason"] = reason
    with pytest.raises(ValueError):
        runner.legacy_response(original_body(), result)


def test_loopback_bridge_roundtrip_is_not_a_model_run(tmp_path, monkeypatch):
    routed = []

    def fake_native(request, timeout):
        assert request.full_url == "http://127.0.0.1:19466/v1/completions"
        assert timeout == 4 * 60 * 60
        routed.append(json.loads(request.data))
        return io.BytesIO(json.dumps(response()).encode())

    monkeypatch.setattr(runner, "urlopen", fake_native)
    body = original_body()
    with runner.bridge("http://127.0.0.1:19466/v1", "fixture-model", tmp_path) as (
        port,
        records,
        errors,
    ):
        request = Request(
            f"http://127.0.0.1:{port}/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as result:
            assert json.load(result) == {"text": [body["prompt"] + " fixture suffix"]}
        assert not errors
        assert len(records) == 1
        assert records[0]["completion_tokens"] == 3
    assert routed == [{**body, "model": "fixture-model"}]
    saved = json.loads((tmp_path / "request-0.private.json").read_text())
    assert saved["legacy_request"] == body
    assert saved["native_response"] == response()


def test_pinned_source_identity_covers_complete_example_code():
    assert len(runner.SOURCE_BLOBS) == 18
    assert "example/profile_vllm_server.py" in runner.SOURCE_BLOBS
    assert "example/framework_inference_call/framework_vllm.py" in runner.SOURCE_BLOBS
    assert set(runner.TRACE_SHA) == set(runner.TRACE_ROWS)
    assert sum(runner.TRACE_ROWS.values()) == 10632194


def test_preparation_is_external_prompt_asset_not_native_completion():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/burstgpt-official-example-preparation-20261025.json"
        ).read_text("utf-8")
    )
    assert receipt["external_prompt_pool_bytes"] == 6473720
    assert receipt["external_prompt_pool_rows"] == 300
    assert receipt["publisher_git_blob_identity_equal"]
    assert not receipt["external_prompt_pool_is_burstgpt_original_private_requests"]
    assert receipt["fixed_trace_rows"] == 60
    assert receipt["official_output_budget"] == 1024
    assert receipt["all_selected_arrivals_nonnegative_monotonic"]
    assert receipt["model_calls"] == 0
    assert not receipt["official_demo_executed"]
    assert not receipt["source_completion_checked"]


def test_delivery_combines_native_results_and_fixed_consumer_without_relabeling_trace():
    docs = ROOT / "docs/workload-data"
    ledger = json.loads((docs / "dataset-validation-status.json").read_text("utf-8"))
    row = ledger["delivery_inventory"]["BURSTGPT"]
    receipt = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    assert row["checked"] and row["status"] == "AVAILABLE"
    assert receipt["source_checkbox_eligible"]
    native = [
        json.loads((ROOT / path).read_text("utf-8"))
        for path in receipt["native_receipts"]
    ]
    # Release 2 is embedded in the frozen intermediate progress snapshot.
    native[1] = native[1]["release2_audit"]
    assert [item["scale"] for item in native] == [1, 1000000, 1000000]
    assert sum(item["native_model_calls_reused"] for item in native) == 60
    assert all(item["original_indices"] == list(range(20)) for item in native)
    assert all(item["original_output_budget"] == 1024 for item in native)
    fixed = receipt["fixed_consumer_audit"]
    assert fixed["provider_installed_payload_files_record_hash_equal"] == 33
    assert fixed["research_installed_python_files_record_hash_equal"] == 40
    assert (
        fixed["public_direct_rows"]
        == fixed["public_index_resolved_rows"]
        == fixed["research_index_resolved_rows"]
        == 9
    )
    assert (
        fixed["inputs_and_outcomes_equal"] and fixed["metadata_request_body_rejected"]
    )
    assert not fixed["research_default_lock_updated"]
    assert receipt["new_model_calls"] == 0
    assert not receipt["original_timing_performance_claim"]
    assert not receipt["external_prompt_pool_is_original_private_user_requests"]
    assert not receipt["raw_payloads_published"]
