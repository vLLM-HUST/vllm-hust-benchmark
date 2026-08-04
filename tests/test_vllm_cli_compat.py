import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts.run_vllm_cli_compat import (
    _latency_prompt_token_ids,
    _sample_request_payload,
    offline_graph_proof,
    record_immutable_inputs,
    require_offline_graph,
    resolved_input_sha256,
)


@dataclass
class _FakeImage:
    prompt: str
    pixels: bytes


def test_resolved_input_hash_is_canonical_and_content_sensitive() -> None:
    first = resolved_input_sha256(
        input_kind="prompt-token-ids",
        inputs=[{"prompt_token_ids": [1, 2, 3], "meta": {"b": 2, "a": 1}}],
    )
    reordered = resolved_input_sha256(
        input_kind="prompt-token-ids",
        inputs=[{"meta": {"a": 1, "b": 2}, "prompt_token_ids": [1, 2, 3]}],
    )
    changed = resolved_input_sha256(
        input_kind="prompt-token-ids",
        inputs=[{"prompt_token_ids": [1, 2, 4], "meta": {"a": 1, "b": 2}}],
    )

    assert first == reordered
    assert first != changed


def test_record_immutable_inputs_writes_exact_contract(tmp_path, monkeypatch) -> None:
    output = tmp_path / "immutable-input-attestation.json"
    data_identity = {
        "kind": "nondeterministic-vllm-generator",
        "requires_repeat_input_sha256": True,
    }
    monkeypatch.setenv("VLLM_HUST_IMMUTABLE_INPUT_ATTESTATION_FILE", str(output))
    monkeypatch.setenv("VLLM_HUST_MODEL_ID", "Qwen/model")
    monkeypatch.setenv("VLLM_HUST_MODEL_REVISION", "a" * 40)
    monkeypatch.setenv("VLLM_HUST_DATA_IDENTITY_JSON", json.dumps(data_identity))

    inputs = [{"prompt_token_ids": [1, 2, 3]}]
    record_immutable_inputs(input_kind="prompt-token-ids", inputs=inputs)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "immutable-input-attestation/v1"
    assert payload["model_id"] == "Qwen/model"
    assert payload["model_revision"] == "a" * 40
    assert payload["data_identity"] == data_identity
    assert payload["resolved_input_kind"] == "prompt-token-ids"
    assert payload["resolved_input_sha256"] == resolved_input_sha256(
        input_kind="prompt-token-ids", inputs=inputs
    )

    record_immutable_inputs(input_kind="prompt-token-ids", inputs=inputs)
    with pytest.raises(RuntimeError, match="changed within one run"):
        record_immutable_inputs(
            input_kind="prompt-token-ids",
            inputs=[{"prompt_token_ids": [9]}],
        )


def test_latency_capture_hashes_actual_generated_token_ids() -> None:
    first = [{"prompt_token_ids": [10, 20]}, {"prompt_token_ids": [30, 40]}]
    second = [{"prompt_token_ids": [10, 20]}, {"prompt_token_ids": [30, 41]}]
    first_tokens = _latency_prompt_token_ids(first)
    assert first_tokens == [[10, 20], [30, 40]]
    assert resolved_input_sha256(
        input_kind="latency-prompt-token-ids", inputs=first_tokens
    ) != resolved_input_sha256(
        input_kind="latency-prompt-token-ids",
        inputs=_latency_prompt_token_ids(second),
    )


def test_serve_sample_request_sequence_is_stable_and_order_sensitive() -> None:
    requests = [
        SimpleNamespace(
            prompt="alpha", prompt_len=1, expected_output_len=2, request_id="0"
        ),
        SimpleNamespace(
            prompt="beta", prompt_len=1, expected_output_len=3, request_id="1"
        ),
    ]
    payloads = [_sample_request_payload(request) for request in requests]
    repeated = [_sample_request_payload(request) for request in requests]
    assert payloads == repeated
    assert resolved_input_sha256(
        input_kind="serve-sample-requests", inputs=payloads
    ) == resolved_input_sha256(input_kind="serve-sample-requests", inputs=repeated)
    assert resolved_input_sha256(
        input_kind="serve-sample-requests", inputs=payloads
    ) != resolved_input_sha256(
        input_kind="serve-sample-requests", inputs=list(reversed(payloads))
    )


def test_capture_rejects_input_kind_mismatch_before_writing(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "immutable-input-attestation.json"
    metadata = {
        "model_id": "Qwen/model",
        "model_revision": "a" * 40,
        "data_identity": {"kind": "deterministic-vllm-generator", "seed": 0},
        "resolved_input_kind": "serve-sample-requests",
    }
    monkeypatch.setenv("VLLM_HUST_IMMUTABLE_INPUT_ATTESTATION_FILE", str(output))
    monkeypatch.setenv("VLLM_HUST_IMMUTABLE_INPUT_METADATA", json.dumps(metadata))
    with pytest.raises(RuntimeError, match="does not match the official spec"):
        record_immutable_inputs(
            input_kind="latency-prompt-token-ids", inputs=[[1, 2, 3]]
        )
    assert not output.exists()


def _fake_llm(*, enforce_eager: bool, mode: str, cudagraph_mode: str) -> object:
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        compilation_config=SimpleNamespace(
            mode=SimpleNamespace(name=mode),
            cudagraph_mode=SimpleNamespace(name=cudagraph_mode),
        ),
    )
    return SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=config))


def test_offline_graph_proof_accepts_effective_piecewise_graph() -> None:
    proof = offline_graph_proof(
        _fake_llm(
            enforce_eager=False,
            mode="VLLM_COMPILE",
            cudagraph_mode="PIECEWISE",
        )
    )

    require_offline_graph(proof)
    assert proof["graph_mode_verified"] is True


@pytest.mark.parametrize(
    ("enforce_eager", "mode", "cudagraph_mode"),
    [
        (True, "NONE", "NONE"),
        (False, "NONE", "PIECEWISE"),
        (False, "VLLM_COMPILE", "NONE"),
    ],
)
def test_offline_graph_proof_rejects_eager_or_disabled_graph(
    enforce_eager: bool,
    mode: str,
    cudagraph_mode: str,
) -> None:
    proof = offline_graph_proof(
        _fake_llm(
            enforce_eager=enforce_eager,
            mode=mode,
            cudagraph_mode=cudagraph_mode,
        )
    )

    with pytest.raises(RuntimeError, match="eager/non-graph"):
        require_offline_graph(proof)
