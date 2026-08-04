from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.immutable_input_attestation import (
    build_metadata,
    expected_resolved_input_kind,
    file_identity,
    resolve_sharegpt_dataset_url,
    resolved_input_sha256,
    validate_attestation_payload,
    verify_data_contract,
    write_trace_attestation,
    write_attestation_atomic,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SHAREGPT_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
SHAREGPT_IDENTITY = {
    "kind": "huggingface-file",
    "repository": "anon8231489123/ShareGPT_Vicuna_unfiltered",
    "revision": SHAREGPT_REVISION,
    "path": "ShareGPT_V3_unfiltered_cleaned_split.json",
}
SHAREGPT_EXACT_URL = (
    "https://hf-mirror.com/datasets/"
    "anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/"
    f"{SHAREGPT_REVISION}/ShareGPT_V3_unfiltered_cleaned_split.json"
)


def test_file_contract_checks_size_and_sha(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    source.write_bytes(b"actual input\n")
    identity = {"kind": "repository-file", "path": source.name, **file_identity(source)}
    assert (
        verify_data_contract(
            identity,
            benchmark_repo=tmp_path,
            vllm_worktree=tmp_path,
            dataset_root=tmp_path,
            sharegpt_url="",
        )
        == identity
    )
    source.write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="mismatch"):
        verify_data_contract(
            identity,
            benchmark_repo=tmp_path,
            vllm_worktree=tmp_path,
            dataset_root=tmp_path,
            sharegpt_url="",
        )


def test_random_latency_contract_pins_generator_without_claiming_seed(
    tmp_path: Path,
) -> None:
    generator = tmp_path / "latency.py"
    generator.write_text("tokens = np.random.randint(1, 10)\n", encoding="utf-8")
    identity = {
        "kind": "nondeterministic-vllm-generator",
        "generator_path": generator.name,
        "generator_sha256": file_identity(generator)["sha256"],
        "requires_repeat_input_sha256": True,
    }
    verify_data_contract(
        identity,
        benchmark_repo=tmp_path,
        vllm_worktree=tmp_path,
        dataset_root=tmp_path,
        sharegpt_url="",
    )


def test_trace_attestation_requires_actual_token_id_hash(tmp_path: Path) -> None:
    output = tmp_path / "immutable-input-attestation.json"
    metadata = {
        "model_id": "model",
        "model_revision": "b" * 40,
        "data_identity": {"kind": "release-asset"},
    }
    summary = {
        "resolved_input_kind": "production-trace-prompt-token-ids",
        "resolved_input_sha256": "c" * 64,
    }
    write_trace_attestation(output, metadata, summary)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["resolved_input_sha256"] == "c" * 64
    with pytest.raises(ValueError, match="actual resolved_input_sha256"):
        write_trace_attestation(
            output, metadata, {"selected_requests_sha256": "d" * 64}
        )


def test_build_metadata_requires_exact_revision_and_data_identity() -> None:
    with pytest.raises(ValueError, match="model_revision"):
        build_metadata({"model": "model", "data_identity": {}})
    with pytest.raises(ValueError, match="data_identity"):
        build_metadata({"model": "model", "model_revision": "d" * 40})


@pytest.mark.parametrize(
    ("scenario", "kind", "expected"),
    [
        (
            "random-latency",
            "nondeterministic-vllm-generator",
            "latency-prompt-token-ids",
        ),
        ("random-online", "deterministic-vllm-generator", "serve-sample-requests"),
        ("sharegpt-online", "huggingface-file", "serve-sample-requests"),
    ],
)
def test_expected_input_kind_matches_b_targets(
    scenario: str, kind: str, expected: str
) -> None:
    assert (
        expected_resolved_input_kind(
            {"scenario": scenario, "data_identity": {"kind": kind}}
        )
        == expected
    )


def test_attestation_validation_rejects_spec_inference_or_drift() -> None:
    metadata = {
        "model_id": "Qwen/model",
        "model_revision": "a" * 40,
        "data_identity": {"kind": "deterministic-vllm-generator", "seed": 0},
        "resolved_input_kind": "serve-sample-requests",
    }
    payload = {
        "schema_version": "immutable-input-attestation/v1",
        **metadata,
        "resolved_inputs": [{"prompt": "captured"}],
    }
    payload["resolved_input_sha256"] = resolved_input_sha256(
        input_kind="serve-sample-requests", inputs=payload["resolved_inputs"]
    )
    validate_attestation_payload(payload, metadata)
    for field, value in (
        ("model_revision", "c" * 40),
        ("data_identity", {"kind": "spec-placeholder"}),
        ("resolved_input_kind", "latency-prompt-token-ids"),
        ("resolved_input_sha256", ""),
    ):
        changed = dict(payload)
        changed[field] = value
        with pytest.raises(ValueError, match="immutable input attestation"):
            validate_attestation_payload(changed, metadata)

    changed = dict(payload)
    changed["resolved_inputs"] = [{"prompt": "tampered"}]
    with pytest.raises(ValueError, match="resolved input SHA256 mismatch"):
        validate_attestation_payload(changed, metadata)


def test_atomic_attestation_failure_leaves_no_final_or_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "immutable-input-attestation.json"

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(
        "vllm_hust_benchmark.immutable_input_attestation.os.replace", fail_replace
    )
    with pytest.raises(OSError, match="atomic replace failure"):
        write_attestation_atomic(output, {"captured": True})

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_sharegpt_url_defaults_to_spec_exact_revision() -> None:
    assert resolve_sharegpt_dataset_url(SHAREGPT_IDENTITY, "") == SHAREGPT_EXACT_URL


def test_sharegpt_url_rejects_main_override() -> None:
    main_url = SHAREGPT_EXACT_URL.replace(
        f"resolve/{SHAREGPT_REVISION}", "resolve/main"
    )
    with pytest.raises(ValueError, match="not the exact revision URL"):
        resolve_sharegpt_dataset_url(SHAREGPT_IDENTITY, main_url)


def test_sharegpt_url_rejects_revision_drift() -> None:
    drifted_url = SHAREGPT_EXACT_URL.replace(SHAREGPT_REVISION, "a" * 40)
    with pytest.raises(ValueError, match="not the exact revision URL"):
        resolve_sharegpt_dataset_url(SHAREGPT_IDENTITY, drifted_url)


def test_sharegpt_url_does_not_change_non_sharegpt_contract() -> None:
    explicit_url = "https://example.test/not-used.json"
    assert (
        resolve_sharegpt_dataset_url(
            {"kind": "deterministic-vllm-generator"}, explicit_url
        )
        == explicit_url
    )


def test_official_runner_wires_real_input_attestation() -> None:
    runner = (REPO_ROOT / "scripts" / "run-official-ascend-goal-baseline.sh").read_text(
        encoding="utf-8"
    )
    assert "verify_immutable_input_contract" in runner
    assert "VLLM_HUST_IMMUTABLE_INPUT_ATTESTATION_FILE" in runner
    assert "IMMUTABLE_INPUT_METADATA=$(verify_immutable_input_contract)" in runner
    assert "finalize_trace_immutable_input_attestation" in runner
    assert "verify_resolved_input_attestation" in runner
    assert "resolve_sharegpt_dataset_url" in runner
    assert "/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" not in runner
