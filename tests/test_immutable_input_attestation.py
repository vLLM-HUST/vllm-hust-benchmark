from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.immutable_input_attestation import (
    build_metadata,
    file_identity,
    verify_data_contract,
    write_trace_attestation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_official_runner_wires_real_input_attestation() -> None:
    runner = (REPO_ROOT / "scripts" / "run-official-ascend-goal-baseline.sh").read_text(
        encoding="utf-8"
    )
    assert "verify_immutable_input_contract" in runner
    assert "VLLM_HUST_IMMUTABLE_INPUT_ATTESTATION_FILE" in runner
    assert "IMMUTABLE_INPUT_METADATA=$(verify_immutable_input_contract)" in runner
    assert "finalize_trace_immutable_input_attestation" in runner
