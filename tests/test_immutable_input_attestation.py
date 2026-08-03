from __future__ import annotations

import importlib.util
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


def load_compat_module():
    path = REPO_ROOT / "scripts" / "run_vllm_cli_compat.py"
    spec = importlib.util.spec_from_file_location("run_vllm_cli_compat", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_hash_handles_mapping_bytes_numpy_and_pil() -> None:
    numpy = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    compat = load_compat_module()
    image = image_module.new("RGB", (2, 1), color=(1, 2, 3))
    left = {
        "array": numpy.array([[1, 2]], dtype=numpy.int64),
        "bytes": b"abc",
        "image": image,
    }
    right = {"image": image.copy(), "bytes": bytearray(b"abc"), "array": left["array"]}
    assert compat.canonical_sha256(left) == compat.canonical_sha256(right)


def test_recorder_rejects_same_process_input_drift(tmp_path: Path) -> None:
    compat = load_compat_module()
    recorder = compat.ImmutableInputRecorder(
        tmp_path / "attestation.json",
        {"model_id": "m", "model_revision": "a" * 40, "data_identity": {}},
    )
    recorder.record("latency-prompt-token-ids", [[1, 2]])
    recorder.record("latency-prompt-token-ids", [[1, 2]])
    with pytest.raises(RuntimeError, match="drifted"):
        recorder.record("latency-prompt-token-ids", [[1, 3]])


def test_file_contract_checks_size_and_sha(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "input.jsonl"
    source.write_bytes(b"actual input\n")
    identity = {"kind": "repository-file", "path": source.name, **file_identity(source)}
    assert (
        verify_data_contract(
            identity,
            benchmark_repo=repo,
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
            benchmark_repo=repo,
            vllm_worktree=tmp_path,
            dataset_root=tmp_path,
            sharegpt_url="",
        )


def test_random_latency_generator_requires_source_hash_not_seed(
    tmp_path: Path,
) -> None:
    generator = tmp_path / "latency.py"
    generator.write_text("tokens = np.random.randint(1, 10)\n", encoding="utf-8")
    identity = {
        "kind": "nondeterministic-vllm-generator",
        "generator_path": "latency.py",
        "generator_sha256": file_identity(generator)["sha256"],
        "seed": None,
    }
    verify_data_contract(
        identity,
        benchmark_repo=tmp_path,
        vllm_worktree=tmp_path,
        dataset_root=tmp_path,
        sharegpt_url="",
    )


def test_trace_attestation_uses_actual_summary_hash(tmp_path: Path) -> None:
    output = tmp_path / "immutable-input-attestation.json"
    metadata = {
        "model_id": "model",
        "model_revision": "b" * 40,
        "data_identity": {"kind": "release-asset"},
    }
    write_trace_attestation(output, metadata, {"selected_requests_sha256": "c" * 64})
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "immutable-input-attestation/v1"
    assert payload["resolved_input_sha256"] == "c" * 64


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
