from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.model_artifact import ModelArtifactError
from vllm_hust_benchmark.model_artifact import verify_local_hf_model


REVISION = "a" * 40


def _write(root: Path, relative: str, payload: bytes, *, revision: str = REVISION) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    metadata = root / ".cache" / "huggingface" / "download" / f"{relative}.metadata"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    object_digest = hashlib.sha256(payload).hexdigest() if relative.endswith(".safetensors") else "b" * 40
    metadata.write_text(f"{revision}\n{object_digest}\n1234.5\n", encoding="utf-8")


def _complete_model(root: Path) -> None:
    _write(root, "config.json", b"{}")
    _write(root, "tokenizer_config.json", b"{}")
    _write(root, "tokenizer.json", b"tokenizer")
    _write(root, "model-00001-of-00002.safetensors", b"first shard")
    _write(root, "model-00002-of-00002.safetensors", b"second shard")
    index = {
        "metadata": {"total_size": 23},
        "weight_map": {
            "layer.0": "model-00001-of-00002.safetensors",
            "layer.1": "model-00002-of-00002.safetensors",
        },
    }
    _write(
        root,
        "model.safetensors.index.json",
        json.dumps(index, sort_keys=True).encode(),
    )


def test_verify_rejects_missing_index_shard_even_when_another_exists(tmp_path: Path) -> None:
    _complete_model(tmp_path)
    (tmp_path / "model-00002-of-00002.safetensors").unlink()

    with pytest.raises(ModelArtifactError, match="missing weight shard.*00002"):
        verify_local_hf_model(tmp_path, REVISION)


def test_verify_rejects_wrong_revision(tmp_path: Path) -> None:
    _complete_model(tmp_path)
    metadata = tmp_path / ".cache/huggingface/download/config.json.metadata"
    metadata.write_text(f"{'c' * 40}\n{'b' * 40}\n1234.5\n", encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="revision mismatch for config.json"):
        verify_local_hf_model(tmp_path, REVISION)


def test_verify_rejects_corrupted_weight_shard(tmp_path: Path) -> None:
    _complete_model(tmp_path)
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"corrupt")

    with pytest.raises(ModelArtifactError, match="weight SHA256 mismatch.*00001"):
        verify_local_hf_model(tmp_path, REVISION)


def test_verify_returns_stable_canonical_manifest_and_digest(tmp_path: Path) -> None:
    _complete_model(tmp_path)

    first = verify_local_hf_model(tmp_path, REVISION)
    second = verify_local_hf_model(tmp_path, REVISION.upper())

    assert first == second
    assert first["manifest"]["revision"] == REVISION
    assert [item["path"] for item in first["manifest"]["files"]] == sorted(
        item["path"] for item in first["manifest"]["files"]
    )
    assert len(first["model_artifact_digest"]) == 64


@pytest.mark.parametrize("missing", ["config.json", "tokenizer_config.json", "tokenizer.json"])
def test_verify_rejects_missing_config_or_tokenizer(tmp_path: Path, missing: str) -> None:
    _complete_model(tmp_path)
    (tmp_path / missing).unlink()

    with pytest.raises(ModelArtifactError, match="missing required|missing tokenizer"):
        verify_local_hf_model(tmp_path, REVISION)


def test_verify_rejects_any_incomplete_download(tmp_path: Path) -> None:
    _complete_model(tmp_path)
    partial = tmp_path / ".cache/huggingface/download/model.safetensors.incomplete"
    partial.write_bytes(b"partial")

    with pytest.raises(ModelArtifactError, match="incomplete Hugging Face downloads"):
        verify_local_hf_model(tmp_path, REVISION)
