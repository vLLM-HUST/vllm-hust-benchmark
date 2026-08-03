"""Fail-closed integrity checks for locally downloaded Hugging Face models."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKENIZER_PAYLOADS = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
)


class ModelArtifactError(ValueError):
    """The local model cannot be proven complete and revision-pinned."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str, *, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ModelArtifactError(f"unsafe {field} path: {value!r}")
    return path


def _metadata_for(local_dir: Path, relative_path: PurePosixPath) -> tuple[str, str]:
    metadata_path = (
        local_dir / ".cache" / "huggingface" / "download" / relative_path
    ).with_name(relative_path.name + ".metadata")
    if not metadata_path.is_file():
        raise ModelArtifactError(f"missing Hugging Face metadata: {relative_path}")
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ModelArtifactError(f"invalid Hugging Face metadata: {relative_path}")
    return lines[0].strip().lower(), lines[1].strip().lower()


def _required_paths(local_dir: Path) -> tuple[list[PurePosixPath], set[PurePosixPath]]:
    required = [PurePosixPath("config.json")]
    if not (local_dir / "config.json").is_file():
        raise ModelArtifactError("missing required model file: config.json")

    tokenizer_config = local_dir / "tokenizer_config.json"
    if not tokenizer_config.is_file():
        raise ModelArtifactError(
            "missing required tokenizer file: tokenizer_config.json"
        )
    required.append(PurePosixPath("tokenizer_config.json"))
    tokenizer_payload = next(
        (name for name in _TOKENIZER_PAYLOADS if (local_dir / name).is_file()), None
    )
    if tokenizer_payload is None:
        expected = ", ".join(_TOKENIZER_PAYLOADS)
        raise ModelArtifactError(
            f"missing tokenizer payload (expected one of: {expected})"
        )
    required.append(PurePosixPath(tokenizer_payload))

    index_paths = sorted(local_dir.glob("*.safetensors.index.json"))
    weight_paths: set[PurePosixPath] = set()
    if index_paths:
        for index_path in index_paths:
            relative_index = PurePosixPath(index_path.name)
            required.append(relative_index)
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ModelArtifactError(
                    f"invalid safetensors index: {index_path.name}"
                ) from error
            weight_map = (
                payload.get("weight_map") if isinstance(payload, dict) else None
            )
            if not isinstance(weight_map, dict) or not weight_map:
                raise ModelArtifactError(
                    f"missing non-empty weight_map in safetensors index: {index_path.name}"
                )
            for shard in weight_map.values():
                if not isinstance(shard, str):
                    raise ModelArtifactError(
                        f"invalid shard path in safetensors index: {index_path.name}"
                    )
                relative_shard = _safe_relative_path(shard, field="weight shard")
                shard_path = local_dir.joinpath(*relative_shard.parts)
                if not shard_path.is_file():
                    raise ModelArtifactError(f"missing weight shard: {relative_shard}")
                weight_paths.add(relative_shard)
    else:
        weight_paths = {
            PurePosixPath(path.relative_to(local_dir).as_posix())
            for path in local_dir.glob("*.safetensors")
            if path.is_file()
        }
        if not weight_paths:
            raise ModelArtifactError("missing safetensors weights or safetensors index")

    required.extend(sorted(weight_paths, key=str))
    return sorted(set(required), key=str), weight_paths


def verify_local_hf_model(
    local_dir: str | Path, expected_revision: str
) -> dict[str, Any]:
    """Verify a local HF model and return a deterministic content manifest.

    The returned ``model_artifact_digest`` is the SHA256 of the canonical JSON
    encoding of ``manifest``. Cache timestamps and absolute paths are excluded.
    """

    root = Path(local_dir).resolve()
    revision = expected_revision.strip().lower()
    if not root.is_dir():
        raise ModelArtifactError(f"local model directory does not exist: {root}")
    if not _COMMIT_RE.fullmatch(revision):
        raise ModelArtifactError(
            "expected revision must be a 40-character Git commit SHA"
        )
    incomplete = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*.incomplete")
    )
    if incomplete:
        raise ModelArtifactError(
            f"incomplete Hugging Face downloads present: {incomplete[0]}"
        )

    required_paths, weight_paths = _required_paths(root)
    files: list[dict[str, Any]] = []
    for relative_path in required_paths:
        path = root.joinpath(*relative_path.parts)
        commit, metadata_digest = _metadata_for(root, relative_path)
        if commit != revision:
            raise ModelArtifactError(
                f"revision mismatch for {relative_path}: expected {revision}, got {commit}"
            )
        actual_digest = _sha256(path)
        is_weight = relative_path in weight_paths
        if is_weight:
            if not _SHA256_RE.fullmatch(metadata_digest):
                raise ModelArtifactError(
                    f"weight metadata is not an LFS SHA256 for {relative_path}"
                )
            if actual_digest != metadata_digest:
                raise ModelArtifactError(f"weight SHA256 mismatch: {relative_path}")
        files.append(
            {
                "path": relative_path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": actual_digest,
                "metadata_object_digest": metadata_digest,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "source": "huggingface-local-dir",
        "revision": revision,
        "files": files,
    }
    canonical = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "manifest": manifest,
        "model_artifact_digest": hashlib.sha256(canonical).hexdigest(),
    }
