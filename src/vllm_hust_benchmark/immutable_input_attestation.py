from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "immutable-input-attestation/v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def _require_file(
    path: Path, identity: Mapping[str, Any], *, require_size: bool = True
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"immutable input file not found: {path}")
    actual = file_identity(path)
    fields = ("size_bytes", "sha256") if require_size else ("sha256",)
    for field in fields:
        if field not in identity:
            raise ValueError(f"data_identity lacks {field}: {identity}")
        if actual[field] != identity[field]:
            raise ValueError(
                f"immutable input {field} mismatch for {path}: "
                f"{actual[field]} != {identity[field]}"
            )


def _require_revision(identity: Mapping[str, Any]) -> str:
    revision = identity.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("data_identity requires an exact 40-character revision")
    return revision


def resolve_sharegpt_dataset_url(
    data_identity: Mapping[str, Any], explicit_url: str
) -> str:
    """Resolve an exact-revision ShareGPT URL or reject a drifting override."""
    if data_identity.get("kind") != "huggingface-file":
        return explicit_url
    revision = _require_revision(data_identity)
    expected_url = (
        "https://hf-mirror.com/datasets/"
        f"{data_identity['repository']}/resolve/{revision}/{data_identity['path']}"
    )
    if explicit_url and explicit_url != expected_url:
        raise ValueError(f"ShareGPT URL is not the exact revision URL: {explicit_url}")
    return expected_url


def verify_data_contract(
    data_identity: Mapping[str, Any],
    *,
    benchmark_repo: Path,
    vllm_worktree: Path,
    dataset_root: Path,
    sharegpt_url: str,
    trace_asset_path: Path | None = None,
) -> dict[str, object]:
    """Fail closed unless the materialized benchmark input matches its contract."""
    kind = data_identity.get("kind")
    if kind == "repository-file":
        _require_file(benchmark_repo / str(data_identity["path"]), data_identity)
    elif kind == "vllm-repository-file":
        _require_file(vllm_worktree / str(data_identity["path"]), data_identity)
    elif kind in {
        "deterministic-vllm-generator",
        "nondeterministic-vllm-generator",
    }:
        generator_identity = {"sha256": data_identity.get("generator_sha256")}
        _require_file(
            vllm_worktree / str(data_identity["generator_path"]),
            generator_identity,
            require_size=False,
        )
    elif kind == "huggingface-file":
        expected_url = resolve_sharegpt_dataset_url(data_identity, sharegpt_url)
        if sharegpt_url != expected_url:
            raise ValueError(
                f"ShareGPT URL is not the exact revision URL: {sharegpt_url}"
            )
        _require_file(dataset_root / str(data_identity["path"]), data_identity)
    elif kind == "huggingface-dataset":
        revision = _require_revision(data_identity)
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                str(data_identity["repository"]),
                repo_type="dataset",
                revision=revision,
                local_files_only=True,
            )
        ).resolve()
        if snapshot.name != revision or snapshot.parent.name != "snapshots":
            raise ValueError(
                f"HF dataset did not resolve to exact snapshot {revision}: {snapshot}"
            )
    elif kind == "release-asset":
        if trace_asset_path is None:
            raise ValueError("release-asset contract requires the actual trace path")
        _require_file(trace_asset_path, data_identity)
    else:
        raise ValueError(f"unsupported data_identity kind: {kind!r}")
    return dict(data_identity)


def build_metadata(spec: Mapping[str, Any]) -> dict[str, object]:
    model_revision = spec.get("model_revision") or spec.get(
        "server_parameters", {}
    ).get("revision")
    if not isinstance(model_revision, str) or len(model_revision) != 40:
        raise ValueError("official spec requires an exact model_revision")
    data_identity = spec.get("data_identity")
    if not isinstance(data_identity, dict) or not data_identity:
        raise ValueError("official spec requires a complete data_identity object")
    return {
        "model_id": spec["model"],
        "model_revision": model_revision,
        "data_identity": data_identity,
    }


def write_trace_attestation(
    output_file: Path,
    metadata: Mapping[str, object],
    summary: Mapping[str, Any],
) -> None:
    digest = summary.get("resolved_input_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("trace summary lacks actual resolved_input_sha256")
    kind = summary.get("resolved_input_kind")
    if kind != "production-trace-prompt-token-ids":
        raise ValueError("trace summary lacks the exact token-ID input kind")
    payload = {
        "schema_version": SCHEMA_VERSION,
        **metadata,
        "resolved_input_kind": kind,
        "resolved_input_sha256": digest,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_file)
