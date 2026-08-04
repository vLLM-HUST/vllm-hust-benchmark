from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping as ABCMapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "immutable-input-attestation/v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")


def canonicalize_resolved_input(value: object) -> object:
    """Convert an executed benchmark input into stable, JSON-safe evidence."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("resolved benchmark input contains a non-finite float")
        return value
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return canonicalize_resolved_input(asdict(value))
    if isinstance(value, ABCMapping):
        return {
            str(key): canonicalize_resolved_input(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize_resolved_input(item) for item in value]
    module_name = type(value).__module__
    if module_name.startswith("numpy") and hasattr(value, "tolist"):
        return canonicalize_resolved_input(value.tolist())
    if module_name.startswith("PIL.") and all(
        hasattr(value, attribute) for attribute in ("mode", "size", "tobytes")
    ):
        pixels = value.tobytes()
        return {
            "type": "pil-image",
            "mode": str(value.mode),
            "size": list(value.size),
            "pixels_sha256": hashlib.sha256(pixels).hexdigest(),
        }
    raise TypeError(
        "resolved benchmark input contains an unsupported value: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def resolved_input_sha256(*, input_kind: str, inputs: object) -> str:
    canonical = {
        "input_kind": input_kind,
        "inputs": canonicalize_resolved_input(inputs),
    }
    encoded = json.dumps(
        canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_attestation_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically persist a fully built attestation beside its final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


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
    if not isinstance(model_revision, str) or not _REVISION_RE.fullmatch(
        model_revision
    ):
        raise ValueError("official spec requires an exact model_revision")
    data_identity = spec.get("data_identity")
    if not isinstance(data_identity, dict) or not data_identity:
        raise ValueError("official spec requires a complete data_identity object")
    return {
        "model_id": spec["model"],
        "model_revision": model_revision,
        "data_identity": data_identity,
        "resolved_input_kind": expected_resolved_input_kind(spec),
    }


def expected_resolved_input_kind(spec: Mapping[str, Any]) -> str:
    data_identity = spec.get("data_identity") or {}
    if data_identity.get("kind") == "release-asset":
        return "production-trace-prompt-token-ids"
    scenario = str(spec.get("scenario") or "")
    if "latency" in scenario:
        return "latency-prompt-token-ids"
    if "throughput" in scenario:
        return "throughput-sample-requests"
    return "serve-sample-requests"


def validate_attestation_payload(
    payload: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    """Fail closed unless a captured input payload exactly matches its spec metadata."""
    expected = {
        "schema_version": SCHEMA_VERSION,
        "model_id": metadata.get("model_id"),
        "model_revision": metadata.get("model_revision"),
        "data_identity": metadata.get("data_identity"),
        "resolved_input_kind": metadata.get("resolved_input_kind"),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"immutable input attestation mismatch for {field}")
    digest = payload.get("resolved_input_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("immutable input attestation lacks a resolved input SHA256")
    data_identity = metadata.get("data_identity")
    is_trace = isinstance(data_identity, Mapping) and data_identity.get("kind") == (
        "release-asset"
    )
    if is_trace:
        return
    if "resolved_inputs" not in payload:
        raise ValueError("immutable input attestation lacks captured resolved inputs")
    inputs = canonicalize_resolved_input(payload["resolved_inputs"])
    if inputs != payload["resolved_inputs"]:
        raise ValueError("immutable input attestation inputs are not canonical")
    recomputed = resolved_input_sha256(
        input_kind=str(payload.get("resolved_input_kind") or ""), inputs=inputs
    )
    if digest != recomputed:
        raise ValueError("immutable input attestation resolved input SHA256 mismatch")


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
