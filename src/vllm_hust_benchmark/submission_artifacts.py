from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from vllm_hust_benchmark.model_registry import normalize_model_identity_payload
from vllm_hust_benchmark.model_registry import validate_model_identity_payload

SUPPORTED_MANIFEST_SCHEMA_VERSIONS = {
    "leaderboard-export-manifest/v1",
    "leaderboard-export-manifest/v2",
}


def load_submission_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path}: manifest payload must be a JSON object")

    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError(
            f"{manifest_path}: unsupported schema_version {payload.get('schema_version')!r}"
        )

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{manifest_path}: entries must be a list")
    return payload


def iter_manifest_artifact_paths(manifest_path: Path) -> list[Path]:
    payload = load_submission_manifest(manifest_path)
    artifact_paths: list[Path] = []
    for index, record in enumerate(payload["entries"]):
        if not isinstance(record, dict):
            raise ValueError(f"{manifest_path}: entries[{index}] must be a JSON object")
        artifact_rel = record.get("leaderboard_artifact")
        if not isinstance(artifact_rel, str) or not artifact_rel.strip():
            raise ValueError(
                f"{manifest_path}: entries[{index}].leaderboard_artifact is required"
            )
        artifact_path = manifest_path.parent / artifact_rel
        if not artifact_path.is_file():
            raise ValueError(
                f"{manifest_path}: missing leaderboard artifact {artifact_path}"
            )
        artifact_paths.append(artifact_path)
    return artifact_paths


def iter_submission_artifact_paths(source_dir: Path) -> list[Path]:
    artifact_paths: list[Path] = []
    for manifest_path in sorted(source_dir.rglob("leaderboard_manifest.json")):
        artifact_paths.extend(iter_manifest_artifact_paths(manifest_path))
    return artifact_paths


def ensure_submission_manifests_in_tree(source_dir: Path) -> list[Path]:
    created: list[Path] = []
    for artifact_path in sorted(source_dir.rglob("run_leaderboard.json")):
        manifest_path = artifact_path.parent / "leaderboard_manifest.json"
        if manifest_path.exists():
            continue
        artifact = load_submission_artifact(artifact_path)
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        entry: dict[str, str] = {
            "leaderboard_artifact": artifact_path.name,
        }
        idempotency_key = str(metadata.get("idempotency_key") or "").strip()
        if idempotency_key:
            entry["idempotency_key"] = idempotency_key
        manifest = {
            "schema_version": "leaderboard-export-manifest/v2",
            "generated_at": str(metadata.get("submitted_at") or ""),
            "entries": [entry],
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        created.append(manifest_path)
    return created


def load_submission_artifact(artifact_path: Path) -> dict[str, Any]:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_path}: leaderboard artifact must be a JSON object")
    return payload


def normalize_submission_artifact_contract(artifact: dict[str, Any]) -> dict[str, Any]:
    model_payload = artifact.get("model")
    if not isinstance(model_payload, dict):
        raise ValueError("leaderboard artifact model payload must be an object")
    normalized_model_payload = normalize_model_identity_payload(model_payload)
    validate_model_identity_payload(normalized_model_payload)
    artifact["model"] = normalized_model_payload
    return artifact


def normalize_submission_artifact_file(artifact_path: Path) -> bool:
    artifact = normalize_submission_artifact_contract(load_submission_artifact(artifact_path))
    serialized = json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
    current = artifact_path.read_text(encoding="utf-8")
    if current == serialized:
        return False
    artifact_path.write_text(serialized, encoding="utf-8")
    return True


def normalize_submission_artifacts_in_tree(source_dir: Path) -> list[Path]:
    ensure_submission_manifests_in_tree(source_dir)
    changed: list[Path] = []
    for artifact_path in iter_submission_artifact_paths(source_dir):
        if normalize_submission_artifact_file(artifact_path):
            changed.append(artifact_path)
    return changed


def validate_submission_artifacts(
    *,
    source_dir: Path,
    validator: Draft7Validator,
) -> list[str]:
    errors: list[str] = []
    artifact_paths = iter_submission_artifact_paths(source_dir)
    if not artifact_paths:
        return errors

    for artifact_path in artifact_paths:
        artifact = load_submission_artifact(artifact_path)
        artifact = normalize_submission_artifact_contract(artifact)
        schema_errors = sorted(
            validator.iter_errors(artifact), key=lambda error: list(error.path)
        )
        if schema_errors:
            first = schema_errors[0]
            errors.append(
                f"{artifact_path}: {first.message} @ {list(first.path)}"
            )
    return errors


def validate_manifest_artifacts(
    *,
    manifests: Iterable[Path],
    validator: Draft7Validator,
) -> list[str]:
    errors: list[str] = []
    for manifest_path in manifests:
        try:
            artifact_paths = iter_manifest_artifact_paths(manifest_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for artifact_path in artifact_paths:
            try:
                artifact = normalize_submission_artifact_contract(
                    load_submission_artifact(artifact_path)
                )
            except ValueError as exc:
                errors.append(f"{artifact_path}: {exc}")
                continue
            schema_errors = sorted(
                validator.iter_errors(artifact), key=lambda error: list(error.path)
            )
            if schema_errors:
                first = schema_errors[0]
                errors.append(
                    f"{artifact_path}: {first.message} @ {list(first.path)}"
                )
    return errors
