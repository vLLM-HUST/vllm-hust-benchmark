from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from vllm_hust_benchmark.leaderboard_export import _sanitize_component_version
from vllm_hust_benchmark.model_registry import normalize_model_identity_payload
from vllm_hust_benchmark.model_registry import validate_model_identity_payload

SUPPORTED_MANIFEST_SCHEMA_VERSIONS = {
    "leaderboard-export-manifest/v1",
    "leaderboard-export-manifest/v2",
}

# Canonical version keys defined by the leaderboard export schema.
# Any extraneous keys in artifact.versions will be stripped during normalization.
CANONICAL_VERSION_KEYS = frozenset({
    "protocol",
    "backend",
    "core",
    "benchmark",
})

RELEASE_LIKE_ENGINE_VERSION_PATTERN = re.compile(
    r"^v?\d+(?:\.\d+)+(?:[A-Za-z0-9._+-]*)?$"
)

HISTORICAL_SAME_SPEC_COMPONENT_OVERRIDES = {
    "vllm-ascend-hust-ci-same-spec": {
        "core": "0.17.2.post1",
        "backend": "0.1.0",
    },
    "vllm-hust-ci-same-spec": {
        "core": "0.17.2.post1",
        "backend": "0.18.0.post1",
    },
}


def _repository_name(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return ""
    parts = [part for part in normalized.split("/") if part]
    return parts[-1] if parts else normalized


def _is_missing_component_version(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"", "n/a", "unknown"}


def _looks_like_release_engine_version(value: Any) -> bool:
    normalized = str(value or "").strip()
    if not normalized:
        return False
    return RELEASE_LIKE_ENGINE_VERSION_PATTERN.match(normalized) is not None


def _get_component_versions(artifact: dict[str, Any]) -> dict[str, Any] | None:
    versions = artifact.get("versions")
    if not isinstance(versions, dict):
        return None
    return versions


def _normalize_component_versions_in_place(artifact: dict[str, Any]) -> dict[str, Any]:
    versions = _get_component_versions(artifact)
    if versions is None:
        return artifact

    for key, value in list(versions.items()):
        versions[key] = _sanitize_component_version(value)

    artifact["versions"] = versions
    return artifact


def _backfill_versions_from_repository(artifact: dict[str, Any]) -> dict[str, Any]:
    versions = _get_component_versions(artifact)
    if versions is None:
        return artifact

    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
    engine_version = str(
        artifact.get("engine_version") or metadata.get("engine_version") or ""
    ).strip()
    sanitized_engine_version = _sanitize_component_version(engine_version)
    if _is_missing_component_version(sanitized_engine_version):
        return artifact

    repository = _repository_name(metadata.get("github_repository"))
    if repository == "vllm-hust" and _is_missing_component_version(versions.get("core")):
        versions["core"] = sanitized_engine_version
    if repository == "vllm-ascend-hust" and _is_missing_component_version(versions.get("backend")):
        versions["backend"] = sanitized_engine_version
    artifact["versions"] = versions
    return artifact


def _backfill_versions_from_historical_source(
    artifact: dict[str, Any], *, keys: tuple[str, ...]
) -> dict[str, Any]:
    versions = _get_component_versions(artifact)
    if versions is None:
        return artifact

    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
    data_source = str(metadata.get("data_source") or "").strip().lower()
    overrides = HISTORICAL_SAME_SPEC_COMPONENT_OVERRIDES.get(data_source)
    if not overrides:
        return artifact

    for key in keys:
        override = overrides.get(key)
        if not override or not _is_missing_component_version(versions.get(key)):
            continue
        versions[key] = _sanitize_component_version(override)

    artifact["versions"] = versions
    return artifact


def _artifact_pairing_key(artifact: dict[str, Any]) -> tuple[str, ...]:
    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
    model = artifact.get("model") if isinstance(artifact.get("model"), dict) else {}
    workload = artifact.get("workload") if isinstance(artifact.get("workload"), dict) else {}
    hardware = artifact.get("hardware") if isinstance(artifact.get("hardware"), dict) else {}
    cluster = artifact.get("cluster") if isinstance(artifact.get("cluster"), dict) else {}
    submitted_at = str(metadata.get("submitted_at") or "").strip()
    return (
        str(model.get("name") or "").strip(),
        str(workload.get("name") or "").strip(),
        str(artifact.get("config_type") or "").strip(),
        str(hardware.get("chip_model") or "").strip(),
        str(hardware.get("chip_count") or "").strip(),
        str(cluster.get("node_count") or 1).strip(),
        submitted_at[:10],
    )


def _artifact_metric_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_metrics = left.get("metrics") if isinstance(left.get("metrics"), dict) else {}
    right_metrics = right.get("metrics") if isinstance(right.get("metrics"), dict) else {}
    for key in ("throughput_tps", "ttft_ms"):
        left_value = left_metrics.get(key)
        right_value = right_metrics.get(key)
        if left_value is None or right_value is None:
            continue
        try:
            left_number = float(left_value)
            right_number = float(right_value)
        except (TypeError, ValueError):
            return False
        if abs(left_number - right_number) > 0.05:
            return False
    return True


def _resolved_component_version(artifact: dict[str, Any], key: str) -> str:
    versions = _get_component_versions(artifact) or {}
    value = str(versions.get(key) or "").strip()
    if not _is_missing_component_version(value):
        return value
    if key == "core":
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        engine_version = str(
            artifact.get("engine_version") or metadata.get("engine_version") or ""
        ).strip()
        sanitized_engine_version = _sanitize_component_version(engine_version)
        if not _is_missing_component_version(sanitized_engine_version):
            return sanitized_engine_version
    return ""


def _backfill_versions_from_matching_artifacts(
    artifacts_by_path: dict[Path, dict[str, Any]],
) -> None:
    artifacts = list(artifacts_by_path.values())
    for artifact in artifacts:
        versions = _get_component_versions(artifact)
        if versions is None:
            continue

        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        repository = _repository_name(metadata.get("github_repository"))
        if repository != "vllm-ascend-hust" or not _is_missing_component_version(versions.get("core")):
            continue

        backend_version = _resolved_component_version(artifact, "backend")
        if not backend_version:
            continue

        matching_core_artifacts = [
            other
            for other in artifacts
            if other is not artifact
            and _repository_name(
                (other.get("metadata") if isinstance(other.get("metadata"), dict) else {}).get(
                    "github_repository"
                )
            )
            == "vllm-hust"
            and _artifact_pairing_key(other) == _artifact_pairing_key(artifact)
            and _artifact_metric_matches(other, artifact)
            and _resolved_component_version(other, "backend") == backend_version
            and _resolved_component_version(other, "core")
        ]

        if len(matching_core_artifacts) != 1:
            continue

        versions["core"] = _resolved_component_version(matching_core_artifacts[0], "core")
        artifact["versions"] = versions


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
    artifact = _normalize_component_versions_in_place(artifact)
    artifact = _backfill_versions_from_repository(artifact)
    artifact = _backfill_versions_from_historical_source(artifact, keys=("backend",))
    # Strip any version keys not in the canonical schema to prevent
    # extraneous fields (e.g. os_version, vllm_version) from causing
    # normalization drift for externally-generated artifacts.
    versions = _get_component_versions(artifact)
    if versions is not None:
        artifact["versions"] = {
            k: v for k, v in versions.items() if k in CANONICAL_VERSION_KEYS
        }
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
    artifact_paths = iter_submission_artifact_paths(source_dir)
    artifacts_by_path = {
        artifact_path: normalize_submission_artifact_contract(
            load_submission_artifact(artifact_path)
        )
        for artifact_path in artifact_paths
    }
    _backfill_versions_from_matching_artifacts(artifacts_by_path)
    for artifact in artifacts_by_path.values():
        _backfill_versions_from_historical_source(artifact, keys=("core",))

    changed: list[Path] = []
    for artifact_path, artifact in artifacts_by_path.items():
        serialized = json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
        current = artifact_path.read_text(encoding="utf-8")
        if current == serialized:
            continue
        artifact_path.write_text(serialized, encoding="utf-8")
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
