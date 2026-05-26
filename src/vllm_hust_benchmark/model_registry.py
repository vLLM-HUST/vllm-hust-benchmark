from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

DEFAULT_MODEL_REGISTRY = "hf"
CANONICAL_ID_PATTERN = re.compile(r"^(?P<registry>[a-z0-9][a-z0-9_-]*):(?P<repo_id>.+)$")
HF_CACHE_PATH_PATTERN = re.compile(
    r"(?:^|/)models--(?P<namespace>[^/]+)--(?P<name>[^/]+)/(?:snapshots|refs)/",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ModelIdentity:
    canonical_id: str
    registry: str
    repo_id: str
    short_name: str
    display_name: str
    aliases: tuple[str, ...] = ()


def _looks_like_repo_id(value: str) -> bool:
    if not value or value.startswith("/"):
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(part.strip() for part in parts)


def _parse_canonical_id(value: str) -> tuple[str, str] | None:
    match = CANONICAL_ID_PATTERN.match(value)
    if not match:
        return None
    return match.group("registry"), match.group("repo_id")


def _extract_hf_repo_id_from_path(value: str) -> str | None:
    match = HF_CACHE_PATH_PATTERN.search(value)
    if not match:
        return None
    namespace = match.group("namespace")
    name = match.group("name")
    return f"{namespace}/{name}"


def _build_fallback_identity(repo_id: str, *, registry: str) -> ModelIdentity:
    short_name = repo_id.rsplit("/", maxsplit=1)[-1]
    return ModelIdentity(
        canonical_id=f"{registry}:{repo_id}",
        registry=registry,
        repo_id=repo_id,
        short_name=short_name,
        display_name=short_name,
        aliases=(),
    )


@lru_cache(maxsize=1)
def load_model_identity_registry() -> tuple[ModelIdentity, ...]:
    with resources.files("vllm_hust_benchmark.data").joinpath(
        "model_identity_registry.json"
    ).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("model identity registry must contain a models array")

    records: list[ModelIdentity] = []
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("model identity registry entries must be objects")
        aliases = item.get("aliases") or []
        if not isinstance(aliases, list):
            raise ValueError("model identity registry aliases must be an array")
        records.append(
            ModelIdentity(
                canonical_id=str(item["canonical_id"]),
                registry=str(item["registry"]),
                repo_id=str(item["repo_id"]),
                short_name=str(item["short_name"]),
                display_name=str(item["display_name"]),
                aliases=tuple(str(alias) for alias in aliases if str(alias).strip()),
            )
        )
    return tuple(records)


@lru_cache(maxsize=1)
def _build_identity_lookup() -> dict[str, ModelIdentity]:
    lookup: dict[str, ModelIdentity] = {}
    for record in load_model_identity_registry():
        keys = {
            record.canonical_id,
            record.repo_id,
            record.short_name,
            *record.aliases,
        }
        for key in keys:
            existing = lookup.get(key)
            if existing is not None and existing != record:
                raise ValueError(f"duplicate model identity alias in registry: {key}")
            lookup[key] = record
    return lookup


def resolve_model_identity(
    raw_model_name: str,
    *,
    default_registry: str = DEFAULT_MODEL_REGISTRY,
) -> ModelIdentity:
    normalized = str(raw_model_name or "").strip()
    if not normalized:
        raise ValueError("model name is required for leaderboard export")

    lookup = _build_identity_lookup()
    direct_match = lookup.get(normalized)
    if direct_match is not None:
        return direct_match

    canonical = _parse_canonical_id(normalized)
    if canonical is not None:
        registry, repo_id = canonical
        seeded = lookup.get(repo_id)
        return seeded if seeded is not None else _build_fallback_identity(repo_id, registry=registry)

    repo_id_from_path = _extract_hf_repo_id_from_path(normalized)
    if repo_id_from_path is not None:
        seeded = lookup.get(repo_id_from_path)
        return (
            seeded
            if seeded is not None
            else _build_fallback_identity(repo_id_from_path, registry=DEFAULT_MODEL_REGISTRY)
        )

    if _looks_like_repo_id(normalized):
        return _build_fallback_identity(normalized, registry=default_registry)

    raise ValueError(
        "unknown short model alias; add it to model_identity_registry.json or pass a repo id"
    )


def resolve_model_identity_from_payload(
    model_payload: Mapping[str, object],
    *,
    default_registry: str = DEFAULT_MODEL_REGISTRY,
) -> ModelIdentity:
    candidates = [
        str(model_payload.get("canonical_id") or "").strip(),
        str(model_payload.get("repo_id") or "").strip(),
        str(model_payload.get("name") or "").strip(),
        str(model_payload.get("short_name") or "").strip(),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        return resolve_model_identity(candidate, default_registry=default_registry)
    raise ValueError("model payload must contain canonical_id, repo_id, name, or short_name")


def normalize_model_identity_payload(
    model_payload: Mapping[str, object],
    *,
    default_registry: str = DEFAULT_MODEL_REGISTRY,
) -> dict[str, object]:
    identity = resolve_model_identity_from_payload(
        model_payload,
        default_registry=default_registry,
    )
    normalized = dict(model_payload)
    normalized.update(
        {
            "canonical_id": identity.canonical_id,
            "repo_id": identity.repo_id,
            "short_name": identity.short_name,
            "display_name": identity.display_name,
            "name": identity.repo_id,
        }
    )
    return normalized


def validate_model_identity_payload(model_payload: Mapping[str, object]) -> None:
    required_fields = (
        "canonical_id",
        "repo_id",
        "short_name",
        "display_name",
        "name",
    )
    missing = [field for field in required_fields if not str(model_payload.get(field) or "").strip()]
    if missing:
        raise ValueError(
            "model payload missing normalized identity fields: " + ", ".join(missing)
        )
    if str(model_payload.get("name") or "").strip() != str(model_payload.get("repo_id") or "").strip():
        raise ValueError("model payload must set model.name equal to model.repo_id")