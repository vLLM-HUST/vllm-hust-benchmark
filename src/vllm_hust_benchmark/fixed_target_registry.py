from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REGISTRY_SCHEMA_VERSION = "fixed-target-registry/v1"
REGISTRY_PATH = Path(__file__).parent / "data" / "fixed_target_registry.json"

_REQUIRED_PROFILE_FIELDS: tuple[str, ...] = (
    "target_id",
    "target_version",
    "profile_name",
    "model",
    "hardware_chip_model",
    "chip_count",
    "model_precision",
    "tensor_parallel_size",
    "workload_name",
    "status",
)
_OPTIONAL_PROFILE_FIELDS: tuple[str, ...] = (
    "gpu_memory_utilization",
    "max_model_len",
)
_VALID_STATUS: tuple[str, ...] = ("active", "retired", "specialty")


@dataclass(frozen=True)
class FixedTargetProfile:
    target_id: str
    target_version: str
    profile_name: str
    model: str
    hardware_chip_model: str
    chip_count: int
    model_precision: str
    tensor_parallel_size: int
    gpu_memory_utilization: float | None
    max_model_len: int | None
    workload_name: str
    status: str  # "active" / "retired" / "specialty"


def load_fixed_target_registry(
    path: Path | None = None,
) -> tuple[FixedTargetProfile, ...]:
    """加载并校验 registry。失败时 raise ValueError。"""
    target_path = path or REGISTRY_PATH
    try:
        with target_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"registry file not found: {target_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"registry file is not valid JSON: {target_path}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError("registry top-level payload must be a JSON object")

    schema_version = str(payload.get("schema_version") or "")
    if schema_version != REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {REGISTRY_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raise ValueError("profiles must be a list")

    profiles: list[FixedTargetProfile] = []
    for index, raw in enumerate(raw_profiles):
        if not isinstance(raw, Mapping):
            raise ValueError(f"profiles[{index}] must be a JSON object")
        profiles.append(_build_profile(raw, index))

    return tuple(profiles)


def get_active_profiles(
    registry: tuple[FixedTargetProfile, ...],
) -> tuple[FixedTargetProfile, ...]:
    """返回 status=active 的 profile。"""
    return tuple(p for p in registry if p.status == "active")


def find_matching_profile(
    entry: Mapping[str, Any],
    registry: tuple[FixedTargetProfile, ...],
) -> FixedTargetProfile | None:
    """根据 entry 的 model、hardware、precision、chip_count、workload 找到匹配的 profile。

    返回首个匹配项；若无匹配返回 None。
    """
    entry_model = entry.get("model")
    entry_model = entry_model if isinstance(entry_model, Mapping) else {}
    entry_hardware = entry.get("hardware")
    entry_hardware = entry_hardware if isinstance(entry_hardware, Mapping) else {}
    entry_workload = entry.get("workload")
    entry_workload = entry_workload if isinstance(entry_workload, Mapping) else {}

    candidate_repo_ids = (
        entry_model.get("repo_id"),
        entry_model.get("canonical_id"),
        entry_model.get("name"),
    )
    entry_chip_model = str(entry_hardware.get("chip_model") or "")
    entry_chip_count = entry_hardware.get("chip_count")
    entry_precision = str(entry_model.get("precision") or "")
    entry_workload_name = str(entry_workload.get("name") or "")

    registry_workload_names = {p.workload_name for p in registry}
    workload_match_required = entry_workload_name in registry_workload_names

    for profile in registry:
        if profile.model not in candidate_repo_ids:
            continue
        if profile.hardware_chip_model != entry_chip_model:
            continue
        if profile.chip_count != entry_chip_count:
            continue
        if profile.model_precision != entry_precision:
            continue
        if workload_match_required and profile.workload_name != entry_workload_name:
            continue
        return profile

    return None


def _build_profile(raw: Mapping[str, Any], index: int) -> FixedTargetProfile:
    missing = [key for key in _REQUIRED_PROFILE_FIELDS if key not in raw]
    if missing:
        raise ValueError(
            f"profiles[{index}] missing required fields: {', '.join(missing)}"
        )

    status = str(raw["status"])
    if status not in _VALID_STATUS:
        raise ValueError(
            f"profiles[{index}].status must be one of {_VALID_STATUS}, got {status!r}"
        )

    if status in ("active", "retired"):
        for key in ("gpu_memory_utilization", "max_model_len"):
            if key not in raw:
                raise ValueError(
                    f"profiles[{index}] missing field {key!r} required for "
                    f"status={status!r}"
                )

    allowed_fields = _REQUIRED_PROFILE_FIELDS + _OPTIONAL_PROFILE_FIELDS
    unexpected = [key for key in raw if key not in allowed_fields]
    if unexpected:
        raise ValueError(
            f"profiles[{index}] has unexpected fields: {', '.join(unexpected)}"
        )

    chip_count = _coerce_int(raw["chip_count"], f"profiles[{index}].chip_count")
    tensor_parallel_size = _coerce_int(
        raw["tensor_parallel_size"], f"profiles[{index}].tensor_parallel_size"
    )

    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    if "gpu_memory_utilization" in raw:
        gpu_memory_utilization = _coerce_float(
            raw["gpu_memory_utilization"],
            f"profiles[{index}].gpu_memory_utilization",
        )
    if "max_model_len" in raw:
        max_model_len = _coerce_int(
            raw["max_model_len"], f"profiles[{index}].max_model_len"
        )

    return FixedTargetProfile(
        target_id=str(raw["target_id"]),
        target_version=str(raw["target_version"]),
        profile_name=str(raw["profile_name"]),
        model=str(raw["model"]),
        hardware_chip_model=str(raw["hardware_chip_model"]),
        chip_count=chip_count,
        model_precision=str(raw["model_precision"]),
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        workload_name=str(raw["workload_name"]),
        status=status,
    )


def _coerce_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc


def _coerce_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number, got {value!r}") from exc
