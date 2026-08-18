from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SNAPSHOT_FILES = ("leaderboard_single.json", "leaderboard_multi.json")
HISTORICAL_UNVERIFIED_PREFIX = (
    "valid historical result; strict baseline target admission not completed"
)


@dataclass(frozen=True)
class OfficialTargetRegistry:
    version: str
    sha256: str
    targets: dict[str, dict[str, Any]]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_official_target_registry(repo_root: Path) -> OfficialTargetRegistry:
    registry_path = repo_root / "leaderboard-data" / "official-targets.json"
    checksum_path = repo_root / "leaderboard-data" / "official-targets.sha256"
    content = registry_path.read_bytes()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    declared_sha256 = checksum_path.read_text(encoding="utf-8").split()[0]
    if actual_sha256 != declared_sha256:
        raise ValueError(
            "official target registry checksum mismatch: "
            f"declared={declared_sha256} actual={actual_sha256}"
        )

    payload = json.loads(content)
    if not isinstance(payload, Mapping) or not payload.get("registry_version"):
        raise ValueError("official target registry must declare registry_version")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list):
        raise TypeError("official target registry targets must be an array")

    targets: dict[str, dict[str, Any]] = {}
    for raw_target in raw_targets:
        if not isinstance(raw_target, dict) or not raw_target.get("target_id"):
            raise ValueError("every official target must declare target_id")
        target_id = str(raw_target["target_id"])
        if target_id in targets:
            raise ValueError(f"duplicate official target_id: {target_id}")
        targets[target_id] = raw_target
    return OfficialTargetRegistry(
        version=str(payload["registry_version"]),
        sha256=actual_sha256,
        targets=targets,
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require_values(
    expected: Mapping[str, Any], actual: Any, *, prefix: str
) -> list[str]:
    if not isinstance(actual, Mapping):
        return [f"{prefix} must be an object"]
    errors: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, Mapping):
            errors.extend(
                _require_values(expected_value, actual_value, prefix=f"{prefix}.{key}")
            )
        elif actual_value != expected_value:
            errors.append(
                f"{prefix}.{key} mismatch: actual={actual_value!r} "
                f"expected={expected_value!r}"
            )
    return errors


def _require_client_values(expected: Mapping[str, Any], actual: Any) -> list[str]:
    if not isinstance(actual, Mapping):
        return ["same_spec.resolved_client_parameters must be an object"]
    remaining = dict(expected)
    errors: list[str] = []
    if expected.get("dataset_name") == "prefix_repetition" and (
        "prefix_repetition_prefix_len" in actual
        or "prefix_repetition_suffix_len" in actual
    ):
        expected_input = remaining.pop("input_len", None)
        expected_output = remaining.pop("output_len", None)
        actual_input = int(actual.get("prefix_repetition_prefix_len") or 0) + int(
            actual.get("prefix_repetition_suffix_len") or 0
        )
        actual_output = actual.get("prefix_repetition_output_len")
        if expected_input is not None and actual_input != expected_input:
            errors.append(
                "same_spec.resolved_client_parameters prefix input mismatch: "
                f"actual={actual_input!r} expected={expected_input!r}"
            )
        if expected_output is not None and actual_output != expected_output:
            errors.append(
                "same_spec.resolved_client_parameters prefix output mismatch: "
                f"actual={actual_output!r} expected={expected_output!r}"
            )
    errors.extend(
        _require_values(
            remaining,
            actual,
            prefix="same_spec.resolved_client_parameters",
        )
    )
    return errors


def official_target_binding_errors(
    entry: Mapping[str, Any], target: Mapping[str, Any]
) -> list[str]:
    """Return every strict registry-contract mismatch for a snapshot entry."""
    errors: list[str] = []
    same_spec = _mapping(entry.get("same_spec"))
    target_id = str(target.get("target_id") or "")
    if str(same_spec.get("spec_id") or "") != target_id:
        errors.append("same_spec.spec_id does not equal registry target_id")
    if str(same_spec.get("scenario") or "") != str(
        _mapping(target.get("workload")).get("name") or ""
    ):
        errors.append("same_spec.scenario does not match registry workload")

    baseline = _mapping(target.get("baseline_runtime"))
    if entry.get("engine") != baseline.get("engine"):
        errors.append("engine does not match target baseline runtime")
    if entry.get("engine_version") != baseline.get("engine_version"):
        errors.append("engine_version does not match target baseline runtime")

    target_model = _mapping(target.get("model"))
    entry_model = _mapping(entry.get("model"))
    expected_entry_model = {
        "repo_id": target_model.get("id"),
        "parameters": target_model.get("parameters"),
        "precision": target_model.get("precision"),
    }
    errors.extend(_require_values(expected_entry_model, entry_model, prefix="model"))
    expected_same_spec_model = {
        "model": target_model.get("id"),
        "model_parameters": target_model.get("parameters"),
        "model_precision": target_model.get("precision"),
        "model_quantization": str(target_model.get("quantization") or ""),
    }
    errors.extend(
        _require_values(
            expected_same_spec_model,
            same_spec,
            prefix="same_spec",
        )
    )

    target_hardware = _mapping(target.get("hardware"))
    expected_entry_hardware = {
        "vendor": target_hardware.get("vendor"),
        "chip_model": target_hardware.get("chip_model"),
        "chip_count": target_hardware.get("chip_count"),
    }
    errors.extend(
        _require_values(
            expected_entry_hardware,
            entry.get("hardware"),
            prefix="hardware",
        )
    )
    expected_same_spec_hardware = {
        "hardware_vendor": target_hardware.get("vendor"),
        "hardware_chip_model": target_hardware.get("chip_model"),
        "chip_count": target_hardware.get("chip_count"),
        "node_count": target_hardware.get("node_count"),
    }
    errors.extend(
        _require_values(
            expected_same_spec_hardware,
            same_spec,
            prefix="same_spec",
        )
    )

    errors.extend(
        _require_values(
            _mapping(target.get("server_parameters")),
            same_spec.get("resolved_server_parameters"),
            prefix="same_spec.resolved_server_parameters",
        )
    )
    expected_client = _mapping(
        _mapping(target.get("workload")).get("client_parameters")
    )
    errors.extend(
        _require_client_values(
            expected_client,
            same_spec.get("resolved_client_parameters"),
        )
    )
    return errors


def bind_entry_to_official_target(
    entry: dict[str, Any], registry: OfficialTargetRegistry
) -> tuple[bool, list[str]]:
    """Bind an exact public entry or explicitly retain it as unverified history."""
    metadata = entry.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("snapshot entry metadata must be an object")
    same_spec = _mapping(entry.get("same_spec"))
    target_id = str(same_spec.get("spec_id") or "")
    target = registry.targets.get(target_id)
    errors: list[str] = []
    if target is None:
        errors.append(f"same_spec.spec_id {target_id!r} is absent from registry")
    else:
        errors.extend(official_target_binding_errors(entry, target))
        if target.get("status") != "active" or target.get("intended_use") != (
            "public-leaderboard"
        ):
            errors.append(
                "registry target is not active public-leaderboard: "
                f"status={target.get('status')!r} "
                f"intended_use={target.get('intended_use')!r}"
            )

    if errors:
        metadata["verified"] = False
        for field in (
            "target_id",
            "target_version",
            "profile_id",
            "target_registry_sha256",
        ):
            metadata.pop(field, None)
        metadata["official_admission_status"] = "historical-unverified"
        metadata["official_admission_reason"] = (
            f"{HISTORICAL_UNVERIFIED_PREFIX}: " + "; ".join(errors)
        )
        return False, errors

    assert target is not None
    metadata.update(
        {
            "verified": True,
            "target_id": target_id,
            "target_version": str(target["target_version"]),
            "profile_id": str(target["profile"]),
            "target_registry_sha256": registry.sha256,
        }
    )
    metadata.pop("official_admission_status", None)
    metadata.pop("official_admission_reason", None)
    return True, []


def bind_snapshot_set(
    snapshot_dir: Path, registry: OfficialTargetRegistry
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "snapshot-target-binding-report/v1",
        "target_registry_version": registry.version,
        "target_registry_sha256": registry.sha256,
        "files": {},
        "verified": 0,
        "historical_unverified": 0,
        "reason_counts": {},
    }
    for file_name in SNAPSHOT_FILES:
        path = snapshot_dir / file_name
        if not path.is_file():
            continue
        payload = _load_json(path)
        if not isinstance(payload, list):
            raise TypeError(f"{path} must contain a JSON array")
        verified = 0
        historical_unverified = 0
        reason_counts: dict[str, int] = {}
        for entry in payload:
            if not isinstance(entry, dict):
                raise TypeError(f"{path} contains a non-object entry")
            is_verified, errors = bind_entry_to_official_target(entry, registry)
            if is_verified:
                verified += 1
            else:
                historical_unverified += 1
                for error in errors:
                    reason = error.split(": actual=", 1)[0]
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["files"][file_name] = {
            "entries": len(payload),
            "verified": verified,
            "historical_unverified": historical_unverified,
            "reason_counts": dict(sorted(reason_counts.items())),
        }
        report["verified"] += verified
        report["historical_unverified"] += historical_unverified
        for reason, count in reason_counts.items():
            report["reason_counts"][reason] = (
                report["reason_counts"].get(reason, 0) + count
            )
    report["reason_counts"] = dict(sorted(report["reason_counts"].items()))
    report_path = snapshot_dir / "target_binding_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
