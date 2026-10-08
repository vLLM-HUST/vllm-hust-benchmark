"""Validate the fixed Dense 1/2/4 matrix target (Issue #136).

This module loads ``leaderboard-data/dense-matrix-issue-136.json`` and checks
that the fixed Dense 1/2/4 matrix target is internally consistent:

* every ``spec-ready`` cell's spec file exists and its ``chip_count`` /
  ``server_parameters.tensor_parallel_size`` match the declared chip key;
* ``server_parameters.tensor_parallel_size`` is required and matches the chip key;
* every ``blocked`` cell carries a ``blocker_reason``;
* every workload has exactly the ``1chip`` / ``2chip`` / ``4chip`` cells;
* cell status is a legal value (``spec-ready`` / ``blocked``);
* the workload set is exactly the four core workloads plus
  ``communication-sensitive`` (no duplicates, no unknown workloads);
* the four core workloads are fully ``spec-ready``;
* v1/v2 keep ``communication-sensitive`` blocked, while v3 requires executable
  fixed/scaled targets using the frozen decode-heavy random profile;
* every workload declares the same frozen-stack identity (``model``,
  ``precision``, ``model_revision``, ``engine_backend_commit``,
  ``node_topology``), so all cells run the same frozen stack (#136).

This module only fixes the matrix target/config/provenance. It never computes
or publishes performance percentages.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "dense-matrix-issue-136/v3"
PREVIOUS_SCHEMA_VERSION = "dense-matrix-issue-136/v2"
LEGACY_SCHEMA_VERSION = "dense-matrix-issue-136/v1"
SUPPORTED_SCHEMA_VERSIONS = (
    LEGACY_SCHEMA_VERSION,
    PREVIOUS_SCHEMA_VERSION,
    SCHEMA_VERSION,
)
VALID_STATUS: tuple[str, ...] = ("spec-ready", "blocked")
VALID_TARGET_STATUS: tuple[str, ...] = ("target-ready", "capacity-pilot-pending")
LOAD_PROFILES: tuple[str, ...] = ("fixed-1-rps", "scaled-load")
CHIP_KEYS: tuple[str, ...] = ("1chip", "2chip", "4chip")
CORE_WORKLOADS: tuple[str, ...] = (
    "random-online",
    "sharegpt-online",
    "prefix-repetition-online",
    "agent-research-online",
)
COMMUNICATION_WORKLOAD = "communication-sensitive"
COMMUNICATION_PROFILE = "decode-heavy-tp-collective-proxy/v1"
COMMUNICATION_INPUT_LEN = 128
COMMUNICATION_OUTPUT_LEN = 1024
EXPECTED_WORKLOADS: tuple[str, ...] = CORE_WORKLOADS + (COMMUNICATION_WORKLOAD,)
# Frozen-stack identity fields that all dense cells must share (#136).
IDENTITY_FIELDS: tuple[str, ...] = (
    "model",
    "precision",
    "model_revision",
    "engine_backend_commit",
    "node_topology",
)
DEFAULT_MATRIX_PATH = Path("leaderboard-data/dense-matrix-issue-136.json")


@dataclass(frozen=True)
class DenseMatrixStatus:
    """Result of validating the fixed Dense matrix target."""

    schema_version: str
    overall: str
    spec_ready_cells: int
    blocked_cells: int
    performance_percentages_published: bool
    errors: tuple[str, ...]


def validate_dense_matrix_target(path: Path | None = None) -> DenseMatrixStatus:
    """Validate the Dense matrix definition file.

    Raise ``ValueError`` for structural problems (missing file, invalid JSON,
    wrong schema version). Semantic inconsistencies are collected into
    ``DenseMatrixStatus.errors``.
    """
    target = path or DEFAULT_MATRIX_PATH
    payload = _load_matrix(target)
    schema_version = str(payload["schema_version"])
    repo_root = target.resolve().parent.parent

    status = payload["status"]
    status = status if isinstance(status, Mapping) else {}
    overall = str(status.get("overall") or "")
    performance_percentages_published = bool(
        status.get("performance_percentages_published", False)
    )

    workloads = payload["workloads"]
    if not isinstance(workloads, list):
        raise ValueError("matrix 'workloads' must be a list")

    errors: list[str] = []
    spec_ready_count = 0
    blocked_count = 0
    seen_workloads: set[str] = set()
    identity: dict[str, str] = {}

    for index, workload in enumerate(workloads):
        if not isinstance(workload, Mapping):
            _error(f"workloads[{index}] must be a JSON object", errors)
            continue
        workload_name = str(workload.get("workload") or "")
        if not workload_name:
            _error(f"workloads[{index}] is missing 'workload'", errors)
            continue
        if workload_name in seen_workloads:
            _error(f"duplicate workload {workload_name!r}", errors)
            continue
        seen_workloads.add(workload_name)

        cells = workload.get("cells")
        if not isinstance(cells, Mapping):
            _error(f"workload {workload_name!r}: 'cells' must be a JSON object", errors)
            continue

        _validate_cells(workload_name, cells, repo_root, errors)

        for chip_key in CHIP_KEYS:
            cell = cells.get(chip_key)
            if not isinstance(cell, Mapping):
                continue
            targets = cell.get("targets")
            if isinstance(targets, Mapping):
                fixed = targets.get("fixed-1-rps")
                if isinstance(fixed, Mapping) and fixed.get("status") == "target-ready":
                    spec_ready_count += 1
                continue
            cell_status = str(cell.get("status") or "")
            if cell_status == "spec-ready":
                spec_ready_count += 1
            elif cell_status == "blocked":
                blocked_count += 1

        if workload_name in CORE_WORKLOADS:
            _require_core_ready(workload_name, cells, errors)
        if workload_name == COMMUNICATION_WORKLOAD:
            if schema_version == SCHEMA_VERSION:
                _require_core_ready(workload_name, cells, errors)
                _require_communication_targets(workload_name, cells, errors)
            else:
                _require_status(workload_name, cells, "blocked", errors)

        _check_identity(workload, workload_name, identity, errors)

    _validate_workload_set(seen_workloads, errors)
    _check_declared_count(status, "spec_ready_cells", spec_ready_count, errors)
    _check_declared_count(status, "blocked_cells", blocked_count, errors)

    return DenseMatrixStatus(
        schema_version=schema_version,
        overall=overall,
        spec_ready_cells=spec_ready_count,
        blocked_cells=blocked_count,
        performance_percentages_published=performance_percentages_published,
        errors=tuple(errors),
    )


def _load_matrix(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"matrix file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"matrix file is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("matrix top-level payload must be a JSON object")
    schema_version = str(payload.get("schema_version") or "")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"schema_version must be one of {SUPPORTED_SCHEMA_VERSIONS!r}, got "
            f"{schema_version!r}"
        )
    if "workloads" not in payload:
        raise ValueError("matrix is missing required field 'workloads'")
    return dict(payload)


def _validate_cells(
    workload_name: str,
    cells: Mapping[str, Any],
    repo_root: Path,
    errors: list[str],
) -> None:
    for chip_key in CHIP_KEYS:
        if chip_key not in cells:
            _error(
                f"workload {workload_name!r}: missing required cell {chip_key!r}",
                errors,
            )
            continue
        cell = cells[chip_key]
        if not isinstance(cell, Mapping):
            _error(
                f"workload {workload_name!r} {chip_key}: cell must be a JSON object",
                errors,
            )
            continue
        _validate_cell(workload_name, chip_key, cell, repo_root, errors)


def _validate_cell(
    workload_name: str,
    chip_key: str,
    cell: Mapping[str, Any],
    repo_root: Path,
    errors: list[str],
) -> None:
    targets = cell.get("targets")
    if isinstance(targets, Mapping):
        if set(targets) != set(LOAD_PROFILES):
            _error(
                f"workload {workload_name!r} {chip_key}: targets must be exactly "
                f"{LOAD_PROFILES!r}",
                errors,
            )
        for load_profile in LOAD_PROFILES:
            target = targets.get(load_profile)
            if not isinstance(target, Mapping):
                _error(
                    f"workload {workload_name!r} {chip_key}: target "
                    f"{load_profile!r} must be a JSON object",
                    errors,
                )
                continue
            _validate_target(
                workload_name,
                chip_key,
                load_profile,
                target,
                repo_root,
                errors,
            )
        return

    cell_status = str(cell.get("status") or "")
    if cell_status not in VALID_STATUS:
        _error(
            f"workload {workload_name!r} {chip_key}: invalid status "
            f"{cell_status!r}, expected one of {VALID_STATUS}",
            errors,
        )
        return

    if cell_status == "blocked":
        if not cell.get("blocker_reason"):
            _error(
                f"workload {workload_name!r} {chip_key}: blocked cell is missing "
                "'blocker_reason'",
                errors,
            )
        return

    _validate_spec(workload_name, chip_key, cell.get("spec"), repo_root, errors)


def _validate_target(
    workload_name: str,
    chip_key: str,
    load_profile: str,
    target: Mapping[str, Any],
    repo_root: Path,
    errors: list[str],
) -> None:
    status = str(target.get("status") or "")
    if status not in VALID_TARGET_STATUS:
        _error(
            f"workload {workload_name!r} {chip_key} {load_profile}: invalid "
            f"target status {status!r}, expected one of {VALID_TARGET_STATUS}",
            errors,
        )
        return
    if load_profile == "fixed-1-rps" and status != "target-ready":
        _error(
            f"workload {workload_name!r} {chip_key}: fixed-1-rps target must be "
            "target-ready",
            errors,
        )
    if status == "capacity-pilot-pending":
        if target.get("spec") is not None or target.get("request_rate") is not None:
            _error(
                f"workload {workload_name!r} {chip_key} {load_profile}: pending "
                "target must not declare a spec or request_rate",
                errors,
            )
        return
    request_rate = target.get("request_rate")
    if request_rate is not None and not _is_valid_rate(request_rate):
        _error(
            f"workload {workload_name!r} {chip_key} {load_profile}: "
            "request_rate must be finite and greater than zero",
            errors,
        )
    _validate_spec(workload_name, chip_key, target.get("spec"), repo_root, errors)


def _validate_spec(
    workload_name: str,
    chip_key: str,
    spec_rel: Any,
    repo_root: Path,
    errors: list[str],
) -> None:
    if not spec_rel:
        _error(
            f"workload {workload_name!r} {chip_key}: ready target is missing 'spec'",
            errors,
        )
        return

    spec_path = repo_root / str(spec_rel)
    if not spec_path.is_file():
        _error(
            f"workload {workload_name!r} {chip_key}: spec file not found: {spec_rel}",
            errors,
        )
        return

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _error(
            f"workload {workload_name!r} {chip_key}: spec file unreadable: "
            f"{spec_rel} ({exc})",
            errors,
        )
        return

    expected_chip = _chip_count(chip_key)
    if not isinstance(spec, Mapping):
        _error(
            f"workload {workload_name!r} {chip_key}: spec must be a JSON object",
            errors,
        )
        return
    if int(spec.get("chip_count", -1)) != expected_chip:
        _error(
            f"workload {workload_name!r} {chip_key}: spec chip_count "
            f"{spec.get('chip_count')!r} does not match {chip_key!r}",
            errors,
        )
    server_parameters = spec.get("server_parameters")
    if not isinstance(server_parameters, Mapping):
        _error(
            f"workload {workload_name!r} {chip_key}: spec is missing required "
            "'server_parameters'",
            errors,
        )
        return
    tensor_parallel_size = server_parameters.get("tensor_parallel_size")
    if tensor_parallel_size is None:
        _error(
            f"workload {workload_name!r} {chip_key}: spec server_parameters is "
            "missing required 'tensor_parallel_size'",
            errors,
        )
    elif int(tensor_parallel_size) != expected_chip:
        _error(
            f"workload {workload_name!r} {chip_key}: spec server "
            f"tensor_parallel_size {tensor_parallel_size!r} does not match "
            f"{chip_key!r}",
            errors,
        )

    client_parameters = spec.get("client_parameters")
    if not isinstance(client_parameters, Mapping):
        _error(
            f"workload {workload_name!r} {chip_key}: spec is missing required "
            "'client_parameters'",
            errors,
        )
        return
    request_rate = client_parameters.get("request_rate")
    if request_rate is not None and not _is_valid_rate(request_rate):
        _error(
            f"workload {workload_name!r} {chip_key}: spec client request_rate "
            "must be finite and greater than zero",
            errors,
        )

    if workload_name == COMMUNICATION_WORKLOAD:
        _validate_communication_spec(
            chip_key,
            spec,
            client_parameters,
            errors,
        )


def _validate_communication_spec(
    chip_key: str,
    spec: Mapping[str, Any],
    client: Mapping[str, Any],
    errors: list[str],
) -> None:
    prefix = f"workload {COMMUNICATION_WORKLOAD!r} {chip_key}"
    expected_client = {
        "dataset_name": "random",
        "input_len": COMMUNICATION_INPUT_LEN,
        "output_len": COMMUNICATION_OUTPUT_LEN,
        "random_range_ratio": 0.0,
        "ignore_eos": True,
        "temperature": 0,
        "seed": 0,
    }
    if spec.get("scenario") != "random-online":
        _error(f"{prefix}: scenario must be 'random-online'", errors)
    for key, expected in expected_client.items():
        if client.get(key) != expected:
            _error(
                f"{prefix}: client {key} must be {expected!r}, got {client.get(key)!r}",
                errors,
            )

    contract = spec.get("issue_136_contract")
    if not isinstance(contract, Mapping):
        _error(f"{prefix}: spec is missing issue_136_contract", errors)
        return
    expected_role = "no-cross-rank-control" if chip_key == "1chip" else "tp-collective"
    expected_contract = {
        "workload_class": COMMUNICATION_PROFILE,
        "input_len": COMMUNICATION_INPUT_LEN,
        "output_len": COMMUNICATION_OUTPUT_LEN,
        "ignore_eos": True,
        "random_range_ratio": 0.0,
        "seed": 0,
        "tp_role": expected_role,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            _error(
                f"{prefix}: issue_136_contract.{key} must be {expected!r}, "
                f"got {contract.get(key)!r}",
                errors,
            )
    if not _is_valid_rate(contract.get("request_rate")):
        _error(
            f"{prefix}: issue_136_contract.request_rate must be finite and "
            "greater than zero",
            errors,
        )


def _check_identity(
    workload: Mapping[str, Any],
    workload_name: str,
    identity: dict[str, str],
    errors: list[str],
) -> None:
    """Collect each workload's frozen-stack identity and enforce consistency."""
    for field in IDENTITY_FIELDS:
        value = workload.get(field)
        if value in (None, ""):
            _error(
                f"workload {workload_name!r}: missing required identity field "
                f"{field!r}",
                errors,
            )
            continue
        value = str(value)
        if field in identity:
            if identity[field] != value:
                _error(
                    f"workload {workload_name!r}: identity field {field!r} "
                    f"{value!r} differs from other workloads "
                    f"({identity[field]!r})",
                    errors,
                )
        else:
            identity[field] = value


def _validate_workload_set(
    seen: set[str],
    errors: list[str],
) -> None:
    """Ensure the workload set is exactly the expected workloads."""
    expected = set(EXPECTED_WORKLOADS)
    actual = set(seen)
    for missing in sorted(expected - actual):
        _error(f"missing required workload {missing!r}", errors)
    for unknown in sorted(actual - expected):
        _error(f"unknown workload {unknown!r}", errors)


def _require_status(
    workload_name: str,
    cells: Mapping[str, Any],
    expected: str,
    errors: list[str],
) -> None:
    for chip_key in CHIP_KEYS:
        cell = cells.get(chip_key)
        if not isinstance(cell, Mapping):
            continue
        cell_status = str(cell.get("status") or "")
        if cell_status != expected:
            _error(
                f"workload {workload_name!r} {chip_key}: expected status "
                f"{expected!r}, got {cell_status!r}",
                errors,
            )


def _require_core_ready(
    workload_name: str,
    cells: Mapping[str, Any],
    errors: list[str],
) -> None:
    for chip_key in CHIP_KEYS:
        cell = cells.get(chip_key)
        if not isinstance(cell, Mapping):
            continue
        targets = cell.get("targets")
        if isinstance(targets, Mapping):
            fixed = targets.get("fixed-1-rps")
            if not isinstance(fixed, Mapping) or fixed.get("status") != "target-ready":
                _error(
                    f"workload {workload_name!r} {chip_key}: fixed-1-rps target "
                    "must be target-ready",
                    errors,
                )
            continue
        _require_status(workload_name, {chip_key: cell}, "spec-ready", errors)


def _require_communication_targets(
    workload_name: str,
    cells: Mapping[str, Any],
    errors: list[str],
) -> None:
    for chip_key in CHIP_KEYS:
        cell = cells.get(chip_key)
        if not isinstance(cell, Mapping) or not isinstance(
            cell.get("targets"), Mapping
        ):
            _error(
                f"workload {workload_name!r} {chip_key}: v3 requires fixed/scaled "
                "targets",
                errors,
            )


def _check_declared_count(
    status: Mapping[str, Any],
    key: str,
    computed: int,
    errors: list[str],
) -> None:
    declared = status.get(key)
    if declared is not None and int(declared) != computed:
        _error(f"status.{key} declares {declared!r} but computed {computed}", errors)


def _chip_count(chip_key: str) -> int:
    return int(chip_key[: -len("chip")])


def _is_valid_rate(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _error(message: str, errors: list[str]) -> None:
    errors.append(message)
