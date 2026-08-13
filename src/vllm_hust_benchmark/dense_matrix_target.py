"""Validate the fixed Dense 1/2/4 matrix target (Issue #136).

This module loads ``leaderboard-data/dense-matrix-issue-136.json`` and checks
that the fixed Dense 1/2/4 matrix target is internally consistent:

* every ``spec-ready`` cell's spec file exists and its ``chip_count`` /
  ``server_parameters.tensor_parallel_size`` match the declared chip key;
* every ``blocked`` cell carries a ``blocker_reason``;
* every workload has exactly the ``1chip`` / ``2chip`` / ``4chip`` cells;
* cell status is a legal value (``spec-ready`` / ``blocked``);
* the four core workloads are fully ``spec-ready``;
* the ``communication-sensitive`` cells are all ``blocked``.

This module only fixes the matrix target/config/provenance. It never computes
or publishes performance percentages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "dense-matrix-issue-136/v1"
VALID_STATUS: tuple[str, ...] = ("spec-ready", "blocked")
CHIP_KEYS: tuple[str, ...] = ("1chip", "2chip", "4chip")
CORE_WORKLOADS: tuple[str, ...] = (
    "random-online",
    "sharegpt-online",
    "prefix-repetition-online",
    "agent-research-online",
)
COMMUNICATION_WORKLOAD = "communication-sensitive"
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

    for index, workload in enumerate(workloads):
        if not isinstance(workload, Mapping):
            _error(f"workloads[{index}] must be a JSON object", errors)
            continue
        workload_name = str(workload.get("workload") or "")
        cells = workload.get("cells")
        if not isinstance(cells, Mapping):
            _error(f"workload {workload_name!r}: 'cells' must be a JSON object", errors)
            continue

        _validate_cells(workload_name, cells, repo_root, errors)

        for chip_key in CHIP_KEYS:
            cell = cells.get(chip_key)
            if not isinstance(cell, Mapping):
                continue
            cell_status = str(cell.get("status") or "")
            if cell_status == "spec-ready":
                spec_ready_count += 1
            elif cell_status == "blocked":
                blocked_count += 1

        if workload_name in CORE_WORKLOADS:
            _require_status(workload_name, cells, "spec-ready", errors)
        if workload_name == COMMUNICATION_WORKLOAD:
            _require_status(workload_name, cells, "blocked", errors)

    _check_declared_count(status, "spec_ready_cells", spec_ready_count, errors)
    _check_declared_count(status, "blocked_cells", blocked_count, errors)

    return DenseMatrixStatus(
        schema_version=str(payload["schema_version"]),
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
    if str(payload.get("schema_version") or "") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION!r}, got "
            f"{str(payload.get('schema_version') or '')!r}"
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

    spec_rel = cell.get("spec")
    if not spec_rel:
        _error(
            f"workload {workload_name!r} {chip_key}: spec-ready cell is missing 'spec'",
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
    if isinstance(server_parameters, Mapping):
        tensor_parallel_size = server_parameters.get("tensor_parallel_size")
        if (
            tensor_parallel_size is not None
            and int(tensor_parallel_size) != expected_chip
        ):
            _error(
                f"workload {workload_name!r} {chip_key}: spec server "
                f"tensor_parallel_size {tensor_parallel_size!r} does not match "
                f"{chip_key!r}",
                errors,
            )


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


def _error(message: str, errors: list[str]) -> None:
    errors.append(message)
