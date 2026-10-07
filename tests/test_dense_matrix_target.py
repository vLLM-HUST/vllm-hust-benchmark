"""Tests for the fixed Dense 1/2/4 matrix target validator (Issue #136).

These tests build a scratch matrix definition and spec files under
``tmp_path`` and exercise ``validate_dense_matrix_target`` against both the
valid matrix and deliberately-broken variants.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from vllm_hust_benchmark.dense_matrix_target import (
    CHIP_KEYS,
    COMMUNICATION_WORKLOAD,
    CORE_WORKLOADS,
    LEGACY_SCHEMA_VERSION,
    PREVIOUS_SCHEMA_VERSION,
    SCHEMA_VERSION,
    VALID_STATUS,
    validate_dense_matrix_target,
)

CORE_CELLS = {
    "1chip": "spec-ready",
    "2chip": "spec-ready",
    "4chip": "spec-ready",
}
COMM_CELLS = {
    "1chip": "blocked",
    "2chip": "blocked",
    "4chip": "blocked",
}


def _spec_filename(scenario: str, chip_key: str) -> str:
    if chip_key == "1chip":
        return f"official-ascend-jan-2026-v0180-{scenario}-qwen25-14b-910b2.json"
    return f"official-ascend-jan-2026-v0180-{scenario}-qwen25-14b-{chip_key}-910b2.json"


def _spec(chip_count: int, scenario: str) -> dict:
    return {
        "id": f"official-ascend-jan-2026-v0.18.0-{scenario}-qwen25-14b-{chip_count}chip-910b2",
        "label": f"test {scenario} {chip_count}chip",
        "baseline_target": {},
        "scenario": scenario,
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "model_precision": "FP16",
        "hardware_chip_model": "910B2",
        "chip_count": chip_count,
        "node_count": 1,
        "server_parameters": {"tensor_parallel_size": chip_count},
        "client_parameters": {},
        "export": {},
    }


def _workload(
    name: str, cell_statuses: dict[str, str], spec_scenario: str | None = None
) -> dict:
    spec_scenario = spec_scenario or name
    cells: dict[str, dict] = {}
    for chip_key in CHIP_KEYS:
        status = cell_statuses[chip_key]
        if status == "spec-ready":
            cell: dict = {
                "spec": f"docs/official-baselines/{_spec_filename(spec_scenario, chip_key)}",
                "status": status,
                "load_profiles": ["fixed-1-rps", "matched-load"],
            }
        else:
            cell = {
                "status": status,
                "load_profiles": ["matched-load"],
                "blocker_reason": f"blocked {name} {chip_key}",
            }
        cells[chip_key] = cell
    return {
        "workload": name,
        "model": "Qwen2.5-14B-Instruct",
        "precision": "FP16",
        "model_revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",  # pragma: allowlist secret
        "engine_backend_commit": "e18643f8a4d5bd9990727654318ad069ea0b56e2",  # pragma: allowlist secret
        "node_topology": "single-node",
        "cells": cells,
    }


def _base_matrix(schema_version: str = LEGACY_SCHEMA_VERSION) -> dict:
    return {
        "schema_version": schema_version,
        "issue": "https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/136",
        "hardware": {"chip_model": "910B2", "node_count": 1},
        "model_revision_contract": "test contract",
        "load_profiles": {
            "fixed-1-rps": {"description": "fixed 1 RPS"},
            "matched-load": {"description": "matched load"},
        },
        "workloads": [
            _workload("random-online", CORE_CELLS),
            _workload("sharegpt-online", CORE_CELLS),
            _workload("prefix-repetition-online", CORE_CELLS),
            _workload("agent-research-online", CORE_CELLS),
            _workload(
                "communication-sensitive",
                COMM_CELLS,
                spec_scenario="unified-comm-online",
            ),
        ],
        "profiler": {"required": True, "profiler_script": "unused.sh", "note": ""},
        "evidence_rules": {
            "reps_per_cell": 3,
            "independent_services": "3 independent service processes per cell",
            "no_cross_stack_percentage": "do not compute",
            "no_cross_sha_anchor": "re-run after freeze",
            "missing_baseline": "mark blocked",
        },
        "status": {
            "overall": "matrix-target-fixed",
            "spec_ready_cells": 12,
            "blocked_cells": 3,
            "performance_percentages_published": False,
            "blockers": ["test blocker"],
        },
    }


def _write_specs(root: Path, matrix: dict) -> None:
    for workload in matrix["workloads"]:
        for chip_key, cell in workload["cells"].items():
            if cell["status"] != "spec-ready":
                continue
            spec_path = root / cell["spec"]
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            chip_count = int(chip_key[: -len("chip")])
            spec_path.write_text(
                json.dumps(_spec(chip_count, workload["workload"])), encoding="utf-8"
            )


def _setup(
    tmp_path: Path, mutate=None, *, schema_version=LEGACY_SCHEMA_VERSION
) -> Path:
    matrix = _base_matrix(schema_version)
    if mutate is not None:
        mutate(matrix)
    matrix_dir = tmp_path / "leaderboard-data"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = matrix_dir / "dense-matrix-issue-136.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    _write_specs(tmp_path, matrix)
    return matrix_path


def test_matrix_schema_loads(tmp_path: Path) -> None:
    matrix_path = _setup(tmp_path)
    status = validate_dense_matrix_target(matrix_path)
    assert status.schema_version == LEGACY_SCHEMA_VERSION
    assert status.overall == "matrix-target-fixed"
    assert list(status.errors) == []


def test_v2_matrix_keeps_blocked_communication_semantics(tmp_path: Path) -> None:
    matrix_path = _setup(tmp_path, schema_version=PREVIOUS_SCHEMA_VERSION)
    status = validate_dense_matrix_target(matrix_path)
    assert status.schema_version == PREVIOUS_SCHEMA_VERSION
    assert status.spec_ready_cells == 12
    assert status.blocked_cells == 3
    assert list(status.errors) == []


def test_repository_v3_matrix_loads() -> None:
    matrix_path = (
        Path(__file__).resolve().parents[1]
        / "leaderboard-data"
        / "dense-matrix-issue-136.json"
    )
    status = validate_dense_matrix_target(matrix_path)
    assert status.schema_version == SCHEMA_VERSION
    assert status.spec_ready_cells == 15
    assert status.blocked_cells == 0
    assert list(status.errors) == []


def test_core_workloads_all_spec_ready(tmp_path: Path) -> None:
    matrix_path = _setup(tmp_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    for workload in matrix["workloads"]:
        if workload["workload"] in CORE_WORKLOADS:
            for chip_key in CHIP_KEYS:
                assert workload["cells"][chip_key]["status"] == "spec-ready"


def test_spec_files_exist_for_spec_ready_cells(tmp_path: Path) -> None:
    matrix_path = _setup(tmp_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    for workload in matrix["workloads"]:
        for chip_key, cell in workload["cells"].items():
            if cell["status"] == "spec-ready":
                spec_path = tmp_path / cell["spec"]
                assert spec_path.is_file(), f"missing {cell['spec']}"
    status = validate_dense_matrix_target(matrix_path)
    assert list(status.errors) == []


@pytest.mark.parametrize(
    "schema_version", [LEGACY_SCHEMA_VERSION, PREVIOUS_SCHEMA_VERSION]
)
def test_legacy_communication_cells_all_blocked(
    tmp_path: Path, schema_version: str
) -> None:
    matrix_path = _setup(tmp_path, schema_version=schema_version)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    for workload in matrix["workloads"]:
        if workload["workload"] == COMMUNICATION_WORKLOAD:
            for chip_key in CHIP_KEYS:
                assert workload["cells"][chip_key]["status"] == "blocked"


def _copy_repository_v3_matrix(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "leaderboard-data" / "dense-matrix-issue-136.json"
    matrix = json.loads(source.read_text(encoding="utf-8"))
    destination = tmp_path / "leaderboard-data" / source.name
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(matrix), encoding="utf-8")
    for workload in matrix["workloads"]:
        for cell in workload["cells"].values():
            fixed = cell["targets"]["fixed-1-rps"]
            source_spec = repo_root / fixed["spec"]
            destination_spec = tmp_path / fixed["spec"]
            destination_spec.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_spec, destination_spec)
    return destination


def test_v3_rejects_blocked_communication_cell(tmp_path: Path) -> None:
    matrix_path = _copy_repository_v3_matrix(tmp_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    communication = next(
        item
        for item in matrix["workloads"]
        if item["workload"] == COMMUNICATION_WORKLOAD
    )
    communication["cells"]["1chip"] = {
        "status": "blocked",
        "blocker_reason": "legacy custom path",
    }
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    status = validate_dense_matrix_target(matrix_path)
    assert any("v3 requires fixed/scaled targets" in error for error in status.errors)


@pytest.mark.parametrize("invalid_rate", [math.nan, math.inf, -math.inf])
def test_v3_rejects_non_finite_spec_rates(tmp_path: Path, invalid_rate: float) -> None:
    matrix_path = _copy_repository_v3_matrix(tmp_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    communication = next(
        item
        for item in matrix["workloads"]
        if item["workload"] == COMMUNICATION_WORKLOAD
    )
    spec_path = (
        tmp_path / communication["cells"]["2chip"]["targets"]["fixed-1-rps"]["spec"]
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["client_parameters"]["request_rate"] = invalid_rate
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    status = validate_dense_matrix_target(matrix_path)
    assert any("finite and greater than zero" in error for error in status.errors)


def test_v3_rejects_communication_shape_drift(tmp_path: Path) -> None:
    matrix_path = _copy_repository_v3_matrix(tmp_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    communication = next(
        item
        for item in matrix["workloads"]
        if item["workload"] == COMMUNICATION_WORKLOAD
    )
    spec_path = (
        tmp_path / communication["cells"]["4chip"]["targets"]["fixed-1-rps"]["spec"]
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["client_parameters"]["output_len"] = 256
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    status = validate_dense_matrix_target(matrix_path)
    assert any("client output_len must be 1024" in error for error in status.errors)


def test_each_workload_has_three_cells(tmp_path: Path) -> None:
    matrix_path = _setup(tmp_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    for workload in matrix["workloads"]:
        assert set(workload["cells"].keys()) == set(CHIP_KEYS)


def test_status_values_are_legal(tmp_path: Path) -> None:
    matrix_path = _setup(tmp_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    for workload in matrix["workloads"]:
        for chip_key in CHIP_KEYS:
            assert workload["cells"][chip_key]["status"] in VALID_STATUS


def test_counts_match_status_block(tmp_path: Path) -> None:
    matrix_path = _setup(tmp_path)
    status = validate_dense_matrix_target(matrix_path)
    assert status.spec_ready_cells == 12
    assert status.blocked_cells == 3


def test_no_performance_percentages_published(tmp_path: Path) -> None:
    matrix_path = _setup(tmp_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert matrix["status"]["performance_percentages_published"] is False
    status = validate_dense_matrix_target(matrix_path)
    assert status.performance_percentages_published is False


def test_detects_missing_spec_file(tmp_path: Path) -> None:
    matrix_path = _setup(tmp_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    spec_path = tmp_path / matrix["workloads"][0]["cells"]["1chip"]["spec"]
    spec_path.unlink()

    status = validate_dense_matrix_target(matrix_path)
    assert any("spec file not found" in error for error in status.errors)


def test_detects_chip_count_mismatch(tmp_path: Path) -> None:
    matrix_path = _setup(tmp_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    spec_path = tmp_path / matrix["workloads"][0]["cells"]["2chip"]["spec"]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["chip_count"] = 4
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    status = validate_dense_matrix_target(matrix_path)
    assert any(
        "chip_count" in error and "does not match" in error for error in status.errors
    )


def test_detects_blocked_missing_blocker_reason(tmp_path: Path) -> None:
    def _mutate(matrix: dict) -> None:
        comm = next(
            w for w in matrix["workloads"] if w["workload"] == COMMUNICATION_WORKLOAD
        )
        del comm["cells"]["1chip"]["blocker_reason"]

    matrix_path = _setup(tmp_path, mutate=_mutate)
    status = validate_dense_matrix_target(matrix_path)
    assert any("blocker_reason" in error for error in status.errors)


def test_detects_missing_cell(tmp_path: Path) -> None:
    def _mutate(matrix: dict) -> None:
        del matrix["workloads"][0]["cells"]["4chip"]

    matrix_path = _setup(tmp_path, mutate=_mutate)
    status = validate_dense_matrix_target(matrix_path)
    assert any("missing required cell" in error for error in status.errors)


def test_detects_invalid_status(tmp_path: Path) -> None:
    def _mutate(matrix: dict) -> None:
        matrix["workloads"][0]["cells"]["1chip"]["status"] = "done"

    matrix_path = _setup(tmp_path, mutate=_mutate)
    status = validate_dense_matrix_target(matrix_path)
    assert any("invalid status" in error for error in status.errors)


def test_detects_core_workload_not_spec_ready(tmp_path: Path) -> None:
    def _mutate(matrix: dict) -> None:
        matrix["workloads"][0]["cells"]["1chip"]["status"] = "blocked"
        matrix["workloads"][0]["cells"]["1chip"]["blocker_reason"] = "blocked"

    matrix_path = _setup(tmp_path, mutate=_mutate)
    status = validate_dense_matrix_target(matrix_path)
    assert any("expected status 'spec-ready'" in error for error in status.errors)


def test_missing_matrix_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        validate_dense_matrix_target(tmp_path / "nope.json")


def test_detects_duplicate_workload(tmp_path: Path) -> None:
    def _mutate(matrix: dict) -> None:
        matrix["workloads"].append(matrix["workloads"][0])

    matrix_path = _setup(tmp_path, mutate=_mutate)
    status = validate_dense_matrix_target(matrix_path)
    assert any("duplicate workload" in error for error in status.errors)


def test_detects_unknown_workload(tmp_path: Path) -> None:
    def _mutate(matrix: dict) -> None:
        matrix["workloads"].append(_workload("unknown-workload", CORE_CELLS))

    matrix_path = _setup(tmp_path, mutate=_mutate)
    status = validate_dense_matrix_target(matrix_path)
    assert any("unknown workload" in error for error in status.errors)


def test_detects_missing_core_workload(tmp_path: Path) -> None:
    def _mutate(matrix: dict) -> None:
        matrix["workloads"] = [
            w for w in matrix["workloads"] if w["workload"] != "random-online"
        ]

    matrix_path = _setup(tmp_path, mutate=_mutate)
    status = validate_dense_matrix_target(matrix_path)
    assert any("missing required workload" in error for error in status.errors)


def test_detects_missing_identity_field(tmp_path: Path) -> None:
    def _mutate(matrix: dict) -> None:
        del matrix["workloads"][0]["model_revision"]

    matrix_path = _setup(tmp_path, mutate=_mutate)
    status = validate_dense_matrix_target(matrix_path)
    assert any("missing required identity" in error for error in status.errors)


def test_detects_identity_mismatch_across_workloads(tmp_path: Path) -> None:
    def _mutate(matrix: dict) -> None:
        matrix["workloads"][1]["model_revision"] = "0" * 40

    matrix_path = _setup(tmp_path, mutate=_mutate)
    status = validate_dense_matrix_target(matrix_path)
    assert any(
        "identity field 'model_revision'" in error and "differs" in error
        for error in status.errors
    )


def test_detects_missing_tensor_parallel_size(tmp_path: Path) -> None:
    matrix_path = _setup(tmp_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    spec_path = tmp_path / matrix["workloads"][0]["cells"]["1chip"]["spec"]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    del spec["server_parameters"]["tensor_parallel_size"]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    status = validate_dense_matrix_target(matrix_path)
    assert any(
        "missing required 'tensor_parallel_size'" in error for error in status.errors
    )
