"""Validate the canonical Dataset Matrix publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


INDEX_FILE = "dataset_validation_index_v1.json"
CHECKSUM_FILE = "SHA256SUMS"
RESULT_STATUSES = {
    "not_tested",
    "baseline_only",
    "queued",
    "running",
    "passed",
    "failed",
    "not_applicable",
}


class DatasetValidationError(ValueError):
    """A dataset-validation publication invariant was violated."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetValidationError(f"JSON document must be an object: {path}")
    return payload


def _unique_ids(items: object, *, field: str, context: str) -> set[str]:
    if not isinstance(items, list) or not items:
        raise DatasetValidationError(f"{context} must be a non-empty array")
    values: set[str] = set()
    for item in items:
        value = item.get(field) if isinstance(item, dict) else None
        if not isinstance(value, str) or not value or value in values:
            raise DatasetValidationError(f"invalid or duplicate {context} {field}: {value}")
        values.add(value)
    return values


def validate_index(index: dict[str, Any]) -> list[dict[str, Any]]:
    if index.get("contract_version") != "dataset-validation-index-v1":
        raise DatasetValidationError("unsupported dataset-validation index contract")
    scenarios = index.get("scenarios")
    scenario_ids = _unique_ids(scenarios, field="id", context="scenario")
    if index.get("default_scenario_id") not in scenario_ids:
        raise DatasetValidationError("default_scenario_id is not declared")

    files: set[str] = set()
    for scenario in scenarios:
        data_file = scenario.get("data_file")
        if (
            not isinstance(data_file, str)
            or not data_file.endswith(".json")
            or Path(data_file).name != data_file
            or data_file == INDEX_FILE
            or data_file in files
        ):
            raise DatasetValidationError(f"invalid or duplicate scenario data_file: {data_file}")
        files.add(data_file)
    return scenarios


def validate_artifact(payload: dict[str, Any], *, expected_scenario_id: str) -> None:
    if payload.get("contract_version") != "dataset-validation-v1":
        raise DatasetValidationError("unsupported dataset-validation artifact contract")
    scenario = payload.get("scenario")
    if not isinstance(scenario, dict) or scenario.get("id") != expected_scenario_id:
        raise DatasetValidationError(
            f"artifact scenario does not match index: expected {expected_scenario_id}"
        )

    dataset_ids = _unique_ids(payload.get("datasets"), field="id", context="dataset")
    metric_ids = _unique_ids(payload.get("metrics"), field="id", context="metric")
    results = payload.get("results")
    if not isinstance(results, list):
        raise DatasetValidationError("results must be an array")

    cells: set[tuple[str, str]] = set()
    for result in results:
        if not isinstance(result, dict):
            raise DatasetValidationError("result cell must be an object")
        dataset_id = result.get("dataset_id")
        metric_id = result.get("metric_id")
        cell = (dataset_id, metric_id)
        if dataset_id not in dataset_ids or metric_id not in metric_ids:
            raise DatasetValidationError(f"result references an undeclared dimension: {cell}")
        if cell in cells:
            raise DatasetValidationError(f"duplicate result cell: {cell}")
        cells.add(cell)
        if result.get("status") not in RESULT_STATUSES:
            raise DatasetValidationError(f"unsupported result status in {cell}")
        value = result.get("current_value")
        if value is None:
            value = result.get("value")
        if value is not None:
            provenance = result.get("provenance")
            if not isinstance(provenance, dict) or not provenance.get("repository"):
                raise DatasetValidationError(f"populated B1 cell lacks provenance: {cell}")
            if not (provenance.get("artifact") or provenance.get("report_url")):
                raise DatasetValidationError(f"populated B1 cell lacks evidence URL: {cell}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(root: Path, expected_files: set[str]) -> None:
    checksum_path = root / CHECKSUM_FILE
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetValidationError(f"missing {CHECKSUM_FILE}") from exc

    recorded: dict[str, str] = {}
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise DatasetValidationError(f"invalid checksum line: {line}")
        digest, name = parts[0], parts[1].lstrip("*")
        if len(digest) != 64 or name in recorded or Path(name).name != name:
            raise DatasetValidationError(f"invalid checksum entry: {line}")
        recorded[name] = digest
    if set(recorded) != expected_files:
        raise DatasetValidationError("SHA256SUMS does not cover the exact JSON publication set")
    for name, digest in recorded.items():
        if _sha256(root / name) != digest:
            raise DatasetValidationError(f"checksum mismatch: {name}")


def validate_publication(root: Path) -> dict[str, int]:
    root = root.resolve()
    index = _load_json(root / INDEX_FILE)
    scenarios = validate_index(index)
    expected_files = {INDEX_FILE}
    result_count = 0
    for scenario in scenarios:
        data_file = scenario["data_file"]
        expected_files.add(data_file)
        artifact = _load_json(root / data_file)
        validate_artifact(artifact, expected_scenario_id=scenario["id"])
        result_count += len(artifact["results"])

    actual_files = {path.name for path in root.glob("*.json")}
    if actual_files != expected_files:
        raise DatasetValidationError(
            "publication JSON files differ from index: "
            f"expected={sorted(expected_files)} actual={sorted(actual_files)}"
        )
    verify_checksums(root, expected_files)
    return {"scenarios": len(scenarios), "results": result_count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("leaderboard-data/dataset-validation"),
    )
    args = parser.parse_args()
    try:
        summary = validate_publication(args.root)
    except DatasetValidationError as exc:
        parser.error(str(exc))
    print(
        "dataset-validation publication valid: "
        f"{summary['scenarios']} scenarios, {summary['results']} result cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
