#!/usr/bin/env python3
"""Generate a fail-closed PPT summary for issue 214 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "issue214-ppt-safe-summary/v1"
RAW_SCHEMA_VERSION = "issue214-ppt-safe-summary/v2"
MANIFEST_VERSION = "issue214-ppt-safe-manifest/v1"
RAW_MANIFEST_VERSION = "issue214-ppt-safe-manifest/v2"
OUTPUT_STEM = "issue214_ppt_safe_summary"
FORBIDDEN_NAMES = ("historical", "frontier", "历史")
MIN_REPEATS = 3
HIGH_IQR_PERCENT = 10.0
EPHEMERAL_PARAMETER_KEYS = {"host", "port", "model", "dataset_path"}
FULL_ARCHIVE_REQUIRED_FILES = {
    "raw_benchmark_result.json",
    "resolved_same_spec.json",
    "runner.log",
    "runtime-contract.json",
    "input-identity.json",
    "submission/STATUS",
    "submission/checksums.sha256",
    "submission/env-manifest.json",
    "submission/leaderboard_manifest.json",
    "submission/pip-packages.json",
    "submission/run_leaderboard.json",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile, matching NumPy's default."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compute_stats(values: Sequence[float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("statistics require at least one value")
    normalized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("statistics require finite numeric values")
    q1 = percentile(normalized, 0.25)
    q3 = percentile(normalized, 0.75)
    return {
        "n": len(normalized),
        "median": statistics.median(normalized),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def _validate_label(label: Any, field: str) -> str:
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{field} must be a non-empty string")
    lowered = label.casefold()
    if any(word in lowered for word in FORBIDDEN_NAMES):
        raise ValueError(f"{field} contains forbidden naming")
    return label


def _validate_commits(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    commits: dict[str, str] = {}
    for name in ("core", "plugin"):
        commit = value.get(name)
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdefABCDEF" for character in commit)
        ):
            raise ValueError(f"{field}.{name} must be a full 40-character Git commit")
        commits[name] = commit
    return commits


def _numeric_values(value: Any, field: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    values: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field} contains a non-numeric value")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field} contains a non-finite value")
        values.append(number)
    return values


def _resolve_path(value: Any, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _submission_dir(repeat_dir: Path) -> Path:
    nested = repeat_dir / "submission"
    return nested if nested.is_dir() else repeat_dir


def _verify_checksum_manifest(
    base: Path,
    checksum_path: Path,
    required_coverage: set[str],
    *,
    exact_coverage: set[str] | None = None,
) -> list[str]:
    if not checksum_path.is_file():
        return [f"{checksum_path.name} is missing"]

    errors: list[str] = []
    covered: set[str] = set()
    resolved_base = base.resolve()
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in parts[0])
        ):
            errors.append(f"invalid checksum line {line_number}")
            continue
        expected, relative_name = parts
        relative_name = relative_name.lstrip("*")
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"unsafe checksum path on line {line_number}")
            continue
        normalized_name = relative_path.as_posix().removeprefix("./")
        if normalized_name in covered:
            errors.append(f"duplicate checksum path: {relative_name}")
            continue
        target = base / relative_path
        if (
            target.is_symlink()
            or not target.resolve().is_relative_to(resolved_base)
            or not target.is_file()
        ):
            errors.append(f"checksummed file is missing: {relative_name}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected.casefold():
            errors.append(f"checksum mismatch: {relative_name}")
        covered.add(normalized_name)
    for required_name in sorted(required_coverage - covered):
        errors.append(f"{required_name} is not covered by checksums")
    if exact_coverage is not None:
        for extra_name in sorted(covered - exact_coverage):
            errors.append(f"unexpected checksummed file: {extra_name}")
        for missing_name in sorted(exact_coverage - covered):
            errors.append(f"archive file is not covered by checksums: {missing_name}")
    return errors


def _verify_checksums(submission: Path) -> list[str]:
    required = {"run_leaderboard.json"}
    if (submission / "input_provenance.json").is_file():
        required.add("input_provenance.json")
    return _verify_checksum_manifest(
        submission, submission / "checksums.sha256", required
    )


def _verify_full_archive(repeat_dir: Path, *, online: bool) -> list[str]:
    required = set(FULL_ARCHIVE_REQUIRED_FILES)
    if online:
        required.add("server.stdout.log")
    unsafe_files = [
        path
        for path in repeat_dir.rglob("*")
        if path.is_symlink()
        or (path.is_file() and not path.resolve().is_relative_to(repeat_dir.resolve()))
    ]
    if unsafe_files:
        return [
            f"unsafe archive path: {path.relative_to(repeat_dir)}"
            for path in unsafe_files
        ]
    exact = {
        path.relative_to(repeat_dir).as_posix()
        for path in repeat_dir.rglob("*")
        if path.is_file() and path.name != "EVIDENCE_SHA256SUMS"
    }
    return _verify_checksum_manifest(
        repeat_dir,
        repeat_dir / "EVIDENCE_SHA256SUMS",
        required,
        exact_coverage=exact,
    )


def _normalized_parameters(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalized_parameters(item)
            for key, item in sorted(value.items())
            if key not in EPHEMERAL_PARAMETER_KEYS
        }
    if isinstance(value, list):
        return [_normalized_parameters(item) for item in value]
    return value


def _portable_identity_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _portable_identity_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_identity_value(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return "<absolute-path-omitted>"
    return value


def _runtime_environment(contract: Any) -> Any:
    if not isinstance(contract, dict):
        return contract
    environment = {
        key: contract.get(key)
        for key in ("cann", "python", "torch", "torch_npu")
        if key in contract
    }
    environment["image"] = contract.get("image_id") or contract.get("image_digest")
    return environment


def identity_projection(
    run: dict[str, Any], input_provenance: dict[str, Any] | None = None
) -> dict[str, Any]:
    workload = run.get("workload", {})
    model = run.get("model", {})
    hardware = run.get("hardware", {})
    same_spec = run.get("same_spec", {})
    return {
        "workload": {
            key: workload.get(key)
            for key in (
                "name",
                "input_length",
                "output_length",
                "batch_size",
                "concurrent_requests",
                "dataset",
            )
        },
        "model": {
            key: model.get(key)
            for key in ("canonical_id", "parameters", "precision", "quantization")
        },
        "hardware": {
            key: hardware.get(key) for key in ("vendor", "chip_model", "chip_count")
        },
        "same_spec": {
            key: same_spec.get(key)
            for key in (
                "scenario",
                "model",
                "model_parameters",
                "model_precision",
                "model_quantization",
                "hardware_vendor",
                "hardware_chip_model",
                "chip_count",
                "node_count",
            )
        }
        | {
            "resolved_server_parameters": _normalized_parameters(
                same_spec.get("resolved_server_parameters", {})
            ),
            "resolved_client_parameters": _normalized_parameters(
                same_spec.get("resolved_client_parameters", {})
            ),
        },
        "input_provenance": _portable_identity_value(input_provenance),
    }


def _raw_evidence(
    side: str,
    workload: str,
    config: dict[str, Any],
    manifest_dir: Path,
    expected_commits: dict[str, str],
    *,
    require_full_archive: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    metric = config.get("primary_metric")
    repeat_values = config.get("repeat_dirs")
    blockers: list[str] = []
    if not isinstance(metric, str) or not metric:
        blockers.append(f"{side} primary_metric is missing")
    if not isinstance(repeat_values, list):
        blockers.append(f"{side} repeat_dirs must be an array")
        repeat_values = []
    if len(repeat_values) < MIN_REPEATS:
        blockers.append(f"{side} has fewer than {MIN_REPEATS} repeat directories")

    repeat_dirs: list[tuple[Path, str]] = []
    for index, value in enumerate(repeat_values):
        try:
            repeat_dirs.append(
                (
                    _resolve_path(value, manifest_dir, f"repeat_dirs[{index}]"),
                    str(value),
                )
            )
        except ValueError as error:
            blockers.append(str(error))
    if len({str(path.resolve()) for path, _ in repeat_dirs}) != len(repeat_dirs):
        blockers.append(f"{side} repeat directories are not unique")

    raw_values: list[float] = []
    projections: list[dict[str, Any]] = []
    runtime_contracts: list[dict[str, Any]] = []
    input_identities: list[dict[str, Any]] = []
    accepted_dirs: list[str] = []
    for repeat_dir, declared_repeat_dir in repeat_dirs:
        submission = _submission_dir(repeat_dir)
        if not submission.is_dir():
            blockers.append(f"repeat directory is missing: {repeat_dir}")
            continue
        if require_full_archive:
            archive_errors = _verify_full_archive(
                repeat_dir, online=config.get("execution_mode") == "online"
            )
            if archive_errors:
                blockers.extend(f"{repeat_dir}: {error}" for error in archive_errors)
                continue
        status_path = submission / "STATUS"
        if (
            not status_path.is_file()
            or status_path.read_text(encoding="utf-8").strip() != "OK"
        ):
            blockers.append(f"repeat STATUS is not OK: {repeat_dir}")
            continue
        checksum_errors = _verify_checksums(submission)
        if checksum_errors:
            blockers.extend(f"{repeat_dir}: {error}" for error in checksum_errors)
            continue
        run_path = submission / "run_leaderboard.json"
        try:
            run = load_json(run_path)
        except (OSError, json.JSONDecodeError) as error:
            blockers.append(f"invalid run_leaderboard.json in {repeat_dir}: {error}")
            continue
        if not isinstance(run, dict):
            blockers.append(f"run_leaderboard.json is not an object: {repeat_dir}")
            continue
        if run.get("workload", {}).get("name") != workload:
            blockers.append(f"workload identity mismatch: {repeat_dir}")
            continue
        runtime_provenance = run.get("metadata", {}).get("runtime_provenance", {})
        observed_commits = {
            "core": runtime_provenance.get("engine", {}).get("commit"),
            "plugin": runtime_provenance.get("plugin", {}).get("commit"),
        }
        if observed_commits != expected_commits:
            blockers.append(f"{side} commits do not match: {repeat_dir}")
            continue
        value = run.get("metrics", {}).get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            blockers.append(f"primary metric {metric} is missing: {repeat_dir}")
            continue
        number = float(value)
        if not math.isfinite(number):
            blockers.append(f"primary metric {metric} is non-finite: {repeat_dir}")
            continue
        raw_values.append(number)
        input_provenance_path = submission / "input_provenance.json"
        try:
            input_provenance = (
                load_json(input_provenance_path)
                if input_provenance_path.is_file()
                else None
            )
        except (OSError, json.JSONDecodeError) as error:
            blockers.append(f"invalid input_provenance.json in {repeat_dir}: {error}")
            raw_values.pop()
            continue
        if input_provenance is not None and not isinstance(input_provenance, dict):
            blockers.append(f"input_provenance.json is not an object: {repeat_dir}")
            raw_values.pop()
            continue
        projections.append(identity_projection(run, input_provenance))
        if require_full_archive:
            try:
                resolved_spec = load_json(repeat_dir / "resolved_same_spec.json")
                runtime_contract = load_json(repeat_dir / "runtime-contract.json")
                input_identity = load_json(repeat_dir / "input-identity.json")
            except (OSError, json.JSONDecodeError) as error:
                blockers.append(
                    f"invalid full archive metadata in {repeat_dir}: {error}"
                )
                raw_values.pop()
                projections.pop()
                continue
            if not all(
                isinstance(value, dict)
                for value in (resolved_spec, runtime_contract, input_identity)
            ):
                blockers.append(f"full archive metadata is not an object: {repeat_dir}")
                raw_values.pop()
                projections.pop()
                continue
            if _normalized_parameters(resolved_spec) != _normalized_parameters(
                run.get("same_spec", {})
            ):
                blockers.append(
                    f"resolved spec does not match exported run: {repeat_dir}"
                )
                raw_values.pop()
                projections.pop()
                continue
            runtime_contracts.append(_portable_identity_value(runtime_contract))
            input_identities.append(_portable_identity_value(input_identity))
        accepted_dirs.append(declared_repeat_dir)

    if projections and any(
        projection != projections[0] for projection in projections[1:]
    ):
        blockers.append(f"{side} repeat identity projections differ")
    if runtime_contracts and any(
        item != runtime_contracts[0] for item in runtime_contracts[1:]
    ):
        blockers.append(f"{side} repeat runtime contracts differ")
    if input_identities and any(
        item != input_identities[0] for item in input_identities[1:]
    ):
        blockers.append(f"{side} repeat input identities differ")
    if len(raw_values) < MIN_REPEATS:
        blockers.append(f"{side} has fewer than {MIN_REPEATS} valid repeats")

    evidence: dict[str, Any] = {
        "raw_values": raw_values,
        "repeat_dirs": accepted_dirs,
        "stats": compute_stats(raw_values) if raw_values else None,
        "evidence_grade": (
            "blocked" if blockers else ("A" if require_full_archive else "B")
        ),
        "identity_projection": projections[0] if projections else None,
        "runtime_contract": runtime_contracts[0] if runtime_contracts else None,
        "input_identity": input_identities[0] if input_identities else None,
    }
    return evidence, blockers


def _candidate_evidence(
    workload: str,
    config: dict[str, Any],
    manifest_dir: Path,
    candidate_commits: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    return _raw_evidence("candidate", workload, config, manifest_dir, candidate_commits)


def _variability(stats: dict[str, Any] | None) -> tuple[str, float | None]:
    if not stats:
        return "unavailable", None
    median = stats["median"]
    relative_iqr = math.inf if median == 0 and stats["iqr"] else 0.0
    if median:
        relative_iqr = abs(stats["iqr"] / median) * 100.0
    return ("high" if relative_iqr >= HIGH_IQR_PERCENT else "normal"), relative_iqr


def _analyze_workload(
    workload: str,
    config: dict[str, Any],
    reference_workloads: dict[str, Any],
    values_key: str,
    reference_grade: str,
    manifest_dir: Path,
    candidate_commits: dict[str, str],
) -> dict[str, Any]:
    blockers: list[str] = []
    comparison_blockers: list[str] = []
    reference_config = reference_workloads.get(workload)
    candidate_metric = config.get("primary_metric")
    unit = config.get("unit")
    direction = config.get("direction")
    input_status = config.get("input_identity_status")

    if not isinstance(reference_config, dict):
        blockers.append("reference workload is missing")
        reference_config = {}
    reference_metric = reference_config.get("primary_metric")
    if reference_metric != candidate_metric:
        blockers.append(
            f"primary metric conflict: reference={reference_metric!r}, "
            f"candidate={candidate_metric!r}"
        )
    if not isinstance(unit, str) or not unit:
        blockers.append("candidate unit is missing")
    if direction not in {"lower_is_better", "higher_is_better"}:
        blockers.append("candidate direction is invalid")
    if not isinstance(input_status, str) or not input_status:
        blockers.append("input_identity_status must be explicit")

    try:
        summary_values = _numeric_values(
            reference_config.get(values_key), f"reference {workload}.{values_key}"
        )
    except ValueError as error:
        blockers.append(str(error))
        summary_values = []
    if len(summary_values) < MIN_REPEATS:
        blockers.append(f"reference has fewer than {MIN_REPEATS} summary values")

    candidate, candidate_blockers = _candidate_evidence(
        workload, config, manifest_dir, candidate_commits
    )
    blockers.extend(candidate_blockers)
    reference_stats = compute_stats(summary_values) if summary_values else None
    reference = {
        "summary_values": summary_values,
        "stats": reference_stats,
        "evidence_grade": reference_grade,
    }

    if input_status != "verified":
        comparison_blockers.append("input identity is not verified")
    if reference_grade != "A":
        comparison_blockers.append("reference raw evidence is not verified")
    if candidate["evidence_grade"] != "A":
        comparison_blockers.append("candidate full raw archive is not verified")
    if blockers:
        comparison_blockers.append("workload evidence is blocked")

    delta_percent: float | None = None
    if (
        not blockers
        and not comparison_blockers
        and reference_stats
        and candidate["stats"]
        and reference_stats["n"] >= MIN_REPEATS
        and candidate["stats"]["n"] >= MIN_REPEATS
        and reference_stats["median"] != 0
    ):
        delta_percent = (
            (candidate["stats"]["median"] - reference_stats["median"])
            / reference_stats["median"]
            * 100.0
        )

    reference_variability, reference_iqr_percent = _variability(reference_stats)
    candidate_variability, candidate_iqr_percent = _variability(candidate["stats"])
    high_sides = [
        side
        for side, state in (
            ("reference", reference_variability),
            ("candidate", candidate_variability),
        )
        if state == "high"
    ]
    notice = None
    if high_sides:
        notice = "High IQR requires cautious stability interpretation: " + ", ".join(
            high_sides
        )

    return {
        "workload": workload,
        "status": "blocked" if blockers else "ready",
        "primary_metric": candidate_metric,
        "unit": unit,
        "direction": direction,
        "input_identity_status": input_status,
        "reference": reference,
        "candidate": candidate,
        "delta_percent": delta_percent,
        "blockers": blockers,
        "comparison_blockers": comparison_blockers,
        "variability": {
            "reference": reference_variability,
            "reference_iqr_percent": reference_iqr_percent,
            "candidate": candidate_variability,
            "candidate_iqr_percent": candidate_iqr_percent,
        },
        "stability_notice": notice,
    }


def _analyze_raw_workload(
    workload: str,
    reference_config: dict[str, Any],
    candidate_config: dict[str, Any],
    manifest_dir: Path,
    reference_commits: dict[str, str],
    candidate_commits: dict[str, str],
) -> dict[str, Any]:
    blockers: list[str] = []
    comparison_blockers: list[str] = []
    reference_metric = reference_config.get("primary_metric")
    candidate_metric = candidate_config.get("primary_metric")
    unit = candidate_config.get("unit")
    direction = candidate_config.get("direction")
    if reference_metric != candidate_metric:
        blockers.append(
            f"primary metric conflict: reference={reference_metric!r}, "
            f"candidate={candidate_metric!r}"
        )
    if reference_config.get("unit") != unit:
        blockers.append("unit conflict between reference and candidate")
    if reference_config.get("direction") != direction:
        blockers.append("direction conflict between reference and candidate")
    if direction not in {"lower_is_better", "higher_is_better"}:
        blockers.append("candidate direction is invalid")

    reference, reference_blockers = _raw_evidence(
        "reference",
        workload,
        reference_config,
        manifest_dir,
        reference_commits,
        require_full_archive=True,
    )
    candidate, candidate_blockers = _raw_evidence(
        "candidate",
        workload,
        candidate_config,
        manifest_dir,
        candidate_commits,
        require_full_archive=True,
    )
    blockers.extend(reference_blockers)
    blockers.extend(candidate_blockers)

    for field, message in (
        ("identity_projection", "resolved workload/spec identity differs"),
        ("input_identity", "input identity differs"),
    ):
        if reference.get(field) != candidate.get(field):
            comparison_blockers.append(message)
    if _runtime_environment(reference.get("runtime_contract")) != _runtime_environment(
        candidate.get("runtime_contract")
    ):
        comparison_blockers.append("runtime environment differs")
    if blockers:
        comparison_blockers.append("workload evidence is blocked")

    delta_percent: float | None = None
    reference_stats = reference.get("stats")
    candidate_stats = candidate.get("stats")
    if (
        not blockers
        and not comparison_blockers
        and reference_stats
        and candidate_stats
        and reference_stats["median"] != 0
    ):
        delta_percent = (
            (candidate_stats["median"] - reference_stats["median"])
            / reference_stats["median"]
            * 100.0
        )

    reference_variability, reference_iqr_percent = _variability(reference_stats)
    candidate_variability, candidate_iqr_percent = _variability(candidate_stats)
    high_sides = [
        side
        for side, state in (
            ("reference", reference_variability),
            ("candidate", candidate_variability),
        )
        if state == "high"
    ]
    return {
        "workload": workload,
        "status": "blocked" if blockers else "ready",
        "primary_metric": candidate_metric,
        "unit": unit,
        "direction": direction,
        "input_identity_status": (
            "verified" if not comparison_blockers else "mismatch-or-unavailable"
        ),
        "reference": reference,
        "candidate": candidate,
        "delta_percent": delta_percent,
        "blockers": blockers,
        "comparison_blockers": comparison_blockers,
        "variability": {
            "reference": reference_variability,
            "reference_iqr_percent": reference_iqr_percent,
            "candidate": candidate_variability,
            "candidate_iqr_percent": candidate_iqr_percent,
        },
        "stability_notice": (
            "High IQR requires cautious stability interpretation: "
            + ", ".join(high_sides)
            if high_sides
            else None
        ),
    }


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    fields = [
        "workload",
        "status",
        "primary_metric",
        "unit",
        "direction",
        "input_identity_status",
        "reference_summary_values",
        "reference_n",
        "reference_median",
        "reference_q1",
        "reference_q3",
        "reference_iqr",
        "reference_evidence_grade",
        "candidate_raw_values",
        "candidate_n",
        "candidate_median",
        "candidate_q1",
        "candidate_q3",
        "candidate_iqr",
        "candidate_evidence_grade",
        "delta_percent",
        "stability_notice",
        "blockers",
        "comparison_blockers",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in result["workloads"]:
            reference_stats = row["reference"]["stats"] or {}
            candidate_stats = row["candidate"]["stats"] or {}
            reference_values = row["reference"].get(
                "raw_values", row["reference"].get("summary_values", [])
            )
            writer.writerow(
                {
                    "workload": row["workload"],
                    "status": row["status"],
                    "primary_metric": row["primary_metric"],
                    "unit": row["unit"],
                    "direction": row["direction"],
                    "input_identity_status": row["input_identity_status"],
                    "reference_summary_values": json.dumps(
                        reference_values, separators=(",", ":")
                    ),
                    **{
                        f"reference_{key}": reference_stats.get(key)
                        for key in ("n", "median", "q1", "q3", "iqr")
                    },
                    "reference_evidence_grade": row["reference"]["evidence_grade"],
                    "candidate_raw_values": json.dumps(
                        row["candidate"]["raw_values"], separators=(",", ":")
                    ),
                    **{
                        f"candidate_{key}": candidate_stats.get(key)
                        for key in ("n", "median", "q1", "q3", "iqr")
                    },
                    "candidate_evidence_grade": row["candidate"]["evidence_grade"],
                    "delta_percent": row["delta_percent"],
                    "stability_notice": row["stability_notice"],
                    "blockers": " | ".join(row["blockers"]),
                    "comparison_blockers": " | ".join(row["comparison_blockers"]),
                }
            )


def _format_number(value: Any) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    raw_pair = result["schema_version"] == "issue214-ppt-safe-summary/v2"
    lines = [
        "# Issue 214 PPT-safe summary",
        "",
        f"Reference: {result['reference']['label']} (`{result['reference']['commits']['core']}` / "
        f"`{result['reference']['commits']['plugin']}`)",
        "",
        f"Candidate: {result['candidate']['label']} (`{result['candidate']['commits']['core']}` / "
        f"`{result['candidate']['commits']['plugin']}`)",
        "",
        (
            "Both arrays are values read from validated full repeat archives."
            if raw_pair
            else "Reference arrays are summary values. Candidate arrays are values read from validated repeat artifacts."
        ),
        "",
        "| Workload | Metric | Reference median (IQR) | Candidate median (IQR) | Delta | Evidence | Status |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["workloads"]:
        reference_stats = row["reference"]["stats"] or {}
        candidate_stats = row["candidate"]["stats"] or {}
        reference_values = row["reference"].get(
            "raw_values", row["reference"].get("summary_values", [])
        )
        delta = (
            "N/A" if row["delta_percent"] is None else f"{row['delta_percent']:+.3f}%"
        )
        evidence = (
            f"{row['reference']['evidence_grade']}/{row['candidate']['evidence_grade']}"
        )
        lines.append(
            f"| {row['workload']} | {row['primary_metric']} ({row['unit']}) | "
            f"{_format_number(reference_stats.get('median'))} "
            f"({_format_number(reference_stats.get('iqr'))}) | "
            f"{_format_number(candidate_stats.get('median'))} "
            f"({_format_number(candidate_stats.get('iqr'))}) | {delta} | "
            f"{evidence} | {row['status']} |"
        )
        lines.extend(
            [
                "",
                f"Reference {'raw' if raw_pair else 'summary'} values: `{json.dumps(reference_values, separators=(',', ':'))}`",
                "",
                f"Candidate {'raw' if raw_pair else 'exported'} values: "
                f"`{json.dumps(row['candidate']['raw_values'], separators=(',', ':'))}`",
            ]
        )
        if row["stability_notice"]:
            lines.extend(["", f"Stability notice: {row['stability_notice']}"])
        if row["blockers"] or row["comparison_blockers"]:
            messages = row["blockers"] + row["comparison_blockers"]
            lines.extend(["", "Restrictions: " + " | ".join(messages)])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in {
        MANIFEST_VERSION,
        RAW_MANIFEST_VERSION,
    }:
        raise ValueError(
            "manifest schema_version must be "
            f"{MANIFEST_VERSION} or {RAW_MANIFEST_VERSION}"
        )

    reference_config = manifest.get("reference")
    candidate_config = manifest.get("candidate")
    if not isinstance(reference_config, dict) or not isinstance(candidate_config, dict):
        raise ValueError("manifest reference and candidate must be objects")
    reference_label = _validate_label(reference_config.get("label"), "reference.label")
    candidate_label = _validate_label(candidate_config.get("label"), "candidate.label")
    reference_commits = _validate_commits(
        reference_config.get("commits"), "reference.commits"
    )
    candidate_commits = _validate_commits(
        candidate_config.get("commits"), "candidate.commits"
    )
    if manifest["schema_version"] == RAW_MANIFEST_VERSION:
        reference_workloads = reference_config.get("workloads")
        candidate_workloads = candidate_config.get("workloads")
        if not isinstance(reference_workloads, dict) or not reference_workloads:
            raise ValueError("reference.workloads must be a non-empty object")
        if not isinstance(candidate_workloads, dict) or not candidate_workloads:
            raise ValueError("candidate.workloads must be a non-empty object")
        if set(reference_workloads) != set(candidate_workloads):
            raise ValueError("reference and candidate workloads must exactly match")
        rows = [
            _analyze_raw_workload(
                workload,
                reference_config=reference_workloads[workload],
                candidate_config=candidate_workloads[workload],
                manifest_dir=manifest_path.parent,
                reference_commits=reference_commits,
                candidate_commits=candidate_commits,
            )
            for workload in sorted(candidate_workloads)
        ]
        result = {
            "schema_version": RAW_SCHEMA_VERSION,
            "reference": {
                "label": reference_label,
                "commits": reference_commits,
                "evidence_grade": "A"
                if all(row["reference"]["evidence_grade"] == "A" for row in rows)
                else "blocked",
                "evidence_description": "validated full repeat archives",
            },
            "candidate": {
                "label": candidate_label,
                "commits": candidate_commits,
                "evidence_grade": "A"
                if all(row["candidate"]["evidence_grade"] == "A" for row in rows)
                else "blocked",
            },
            "quartile_method": "linear interpolation at 25% and 75%",
            "minimum_valid_repeats": MIN_REPEATS,
            "workloads": rows,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{OUTPUT_STEM}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_csv(output_dir / f"{OUTPUT_STEM}.csv", result)
        _write_markdown(output_dir / f"{OUTPUT_STEM}.md", result)
        return result

    values_key = reference_config.get("values_key")
    if not isinstance(values_key, str) or not values_key:
        raise ValueError("reference.values_key must be a non-empty string")

    artifact_path = _resolve_path(
        reference_config.get("artifact_suites"),
        manifest_path.parent,
        "reference.artifact_suites",
    )
    artifact_suites = load_json(artifact_path)
    if not isinstance(artifact_suites, dict):
        raise ValueError("reference artifact_suites must contain an object")
    source_pair = artifact_suites.get("pairs", {}).get("current")
    if source_pair != reference_commits:
        raise ValueError("reference commits do not match artifact_suites pairs.current")
    reference_workloads = artifact_suites.get("workloads")
    if not isinstance(reference_workloads, dict):
        raise ValueError("reference artifact_suites.workloads must be an object")
    metric_overlays = reference_config.get("metric_overlays", {})
    if not isinstance(metric_overlays, dict):
        raise ValueError("reference.metric_overlays must be an object")
    normalized_reference_workloads = {
        name: dict(config) if isinstance(config, dict) else config
        for name, config in reference_workloads.items()
    }
    applied_overlays: dict[str, Any] = {}
    for workload, overlay in metric_overlays.items():
        if workload not in normalized_reference_workloads or not isinstance(
            overlay, dict
        ):
            raise ValueError(f"invalid metric overlay for {workload}")
        source_metric = overlay.get("source_metric")
        canonical_metric = overlay.get("canonical_metric")
        reason = overlay.get("reason")
        if (
            workload != "random-latency"
            or source_metric != "ttft_ms"
            or canonical_metric != "batch_latency_ms"
            or normalized_reference_workloads[workload].get("primary_metric")
            != source_metric
            or not isinstance(reason, str)
            or not reason
        ):
            raise ValueError(f"invalid metric overlay contract for {workload}")
        normalized_reference_workloads[workload]["primary_metric"] = canonical_metric
        applied_overlays[workload] = overlay
    declared_evidence_root = artifact_suites.get("evidence_root")
    evidence_root = _resolve_path(
        declared_evidence_root, artifact_path.parent, "evidence_root"
    )
    # artifact_suites contains derived arrays, not validated raw repetitions. Merely
    # finding its referenced directory is not enough to promote those arrays to A.
    reference_grade = "B"
    reference_evidence_available = evidence_root.is_dir()

    workload_configs = candidate_config.get("workloads")
    if not isinstance(workload_configs, dict) or not workload_configs:
        raise ValueError("candidate.workloads must be a non-empty object")
    if set(workload_configs) != set(normalized_reference_workloads):
        missing = sorted(set(normalized_reference_workloads) - set(workload_configs))
        extra = sorted(set(workload_configs) - set(normalized_reference_workloads))
        raise ValueError(
            f"candidate workloads must exactly match reference workloads; "
            f"missing={missing}, extra={extra}"
        )
    rows = [
        _analyze_workload(
            workload,
            config if isinstance(config, dict) else {},
            normalized_reference_workloads,
            values_key,
            reference_grade,
            manifest_path.parent,
            candidate_commits,
        )
        for workload, config in sorted(workload_configs.items())
    ]

    result = {
        "schema_version": SCHEMA_VERSION,
        "reference": {
            "label": reference_label,
            "commits": reference_commits,
            "source": str(reference_config.get("artifact_suites")),
            "values_key": values_key,
            "evidence_root": (
                str(declared_evidence_root) if reference_evidence_available else None
            ),
            "evidence_grade": reference_grade,
            "referenced_evidence_directory_available": reference_evidence_available,
            "evidence_description": "summary arrays; raw repetitions are not validated",
            "metric_overlays": applied_overlays,
        },
        "candidate": {
            "label": candidate_label,
            "commits": candidate_commits,
        },
        "quartile_method": "linear interpolation at 25% and 75%",
        "minimum_valid_repeats": MIN_REPEATS,
        "workloads": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{OUTPUT_STEM}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / f"{OUTPUT_STEM}.csv", result)
    _write_markdown(output_dir / f"{OUTPUT_STEM}.md", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = generate(args.manifest, args.output_dir)
    blocked = sum(row["status"] == "blocked" for row in result["workloads"])
    print(
        f"Wrote {len(result['workloads'])} workload rows to {args.output_dir}; "
        f"blocked={blocked}"
    )


if __name__ == "__main__":
    main()
