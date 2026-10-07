"""Fail-closed publication builder for Issue #136 Dense matrix evidence.

The builder consumes the *publishable archives* produced by the Issue #136
evidence archiver.  It deliberately does not consume a live matrix directory
and it never mutates the target registry or committed snapshots.  A complete
fixed-load archive and a complete scaled-load archive are required before a
promotion bundle can be emitted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, Mapping, Sequence

from vllm_hust_benchmark.aggregate_results import (
    apply_aggregate_to_entry,
    compute_canonical_aggregate,
)


SCHEMA_VERSION = "issue-136-publication-bundle/v1"
ARCHIVE_INVENTORY_SCHEMA = "issue-136-evidence-archive-inventory/v1"
ANALYSIS_SCHEMA = "issue-136-formal-matrix-analysis/v1"
WORKLOADS = (
    "random-online",
    "sharegpt-online",
    "prefix-repetition-online",
    "agent-research-online",
)
TENSOR_PARALLEL_SIZES = (1, 2, 4)
LOAD_PROFILES = ("fixed-1-rps", "scaled-load")
REPEATS = (0, 1, 2)
REQUIRED_SUBMISSION_FILES = (
    "STATUS",
    "checksums.sha256",
    "env-manifest.json",
    "raw_benchmark_result.json",
    "resolved_same_spec.json",
    "run_leaderboard.json",
    "server.stdout.log",
)


class PublicationError(ValueError):
    """Evidence cannot be promoted without violating the publication contract."""


@dataclass(frozen=True)
class ValidatedArchive:
    root: Path
    load_profile: str
    archive_sha256: str
    source_commits: Mapping[str, str]
    runtime_provenance: Mapping[str, Any]
    entries: tuple[dict[str, Any], ...]
    cells: tuple[dict[str, Any], ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"invalid {label}: {path}") from exc


def _regular_file(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise PublicationError(f"unsafe archive path: {relative}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise PublicationError(f"missing regular archive file: {relative}")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise PublicationError(f"archive file escapes root: {relative}") from exc
    return path


def _parse_checksums(root: Path, relative: Path) -> dict[Path, str]:
    manifest = _regular_file(root, relative)
    entries: dict[Path, str] = {}
    for line_number, raw in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise PublicationError(f"malformed {relative}:{line_number}")
        try:
            int(parts[0], 16)
        except ValueError as exc:
            raise PublicationError(f"malformed {relative}:{line_number}") from exc
        posix = PurePosixPath(parts[1].removeprefix("*").removeprefix("./"))
        target_relative = Path(*posix.parts)
        target = _regular_file(root, target_relative)
        if target_relative in entries:
            raise PublicationError(f"duplicate checksum entry: {target_relative}")
        if sha256(target) != parts[0]:
            raise PublicationError(f"checksum mismatch: {target_relative}")
        entries[target_relative] = parts[0]
    if not entries:
        raise PublicationError(f"empty checksum manifest: {relative}")
    return entries


def _validate_archive_manifest(root: Path) -> str:
    if any(path.is_symlink() for path in root.rglob("*")):
        raise PublicationError("archive must not contain symbolic links")
    checksums = _parse_checksums(root, Path("SHA256SUMS"))
    present = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
    }
    if set(checksums) != present:
        raise PublicationError("archive SHA256SUMS does not cover exactly all files")
    return sha256(root / "SHA256SUMS")


def _cell_key(cell: Mapping[str, Any]) -> tuple[str, int]:
    workload = cell.get("workload")
    tp = cell.get("tensor_parallel_size", cell.get("tp"))
    if workload not in WORKLOADS or tp not in TENSOR_PARALLEL_SIZES:
        raise PublicationError(f"invalid matrix cell identity: {workload!r}/TP{tp!r}")
    return str(workload), int(tp)


def _numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PublicationError(f"{label} must be finite")
    return result


def _find_submission(root: Path, declared: str) -> Path:
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        raise PublicationError(f"unsafe submission path: {declared}")
    path = root / relative
    if path.is_symlink() or not path.is_dir():
        raise PublicationError(f"missing submission directory: {declared}")
    return path


def _validate_submission(
    archive_root: Path,
    submission: Path,
    *,
    cell: Mapping[str, Any],
    repeat: Mapping[str, Any],
    load_profile: str,
    source_commits: Mapping[str, str],
) -> tuple[dict[str, Any], float, int, dict[str, Any]]:
    if (submission / "STATUS").read_text(encoding="utf-8").strip() != "OK":
        raise PublicationError(f"submission is not OK: {submission}")
    checksums = _parse_checksums(submission, Path("checksums.sha256"))
    missing = {Path(name) for name in REQUIRED_SUBMISSION_FILES[2:]} - set(checksums)
    if missing:
        raise PublicationError(f"submission checksum coverage is incomplete: {missing}")

    raw = _load_json(submission / "raw_benchmark_result.json", "raw result")
    resolved = _load_json(submission / "resolved_same_spec.json", "resolved spec")
    environment = _load_json(submission / "env-manifest.json", "environment")
    entry = _load_json(submission / "run_leaderboard.json", "leaderboard entry")
    if not all(isinstance(item, dict) for item in (raw, resolved, environment, entry)):
        raise PublicationError(f"submission payload is not an object: {submission}")

    repeat_index = repeat.get("repeat_index")
    campaign = environment.get("campaign")
    git_info = environment.get("git_info")
    frozen = environment.get("frozen_inputs")
    if (
        repeat_index not in REPEATS
        or not isinstance(campaign, dict)
        or campaign.get("campaign_id") != "issue-136-current-main/v1"
        or campaign.get("coverage_class") != "full-matrix"
        or campaign.get("point_role") != "checkpoint"
        or campaign.get("repeat_index") != repeat_index
        or campaign.get("repetitions") != 3
        or campaign.get("load_profile") != load_profile
    ):
        raise PublicationError(f"campaign repetition mismatch: {submission}")
    expected_git = {
        "benchmark": source_commits.get("benchmark"),
        "vllm_hust": {
            "declared": source_commits.get("core"),
            "observed": source_commits.get("core"),
        },
        "vllm_ascend_hust": {
            "declared": source_commits.get("plugin"),
            "observed": source_commits.get("plugin"),
        },
    }
    if not isinstance(git_info, dict) or any(
        git_info.get(key) != value for key, value in expected_git.items()
    ):
        raise PublicationError(f"source provenance mismatch: {submission}")
    if environment.get("frozen_inputs_required") is not True or not isinstance(
        frozen, dict
    ):
        raise PublicationError(f"frozen runtime provenance missing: {submission}")
    runtime_provenance = {
        key: copy.deepcopy(frozen.get(key))
        for key in (
            "image_id",
            "model_revision",
            "cann",
            "torch_npu_version",
            "topology",
        )
    }
    if any(value in (None, "", {}) for value in runtime_provenance.values()):
        raise PublicationError(f"frozen runtime provenance incomplete: {submission}")

    workload, tp = _cell_key(cell)
    resolved_client = resolved.get("resolved_client_parameters")
    resolved_server = resolved.get("resolved_server_parameters")
    if (
        not isinstance(resolved_client, dict)
        or not isinstance(resolved_server, dict)
        or resolved.get("spec_id") != cell.get("spec_id")
        or resolved_client.get("request_rate") != cell.get("request_rate")
        or resolved_server.get("tensor_parallel_size") != tp
        or resolved_client.get("temperature") != 0
    ):
        raise PublicationError(f"resolved setting mismatch: {submission}")
    errors = raw.get("errors")
    if (
        raw.get("failed") != 0
        or not isinstance(raw.get("completed"), int)
        or raw["completed"] <= 0
        or not isinstance(errors, list)
        or any(error not in (None, "") for error in errors)
    ):
        raise PublicationError(f"failed or incomplete raw result: {submission}")

    entry_workload = entry.get("workload")
    hardware = entry.get("hardware")
    model = entry.get("model")
    metrics = entry.get("metrics")
    if (
        not isinstance(entry_workload, dict)
        or entry_workload.get("name") != workload
        or not isinstance(hardware, dict)
        or hardware.get("chip_count") != tp
        or not isinstance(model, dict)
        or not model.get("canonical_id")
        or not isinstance(metrics, dict)
    ):
        raise PublicationError(f"leaderboard setting mismatch: {submission}")
    output_throughput = _numeric(
        raw.get("output_throughput"), f"{submission}:output_throughput"
    )
    if repeat.get("raw_sha256") != sha256(submission / "raw_benchmark_result.json"):
        raise PublicationError(f"analysis raw digest mismatch: {submission}")
    return (
        copy.deepcopy(entry),
        output_throughput,
        int(repeat_index),
        runtime_provenance,
    )


def _setting_signature(
    *, load_profile: str, cell: Mapping[str, Any], entry: Mapping[str, Any]
) -> str:
    model = entry["model"]
    hardware = entry["hardware"]
    parts = (
        load_profile,
        str(cell["workload"]),
        f"tp{cell['tensor_parallel_size']}",
        f"rate={cell['request_rate']}",
        str(cell["spec_id"]),
        str(model["canonical_id"]),
        str(model.get("precision") or ""),
        str(hardware.get("chip_model") or ""),
        str(entry["workload"].get("dataset") or ""),
        f"input={entry['workload'].get('input_length')}",
        f"output={entry['workload'].get('output_length')}",
        str(entry.get("engine") or ""),
        str(entry.get("engine_version") or ""),
    )
    return "|".join(parts)


def _canonical_entry(
    *,
    load_profile: str,
    cell: Mapping[str, Any],
    repeats: Sequence[tuple[dict[str, Any], float, int, dict[str, Any]]],
    archive_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered = sorted(repeats, key=lambda item: item[2])
    repeat_signatures = {
        _setting_signature(load_profile=load_profile, cell=cell, entry=item[0])
        for item in ordered
    }
    if len(repeat_signatures) != 1:
        raise PublicationError(
            f"leaderboard setting differs across repeats for "
            f"{cell['workload']}/TP{cell['tensor_parallel_size']}"
        )
    throughputs = [item[1] for item in ordered]
    median_throughput = median(throughputs)
    # With three repetitions, the canonical row is the actual median-output
    # run.  A tie is resolved by the lowest repeat index for determinism.
    selected = min(
        (item for item in ordered if item[1] == median_throughput),
        key=lambda item: item[2],
    )
    entries = []
    for entry, _throughput, repeat_index, _provenance in ordered:
        candidate = copy.deepcopy(entry)
        candidate["repeat_index"] = repeat_index
        entries.append(candidate)
    aggregate = compute_canonical_aggregate(
        entries, method="median", outlier_handling="none"
    )
    canonical = apply_aggregate_to_entry(selected[0], aggregate)
    signature = _setting_signature(
        load_profile=load_profile, cell=cell, entry=canonical
    )
    canonical["repeat_group"] = f"issue-136|{signature}"
    canonical["repeat_index"] = selected[2]
    metadata = dict(canonical.get("metadata") or {})
    metadata["issue_136_publication"] = {
        "schema_version": SCHEMA_VERSION,
        "load_profile": load_profile,
        "setting_signature": signature,
        "selection_rule": "median-output-throughput-run; tie=lowest-repeat-index",
        "selected_repeat_index": selected[2],
        "raw_output_throughput_values": throughputs,
        "raw_output_throughput_median": median_throughput,
        "archive_sha256": archive_sha256,
        "scaling_efficiency_claim_allowed": load_profile == "scaled-load",
    }
    canonical["metadata"] = metadata
    return canonical, {
        "workload": cell["workload"],
        "tensor_parallel_size": cell["tensor_parallel_size"],
        "request_rate": cell["request_rate"],
        "spec_id": cell["spec_id"],
        "setting_signature": signature,
        "selected_repeat_index": selected[2],
        "raw_output_throughput_values": throughputs,
        "raw_output_throughput_median": median_throughput,
    }


def validate_archive(path: Path, *, expected_load_profile: str) -> ValidatedArchive:
    root = path.resolve(strict=True)
    if expected_load_profile not in LOAD_PROFILES:
        raise PublicationError(f"unsupported load profile: {expected_load_profile}")
    archive_sha = _validate_archive_manifest(root)
    if _regular_file(root, Path("STATUS")).read_text(encoding="utf-8").strip() != "ok":
        raise PublicationError("archived matrix STATUS is not ok")
    inventory = _load_json(root / "inventory.json", "archive inventory")
    plan = _load_json(root / "matrix-plan.json", "matrix plan")
    analysis = _load_json(root / "analysis-summary.json", "analysis summary")
    summary = _load_json(root / "matrix-summary.json", "matrix summary")
    if not all(isinstance(item, dict) for item in (inventory, plan, analysis, summary)):
        raise PublicationError("archive control payload must be JSON objects")
    policy = inventory.get("policy")
    if (
        inventory.get("schema_version") != ARCHIVE_INVENTORY_SCHEMA
        or not isinstance(policy, dict)
        or policy.get("source_mutated") is not False
        or policy.get("expected_cells") != 12
        or policy.get("expected_submissions_per_cell") != 3
        or len(inventory.get("cells") or []) != 12
        or len(inventory.get("submissions") or []) != 36
    ):
        raise PublicationError("archive inventory is incomplete or unsafe")
    if (
        analysis.get("schema_version") != ANALYSIS_SCHEMA
        or analysis.get("status") != "validated"
        or summary.get("status") != "ok"
        or plan.get("load_profile") != expected_load_profile
        or analysis.get("load_profile") != expected_load_profile
    ):
        raise PublicationError("matrix profile/status mismatch")
    claim_boundary = str(analysis.get("claim_boundary") or "").lower()
    if expected_load_profile == "fixed-1-rps" and (
        "does not support capacity-scaling claims" not in claim_boundary
    ):
        raise PublicationError("fixed 1 RPS archive does not forbid scaling claims")

    source_commits = analysis.get("source_commits")
    if not isinstance(source_commits, dict) or source_commits != {
        "core": plan.get("core_commit"),
        "plugin": plan.get("plugin_commit"),
        "benchmark": plan.get("benchmark_commit"),
    }:
        raise PublicationError("matrix source commit provenance mismatch")
    plan_cells = plan.get("cells")
    analysis_cells = analysis.get("cells")
    if not isinstance(plan_cells, list) or not isinstance(analysis_cells, list):
        raise PublicationError("matrix cell lists are missing")
    expected_keys = {
        (workload, tp) for workload in WORKLOADS for tp in TENSOR_PARALLEL_SIZES
    }
    plan_by_key = {
        _cell_key(cell): cell for cell in plan_cells if isinstance(cell, dict)
    }
    analysis_by_key = {
        _cell_key(cell): cell for cell in analysis_cells if isinstance(cell, dict)
    }
    if (
        len(plan_cells) != 12
        or len(analysis_cells) != 12
        or set(plan_by_key) != expected_keys
        or set(analysis_by_key) != expected_keys
    ):
        raise PublicationError(
            "matrix must contain exactly the 4 workload x 3 TP cells"
        )
    expected_inventory_cells = {
        f"cells/{plan_by_key[key]['campaign_prefix']}" for key in expected_keys
    }
    if set(inventory["cells"]) != expected_inventory_cells:
        raise PublicationError("archive cell inventory does not match the matrix plan")

    declared_submissions = set(inventory["submissions"])
    used_submissions: set[str] = set()
    canonical_entries: list[dict[str, Any]] = []
    publication_cells: list[dict[str, Any]] = []
    signatures: set[str] = set()
    runtime_provenance: dict[str, Any] | None = None
    for key in sorted(expected_keys):
        planned = plan_by_key[key]
        analyzed = analysis_by_key[key]
        if analyzed.get("spec_id") != planned.get("spec_id") or analyzed.get(
            "request_rate"
        ) != planned.get("rate"):
            raise PublicationError(f"plan/analysis mismatch for {key}")
        repeats = analyzed.get("repeats")
        if not isinstance(repeats, list) or len(repeats) != 3:
            raise PublicationError(f"expected exactly three repeats for {key}")
        validated_repeats = []
        for repeat in repeats:
            if not isinstance(repeat, dict):
                raise PublicationError(f"invalid repeat record for {key}")
            submission_name = repeat.get("submission")
            if (
                not isinstance(submission_name, str)
                or Path(submission_name).name != submission_name
            ):
                raise PublicationError(f"unsafe submission name for {key}")
            expected_submission = (
                f"cells/{planned['campaign_prefix']}/submissions/{submission_name}"
            )
            matches = [
                declared
                for declared in declared_submissions
                if declared == expected_submission
            ]
            if len(matches) != 1:
                raise PublicationError(
                    f"submission inventory mismatch: {submission_name}"
                )
            declared = matches[0]
            if declared in used_submissions:
                raise PublicationError(f"submission reused across cells: {declared}")
            used_submissions.add(declared)
            validated_repeats.append(
                _validate_submission(
                    root,
                    _find_submission(root, declared),
                    cell=analyzed,
                    repeat=repeat,
                    load_profile=expected_load_profile,
                    source_commits=source_commits,
                )
            )
        if {item[2] for item in validated_repeats} != set(REPEATS):
            raise PublicationError(f"repeat indices must be exactly 0,1,2 for {key}")
        for _entry, _throughput, _repeat_index, provenance in validated_repeats:
            if runtime_provenance is None:
                runtime_provenance = provenance
            elif provenance != runtime_provenance:
                raise PublicationError(
                    "runtime provenance differs across matrix repeats"
                )
        canonical, publication_cell = _canonical_entry(
            load_profile=expected_load_profile,
            cell=analyzed,
            repeats=validated_repeats,
            archive_sha256=archive_sha,
        )
        signature = publication_cell["setting_signature"]
        if signature in signatures:
            raise PublicationError(f"duplicate setting signature: {signature}")
        signatures.add(signature)
        canonical_entries.append(canonical)
        publication_cells.append(publication_cell)
    if used_submissions != declared_submissions:
        raise PublicationError("inventory contains unconsumed submissions")
    return ValidatedArchive(
        root=root,
        load_profile=expected_load_profile,
        archive_sha256=archive_sha,
        source_commits=source_commits,
        runtime_provenance=runtime_provenance or {},
        entries=tuple(canonical_entries),
        cells=tuple(publication_cells),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_publication_bundle(
    fixed_archive: Path, scaled_archive: Path, output_dir: Path
) -> Path:
    """Validate both archives and atomically emit canonical snapshot candidates."""
    fixed = validate_archive(fixed_archive, expected_load_profile="fixed-1-rps")
    scaled = validate_archive(scaled_archive, expected_load_profile="scaled-load")
    if fixed.source_commits != scaled.source_commits:
        raise PublicationError("fixed and scaled archives use different source commits")
    if fixed.runtime_provenance != scaled.runtime_provenance:
        raise PublicationError(
            "fixed and scaled archives use different runtime provenance"
        )
    fixed_signatures = {cell["setting_signature"] for cell in fixed.cells}
    scaled_signatures = {cell["setting_signature"] for cell in scaled.cells}
    if fixed_signatures & scaled_signatures:
        raise PublicationError("fixed and scaled setting signatures overlap")

    destination = output_dir.absolute()
    if destination.exists():
        raise PublicationError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        entries = [copy.deepcopy(entry) for entry in fixed.entries + scaled.entries]
        single = sorted(
            (entry for entry in entries if entry["hardware"]["chip_count"] == 1),
            key=lambda entry: entry["metadata"]["issue_136_publication"][
                "setting_signature"
            ],
        )
        multi = sorted(
            (entry for entry in entries if entry["hardware"]["chip_count"] > 1),
            key=lambda entry: entry["metadata"]["issue_136_publication"][
                "setting_signature"
            ],
        )
        _write_json(stage / "leaderboard_single.candidates.json", single)
        _write_json(stage / "leaderboard_multi.candidates.json", multi)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "promotion-ready",
            "promotion_policy": {
                "requires_complete_fixed_and_scaled_archives": True,
                "expected_cells_per_profile": 12,
                "expected_repeats_per_cell": 3,
                "canonical_selection": "median-output-throughput-run; tie=lowest-repeat-index",
                "fixed_1_rps_scaling_efficiency_claim_allowed": False,
                "scaled_load_scaling_efficiency_claim_allowed": True,
                "specialty_observations_included": False,
                "specialty_observations_note": (
                    "communication, MoE/EPLB, and profiler evidence must be published "
                    "as separate specialty observations"
                ),
            },
            "source_commits": dict(fixed.source_commits),
            "runtime_provenance": dict(fixed.runtime_provenance),
            "archives": {
                "fixed-1-rps": fixed.archive_sha256,
                "scaled-load": scaled.archive_sha256,
            },
            "profiles": {
                "fixed-1-rps": list(fixed.cells),
                "scaled-load": list(scaled.cells),
            },
            "snapshot_candidates": {
                "single": len(single),
                "multi": len(multi),
                "total": len(entries),
            },
        }
        _write_json(stage / "promotion-manifest.json", manifest)
        files = sorted(path for path in stage.iterdir() if path.is_file())
        (stage / "SHA256SUMS").write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in files),
            encoding="utf-8",
        )
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination
