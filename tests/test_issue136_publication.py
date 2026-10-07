from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.issue136_publication import (
    EXPECTED_GRAPH_CONFIG,
    EXPECTED_RUNTIME_PROVENANCE,
    EXPECTED_SOURCE_COMMITS,
    PublicationError,
    WORKLOADS,
    build_publication_bundle,
    validate_archive,
)


COMMITS = EXPECTED_SOURCE_COMMITS


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_manifest(root: Path, name: str, paths: list[Path]) -> None:
    (root / name).write_text(
        "".join(
            f"{_sha(path)}  {path.relative_to(root).as_posix()}\n"
            for path in sorted(paths)
        ),
        encoding="utf-8",
    )


def _make_archive(root: Path, load_profile: str) -> Path:
    root.mkdir()
    (root / "STATUS").write_text("ok\n", encoding="utf-8")
    plan_cells = []
    analysis_cells = []
    inventory_cells = []
    inventory_submissions = []
    for workload_index, workload in enumerate(WORKLOADS):
        for tp in (1, 2, 4):
            prefix = f"issue136-{load_profile}-{workload}-tp{tp}"
            rate = 1.0 if load_profile == "fixed-1-rps" else float(tp * 2)
            spec_id = f"issue136-{load_profile}-{workload}-tp{tp}"
            plan_cells.append(
                {
                    "workload": workload,
                    "tp": tp,
                    "rate": rate,
                    "spec_id": spec_id,
                    "campaign_prefix": prefix,
                    "spec_sha256": "a" * 64,
                }
            )
            inventory_cells.append(f"cells/{prefix}")
            repeats = []
            # The median output run is repeat 2, not the chronologically last
            # or the fastest run.
            throughputs = (30.0, 10.0, 20.0)
            for repeat_index, output_throughput in enumerate(throughputs):
                submission_name = f"{prefix}-r{repeat_index}"
                relative = f"cells/{prefix}/submissions/{submission_name}"
                submission = root / relative
                submission.mkdir(parents=True)
                inventory_submissions.append(relative)
                (submission / "STATUS").write_text("OK\n", encoding="utf-8")
                raw = {
                    "completed": 10,
                    "num_prompts": 10,
                    "failed": 0,
                    "errors": [],
                    "output_throughput": output_throughput,
                    "request_throughput": 1.0,
                    "mean_ttft_ms": 100.0 + repeat_index + workload_index,
                    "mean_itl_ms": 5.0 + repeat_index,
                }
                _write_json(submission / "raw_benchmark_result.json", raw)
                _write_json(
                    submission / "resolved_same_spec.json",
                    {
                        "spec_id": spec_id,
                        "resolved_client_parameters": {
                            "request_rate": rate,
                            "temperature": 0,
                            "num_prompts": 10,
                        },
                        "resolved_server_parameters": {
                            "tensor_parallel_size": tp,
                            "compilation_config": EXPECTED_GRAPH_CONFIG,
                        },
                    },
                )
                _write_json(
                    submission / "env-manifest.json",
                    {
                        "frozen_inputs_required": True,
                        "frozen_inputs": EXPECTED_RUNTIME_PROVENANCE,
                        "campaign": {
                            "campaign_id": "issue-136-current-main/v1",
                            "coverage_class": "full-matrix",
                            "point_role": "checkpoint",
                            "repeat_index": repeat_index,
                            "repetitions": 3,
                            "load_profile": load_profile,
                        },
                        "git_info": {
                            "benchmark": COMMITS["benchmark"],
                            "vllm_hust": {
                                "declared": COMMITS["core"],
                                "observed": COMMITS["core"],
                            },
                            "vllm_ascend_hust": {
                                "declared": COMMITS["plugin"],
                                "observed": COMMITS["plugin"],
                            },
                        },
                    },
                )
                _write_json(
                    submission / "run_leaderboard.json",
                    {
                        "entry_id": submission_name,
                        "engine": "vllm",
                        "engine_version": "0.23.0",
                        "config_type": "single_gpu" if tp == 1 else "multi_gpu",
                        "hardware": {
                            "chip_model": "910B2",
                            "chip_count": tp,
                        },
                        "model": {
                            "canonical_id": "hf:Qwen/Qwen2.5-14B-Instruct",
                            "precision": "FP16",
                        },
                        "workload": {"name": workload},
                        "metrics": {
                            "throughput_tps": output_throughput,
                            "ttft_ms": 100.0 + repeat_index + workload_index,
                            "tbt_ms": 5.0 + repeat_index,
                            "error_rate": 0.0,
                        },
                        "metadata": {
                            "target_contract_id": spec_id,
                            "git_commit": COMMITS["core"],
                        },
                    },
                )
                (submission / "server.stdout.log").write_text(
                    "server complete\n", encoding="utf-8"
                )
                covered = [
                    submission / "env-manifest.json",
                    submission / "raw_benchmark_result.json",
                    submission / "resolved_same_spec.json",
                    submission / "run_leaderboard.json",
                    submission / "server.stdout.log",
                ]
                _write_manifest(submission, "checksums.sha256", covered)
                repeats.append(
                    {
                        "submission": submission_name,
                        "raw_sha256": _sha(submission / "raw_benchmark_result.json"),
                        "repeat_index": repeat_index,
                        "metrics": {
                            "request_throughput": 1.0,
                            "output_throughput": output_throughput,
                            "mean_ttft_ms": 100.0 + repeat_index + workload_index,
                            "mean_itl_ms": 5.0 + repeat_index,
                        },
                    }
                )
            analysis_cells.append(
                {
                    "workload": workload,
                    "tensor_parallel_size": tp,
                    "request_rate": rate,
                    "spec_id": spec_id,
                    "spec_sha256": "a" * 64,
                    "expected_prompts": 10,
                    "repeats": repeats,
                }
            )

    _write_json(
        root / "matrix-plan.json",
        {
            "load_profile": load_profile,
            "benchmark_commit": COMMITS["benchmark"],
            "core_commit": COMMITS["core"],
            "plugin_commit": COMMITS["plugin"],
            "cells": plan_cells,
        },
    )
    _write_json(root / "matrix-summary.json", {"status": "ok"})
    _write_json(
        root / "analysis-summary.json",
        {
            "schema_version": "issue-136-formal-matrix-analysis/v1",
            "status": "validated",
            "load_profile": load_profile,
            "claim_boundary": (
                "fixed 1 RPS is an offered-load latency checkpoint and does not "
                "support capacity-scaling claims"
                if load_profile == "fixed-1-rps"
                else "scaled-load rates are workload/TP-specific capacity-pilot decisions"
            ),
            "source_commits": COMMITS,
            "cells": analysis_cells,
            "scaling": [],
        },
    )
    _write_json(
        root / "inventory.json",
        {
            "schema_version": "issue-136-evidence-archive-inventory/v1",
            "policy": {
                "source_mutated": False,
                "expected_cells": 12,
                "expected_submissions_per_cell": 3,
            },
            "cells": inventory_cells,
            "submissions": inventory_submissions,
        },
    )
    files = [path for path in root.rglob("*") if path.is_file()]
    _write_manifest(root, "SHA256SUMS", files)
    return root


def _refresh_archive_manifest(root: Path) -> None:
    (root / "SHA256SUMS").unlink()
    _write_manifest(
        root, "SHA256SUMS", [path for path in root.rglob("*") if path.is_file()]
    )


def _refresh_submission_manifest(submission: Path) -> None:
    _write_manifest(
        submission,
        "checksums.sha256",
        [
            submission / "env-manifest.json",
            submission / "raw_benchmark_result.json",
            submission / "resolved_same_spec.json",
            submission / "run_leaderboard.json",
            submission / "server.stdout.log",
        ],
    )


def test_builds_promotion_bundle_only_from_two_complete_profiles(
    tmp_path: Path,
) -> None:
    fixed = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    scaled = _make_archive(tmp_path / "scaled", "scaled-load")
    output = build_publication_bundle(fixed, scaled, tmp_path / "bundle")

    manifest = json.loads((output / "promotion-manifest.json").read_text())
    single = json.loads((output / "leaderboard_single.candidates.json").read_text())
    multi = json.loads((output / "leaderboard_multi.candidates.json").read_text())
    assert manifest["status"] == "promotion-ready"
    assert manifest["snapshot_candidates"] == {"single": 8, "multi": 16, "total": 24}
    assert (
        manifest["promotion_policy"]["fixed_1_rps_scaling_efficiency_claim_allowed"]
        is False
    )
    assert manifest["promotion_policy"]["specialty_observations_included"] is False
    assert len(single) + len(multi) == 24
    assert all(
        entry["canonical_aggregate"]["method"] == "median" for entry in single + multi
    )
    assert all(entry["canonical_aggregate"]["count"] == 3 for entry in single + multi)
    assert all(entry["repeat_index"] == 2 for entry in single + multi)
    assert all(
        entry["metadata"]["issue_136_publication"]["raw_output_throughput_median"]
        == 20.0
        for entry in single + multi
    )
    signatures = {
        entry["metadata"]["issue_136_publication"]["setting_signature"]
        for entry in single + multi
    }
    assert len(signatures) == 24


def test_rejects_incomplete_cell_matrix(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    analysis_path = archive / "analysis-summary.json"
    analysis = json.loads(analysis_path.read_text())
    analysis["cells"].pop()
    _write_json(analysis_path, analysis)
    _refresh_archive_manifest(archive)
    with pytest.raises(PublicationError, match="exactly the 4 workload x 3 TP"):
        validate_archive(archive, expected_load_profile="fixed-1-rps")


def test_rejects_submission_checksum_tamper(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    raw = next(archive.glob("cells/*/submissions/*/raw_benchmark_result.json"))
    raw.write_text("{}\n", encoding="utf-8")
    _refresh_archive_manifest(archive)
    with pytest.raises(PublicationError, match="checksum mismatch"):
        validate_archive(archive, expected_load_profile="fixed-1-rps")


def test_rejects_fixed_archive_without_no_scaling_claim_boundary(
    tmp_path: Path,
) -> None:
    archive = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    analysis_path = archive / "analysis-summary.json"
    analysis = json.loads(analysis_path.read_text())
    analysis["claim_boundary"] = "scaling efficiency is available"
    _write_json(analysis_path, analysis)
    _refresh_archive_manifest(archive)
    with pytest.raises(PublicationError, match="does not forbid scaling claims"):
        validate_archive(archive, expected_load_profile="fixed-1-rps")


def test_rejects_cross_profile_source_drift(tmp_path: Path) -> None:
    fixed = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    scaled = _make_archive(tmp_path / "scaled", "scaled-load")
    plan_path = scaled / "matrix-plan.json"
    analysis_path = scaled / "analysis-summary.json"
    plan = json.loads(plan_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    plan["core_commit"] = "f" * 40
    analysis["source_commits"]["core"] = "f" * 40
    _write_json(plan_path, plan)
    _write_json(analysis_path, analysis)
    for env_path in scaled.glob("cells/*/submissions/*/env-manifest.json"):
        environment = json.loads(env_path.read_text())
        environment["git_info"]["vllm_hust"] = {
            "declared": "f" * 40,
            "observed": "f" * 40,
        }
        _write_json(env_path, environment)
        submission = env_path.parent
        covered = [
            submission / "env-manifest.json",
            submission / "raw_benchmark_result.json",
            submission / "resolved_same_spec.json",
            submission / "run_leaderboard.json",
            submission / "server.stdout.log",
        ]
        _write_manifest(submission, "checksums.sha256", covered)
    _refresh_archive_manifest(scaled)
    with pytest.raises(PublicationError, match="source commit provenance mismatch"):
        build_publication_bundle(fixed, scaled, tmp_path / "bundle")


def test_rejects_raw_leaderboard_metric_drift(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    leaderboard = next(archive.glob("cells/*/submissions/*/run_leaderboard.json"))
    payload = json.loads(leaderboard.read_text())
    payload["metrics"]["throughput_tps"] += 1
    _write_json(leaderboard, payload)
    _refresh_submission_manifest(leaderboard.parent)
    _refresh_archive_manifest(archive)
    with pytest.raises(PublicationError, match="raw/leaderboard metric mismatch"):
        validate_archive(archive, expected_load_profile="fixed-1-rps")


def test_rejects_analysis_raw_metric_drift(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    analysis_path = archive / "analysis-summary.json"
    analysis = json.loads(analysis_path.read_text())
    analysis["cells"][0]["repeats"][0]["metrics"]["output_throughput"] += 1
    _write_json(analysis_path, analysis)
    _refresh_archive_manifest(archive)
    with pytest.raises(PublicationError, match="analysis/raw metric mismatch"):
        validate_archive(archive, expected_load_profile="fixed-1-rps")


def test_rejects_runtime_provenance_drift(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    environment = next(archive.glob("cells/*/submissions/*/env-manifest.json"))
    payload = json.loads(environment.read_text())
    payload["frozen_inputs"]["model_revision"] = "f" * 64
    _write_json(environment, payload)
    _refresh_submission_manifest(environment.parent)
    _refresh_archive_manifest(archive)
    with pytest.raises(PublicationError, match="runtime provenance mismatch"):
        validate_archive(archive, expected_load_profile="fixed-1-rps")


def test_rejects_resolved_drift_across_repeats(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    resolved_paths = sorted(
        archive.glob("cells/*/submissions/*/resolved_same_spec.json")
    )
    resolved = resolved_paths[0]
    payload = json.loads(resolved.read_text())
    payload["resolved_client_parameters"]["dataset_path"] = "/unexpected.json"
    _write_json(resolved, payload)
    _refresh_submission_manifest(resolved.parent)
    _refresh_archive_manifest(archive)
    with pytest.raises(PublicationError, match="resolved settings differ"):
        validate_archive(archive, expected_load_profile="fixed-1-rps")


def test_rejects_non_one_rps_fixed_contract(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    plan_path = archive / "matrix-plan.json"
    analysis_path = archive / "analysis-summary.json"
    plan = json.loads(plan_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    plan["cells"][0]["rate"] = 2
    analysis["cells"][0]["request_rate"] = 2
    _write_json(plan_path, plan)
    _write_json(analysis_path, analysis)
    _refresh_archive_manifest(archive)
    with pytest.raises(PublicationError, match="invalid frozen cell contract"):
        validate_archive(archive, expected_load_profile="fixed-1-rps")
