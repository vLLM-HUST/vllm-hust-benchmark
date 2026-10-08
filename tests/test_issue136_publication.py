from __future__ import annotations

import copy
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
from vllm_hust_benchmark.issue136_target_promotion import (
    PromotionError,
    project_registry_promotion,
    write_promotion_projection,
)
from vllm_hust_benchmark.snapshot_target_binding import (
    OfficialTargetRegistry,
    bind_entry_to_official_target,
    bind_snapshot_set,
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
            spec_id = f"specialty-ascend-vllm-issue136-{load_profile}-{workload}-tp{tp}"
            port = 8400 + workload_index * 10 + tp
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
                    "mean_tpot_ms": 5.0 + repeat_index,
                    "mean_itl_ms": 6.0 + repeat_index,
                    "p95_ttft_ms": None,
                    "p99_ttft_ms": 120.0 + repeat_index,
                    "p95_tpot_ms": None,
                    "p99_tpot_ms": 8.0 + repeat_index,
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
                            "host": "127.0.0.1",
                            "port": port,
                        },
                        "resolved_server_parameters": {
                            "tensor_parallel_size": tp,
                            "max_model_len": 32768,
                            "host": "127.0.0.1",
                            "port": port,
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
                        "engine": "vllm-hust",
                        "engine_version": "0.23.0",
                        "config_type": "single_gpu" if tp == 1 else "multi_gpu",
                        "hardware": {
                            "vendor": "Huawei",
                            "chip_model": "910B2",
                            "chip_count": tp,
                        },
                        "model": {
                            "canonical_id": "hf:Qwen/Qwen2.5-14B-Instruct",
                            "repo_id": "Qwen/Qwen2.5-14B-Instruct",
                            "parameters": "14B",
                            "precision": "FP16",
                        },
                        "workload": {"name": workload},
                        "metrics": {
                            "throughput_tps": output_throughput,
                            "ttft_ms": 100.0 + repeat_index + workload_index,
                            "tbt_ms": 5.0 + repeat_index,
                            "error_rate": 0.0,
                        },
                        "constraints": {
                            "metrics": {
                                "long_context_length": 32768,
                                "long_context_throughput_stable": True,
                                "long_context_ttft_p95_ms": None,
                                "long_context_ttft_p99_ms": 120.0 + repeat_index,
                                "long_context_tpot_p95_ms": None,
                                "long_context_tpot_p99_ms": 8.0 + repeat_index,
                                "long_context_ttft_p95_stable": None,
                                "long_context_ttft_p99_stable": True,
                                "long_context_tpot_p95_stable": None,
                                "long_context_tpot_p99_stable": True,
                            }
                        },
                        "metadata": {
                            "target_contract_id": spec_id,
                            "git_commit": COMMITS["core"],
                        },
                        "same_spec": {
                            "schema_version": "benchmark-same-spec/v1",
                            "spec_id": spec_id,
                            "scenario": workload,
                            "model": "Qwen/Qwen2.5-14B-Instruct",
                            "model_parameters": "14B",
                            "model_precision": "FP16",
                            "model_quantization": "",
                            "hardware_vendor": "Huawei",
                            "hardware_chip_model": "910B2",
                            "chip_count": tp,
                            "node_count": 1,
                            "resolved_client_parameters": {
                                "request_rate": rate,
                                "temperature": 0,
                                "num_prompts": 10,
                                "host": "127.0.0.1",
                                "port": port,
                            },
                            "resolved_server_parameters": {
                                "tensor_parallel_size": tp,
                                "max_model_len": 32768,
                                "host": "127.0.0.1",
                                "port": port,
                                "compilation_config": EXPECTED_GRAPH_CONFIG,
                            },
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
                            "mean_tpot_ms": 5.0 + repeat_index,
                            "mean_itl_ms": 6.0 + repeat_index,
                            "p99_ttft_ms": 120.0 + repeat_index,
                            "p99_tpot_ms": 8.0 + repeat_index,
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


def _registry_for_bundle(bundle: Path) -> OfficialTargetRegistry:
    manifest = json.loads((bundle / "promotion-manifest.json").read_text())
    cells = {
        cell["spec_id"]: cell
        for profile_cells in manifest["profiles"].values()
        for cell in profile_cells
    }
    candidates = []
    for name in (
        "leaderboard_single.candidates.json",
        "leaderboard_multi.candidates.json",
    ):
        candidates.extend(json.loads((bundle / name).read_text()))
    targets = {}
    for entry in candidates:
        same_spec = entry["same_spec"]
        target_id = same_spec["spec_id"]
        server = copy.deepcopy(same_spec["resolved_server_parameters"])
        client = copy.deepcopy(same_spec["resolved_client_parameters"])
        server.update(host="0.0.0.0", port=8000)
        client.update(host="127.0.0.1", port=8000)
        targets[target_id] = {
            "target_id": target_id,
            "target_version": "test",
            "status": "provisional",
            "effective_from": "2026-10-07",
            "supersedes": [],
            "profile": "dense-scaling",
            "intended_use": "specialty",
            "baseline_runtime": {"engine": "vllm", "engine_version": "0.23.0"},
            "model": {
                "id": same_spec["model"],
                "parameters": same_spec["model_parameters"],
                "precision": same_spec["model_precision"],
            },
            "hardware": {
                "vendor": same_spec["hardware_vendor"],
                "chip_model": same_spec["hardware_chip_model"],
                "chip_count": same_spec["chip_count"],
                "node_count": same_spec["node_count"],
            },
            "server_parameters": server,
            "workload": {
                "name": same_spec["scenario"],
                "client_parameters": client,
            },
            "source_spec": {
                "path": f"docs/official-baselines/{target_id}.json",
                "sha256": cells[target_id]["spec_sha256"],
            },
            "compatibility_policy": {
                "model": "exact",
                "hardware": "exact",
                "server_parameters": "exact",
                "workload_parameters": "exact",
                "exceptions": "new-target-version-required",
            },
        }
    return OfficialTargetRegistry(version="test", sha256="b" * 64, targets=targets)


def _projected_registry(payload: dict) -> OfficialTargetRegistry:
    return OfficialTargetRegistry(
        version=payload["base_registry"]["version"],
        sha256=payload["base_registry"]["sha256"],
        targets={target["target_id"]: target for target in payload["targets"]},
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
    # Public tbt_ms is TPOT by contract, not raw inter-token latency (ITL).
    assert all(entry["metrics"]["tbt_ms"] == 6.0 for entry in single + multi)
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


def test_accepts_successful_upstream_raw_without_errors_key(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    analysis_path = archive / "analysis-summary.json"
    analysis = json.loads(analysis_path.read_text())
    repeats_by_submission = {
        repeat["submission"]: repeat
        for cell in analysis["cells"]
        for repeat in cell["repeats"]
    }
    for raw_path in archive.glob("cells/*/submissions/*/raw_benchmark_result.json"):
        raw = json.loads(raw_path.read_text())
        raw.pop("errors")
        _write_json(raw_path, raw)
        repeats_by_submission[raw_path.parent.name]["raw_sha256"] = _sha(raw_path)
        _refresh_submission_manifest(raw_path.parent)
    _write_json(analysis_path, analysis)
    _refresh_archive_manifest(archive)
    result = validate_archive(archive, expected_load_profile="fixed-1-rps")
    assert len(result.entries) == 12


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


def test_rejects_itl_mislabeled_as_public_tbt(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    leaderboard = next(archive.glob("cells/*/submissions/*/run_leaderboard.json"))
    raw = json.loads((leaderboard.parent / "raw_benchmark_result.json").read_text())
    payload = json.loads(leaderboard.read_text())
    assert raw["mean_itl_ms"] != raw["mean_tpot_ms"]
    payload["metrics"]["tbt_ms"] = raw["mean_itl_ms"]
    _write_json(leaderboard, payload)
    _refresh_submission_manifest(leaderboard.parent)
    _refresh_archive_manifest(archive)
    with pytest.raises(PublicationError, match="raw/leaderboard metric mismatch"):
        validate_archive(archive, expected_load_profile="fixed-1-rps")


def test_rejects_raw_leaderboard_constraint_drift(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    leaderboard = next(archive.glob("cells/*/submissions/*/run_leaderboard.json"))
    payload = json.loads(leaderboard.read_text())
    payload["constraints"]["metrics"]["long_context_tpot_p99_ms"] += 1
    _write_json(leaderboard, payload)
    _refresh_submission_manifest(leaderboard.parent)
    _refresh_archive_manifest(archive)
    with pytest.raises(PublicationError, match="constraint mismatch"):
        validate_archive(archive, expected_load_profile="fixed-1-rps")


def test_rejects_public_model_identity_drift(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    leaderboard = next(archive.glob("cells/*/submissions/*/run_leaderboard.json"))
    payload = json.loads(leaderboard.read_text())
    payload["model"]["repo_id"] = "Qwen/not-the-measured-model"
    _write_json(leaderboard, payload)
    _refresh_submission_manifest(leaderboard.parent)
    _refresh_archive_manifest(archive)
    with pytest.raises(PublicationError, match="leaderboard setting mismatch"):
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


def test_evidence_backed_projection_admits_without_mutating_observed_values(
    tmp_path: Path,
) -> None:
    fixed = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    scaled = _make_archive(tmp_path / "scaled", "scaled-load")
    bundle = build_publication_bundle(fixed, scaled, tmp_path / "bundle")
    registry = _registry_for_bundle(bundle)
    candidate = json.loads((bundle / "leaderboard_multi.candidates.json").read_text())[
        0
    ]
    original_same_spec = copy.deepcopy(candidate["same_spec"])

    admitted, errors = bind_entry_to_official_target(copy.deepcopy(candidate), registry)
    assert admitted is False
    assert "registry target is not active public-leaderboard" in errors[-1]

    patch, projected_payload = project_registry_promotion(bundle, registry)
    assert patch["status"] == "review-required"
    assert len(patch["operations"]) == 24
    projected = _projected_registry(projected_payload)
    admitted, errors = bind_entry_to_official_target(candidate, projected)
    assert admitted is True
    assert errors == []
    assert candidate["same_spec"] == original_same_spec
    audit = candidate["metadata"]["binding_projection_audit"]
    assert audit["engine"] == {
        "observed": "vllm-hust",
        "target": "vllm",
        "alias_authorized": True,
    }
    assert audit["ephemeral_transport"]["server_parameters.port"] != 8000
    assert audit["observed_values_preserved"] is True


def test_projected_registry_binds_all_candidates_only_after_promotion(
    tmp_path: Path,
) -> None:
    fixed = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    scaled = _make_archive(tmp_path / "scaled", "scaled-load")
    bundle = build_publication_bundle(fixed, scaled, tmp_path / "bundle")
    registry = _registry_for_bundle(bundle)
    unpromoted = tmp_path / "unpromoted-snapshots"
    unpromoted.mkdir()
    (unpromoted / "leaderboard_single.json").write_bytes(
        (bundle / "leaderboard_single.candidates.json").read_bytes()
    )
    (unpromoted / "leaderboard_multi.json").write_bytes(
        (bundle / "leaderboard_multi.candidates.json").read_bytes()
    )
    unpromoted_report = bind_snapshot_set(unpromoted, registry)
    assert unpromoted_report["verified"] == 0
    assert unpromoted_report["historical_unverified"] == 24

    _, projected_payload = project_registry_promotion(bundle, registry)
    projected = _projected_registry(projected_payload)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    (snapshots / "leaderboard_single.json").write_bytes(
        (bundle / "leaderboard_single.candidates.json").read_bytes()
    )
    (snapshots / "leaderboard_multi.json").write_bytes(
        (bundle / "leaderboard_multi.candidates.json").read_bytes()
    )
    report = bind_snapshot_set(snapshots, projected)
    assert report["verified"] == 24
    assert report["historical_unverified"] == 0


def test_projection_writer_emits_review_artifacts_not_snapshots(tmp_path: Path) -> None:
    fixed = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    scaled = _make_archive(tmp_path / "scaled", "scaled-load")
    bundle = build_publication_bundle(fixed, scaled, tmp_path / "bundle")
    output = write_promotion_projection(
        bundle, _registry_for_bundle(bundle), tmp_path / "projection"
    )
    assert {path.name for path in output.iterdir()} == {
        "SHA256SUMS",
        "registry-promotion.patch.json",
        "projected-official-targets.json",
    }
    assert not list(output.glob("leaderboard_*.json"))


def test_projection_does_not_authorize_non_transport_drift(tmp_path: Path) -> None:
    fixed = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    scaled = _make_archive(tmp_path / "scaled", "scaled-load")
    bundle = build_publication_bundle(fixed, scaled, tmp_path / "bundle")
    registry = _registry_for_bundle(bundle)
    _, projected_payload = project_registry_promotion(bundle, registry)
    projected = _projected_registry(projected_payload)
    candidate = json.loads((bundle / "leaderboard_multi.candidates.json").read_text())[
        0
    ]
    candidate["same_spec"]["resolved_server_parameters"]["max_model_len"] = 1
    admitted, errors = bind_entry_to_official_target(candidate, projected)
    assert admitted is False
    assert any("max_model_len mismatch" in error for error in errors)


def test_malformed_projection_policy_does_not_authorize_alias(tmp_path: Path) -> None:
    fixed = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    scaled = _make_archive(tmp_path / "scaled", "scaled-load")
    bundle = build_publication_bundle(fixed, scaled, tmp_path / "bundle")
    registry = _registry_for_bundle(bundle)
    _, projected_payload = project_registry_promotion(bundle, registry)
    target = projected_payload["targets"][0]
    target["compatibility_policy"]["evidence_backed_projection"][
        "ephemeral_transport_fields"
    ] = ["server_parameters.port"]
    projected = _projected_registry(projected_payload)
    candidate = next(
        entry
        for name in (
            "leaderboard_single.candidates.json",
            "leaderboard_multi.candidates.json",
        )
        for entry in json.loads((bundle / name).read_text())
        if entry["same_spec"]["spec_id"] == target["target_id"]
    )
    admitted, errors = bind_entry_to_official_target(candidate, projected)
    assert admitted is False
    assert "evidence-backed binding projection policy is malformed" in errors


def test_incomplete_bundle_cannot_generate_registry_promotion(tmp_path: Path) -> None:
    fixed = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    scaled = _make_archive(tmp_path / "scaled", "scaled-load")
    bundle = build_publication_bundle(fixed, scaled, tmp_path / "bundle")
    registry = _registry_for_bundle(bundle)
    manifest_path = bundle / "promotion-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["profiles"]["scaled-load"].pop()
    _write_json(manifest_path, manifest)
    _refresh_archive_manifest(bundle)
    with pytest.raises(PromotionError, match="twelve cells"):
        project_registry_promotion(bundle, registry)


def test_projection_rejects_missing_or_source_drifted_registry_target(
    tmp_path: Path,
) -> None:
    fixed = _make_archive(tmp_path / "fixed", "fixed-1-rps")
    scaled = _make_archive(tmp_path / "scaled", "scaled-load")
    bundle = build_publication_bundle(fixed, scaled, tmp_path / "bundle")
    registry = _registry_for_bundle(bundle)
    removed_id, removed = registry.targets.popitem()
    with pytest.raises(PromotionError, match="missing promotion targets"):
        project_registry_promotion(bundle, registry)
    registry.targets[removed_id] = removed
    removed["source_spec"]["sha256"] = "f" * 64
    with pytest.raises(PromotionError, match="not an exact promotable Dense target"):
        project_registry_promotion(bundle, registry)
