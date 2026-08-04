from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.official_baseline_attestation import (
    attest_completed_baseline,
)
from vllm_hust_benchmark.strict_execution_contract import CANONICAL_WORKER_RULE

TRACE_IMAGE = "quay.io/ascend/vllm-ascend@sha256:" + "b" * 64
TRACE_DIGEST = "sha256:" + "b" * 64
TRACE_PACKAGES = {
    "transformers": "5.5.4",
    "huggingface-hub": "1.21.0",
    "click": "8.4.1",
    "vllm": "0.22.1+empty",
    "vllm-ascend": "0.22.1rc1",
    "torch": "2.10.0+cpu",
    "torch-npu": "2.10.0",
}
TRACE_ENVIRONMENT = {"VLLM_BATCH_INVARIANT": "1"}
MODEL_REVISION = "a" * 40
STATIC_DATA_IDENTITY = {
    "kind": "vllm-repository-file",
    "path": "benchmarks/sonnet.txt",
    "sha256": "d" * 64,
}


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_strict_execution_evidence(
    repeat: Path,
    number: int,
    *,
    chip_count: int = 1,
    peak_hbm_mb: int = 2048,
    runtime_image_digest: str = TRACE_DIGEST,
    model_id: str = "Qwen/model",
    model_revision: str = MODEL_REVISION,
    data_identity: dict | None = None,
    resolved_input_sha256: str = "e" * 64,
    resolved_input_kind: str = "throughput-sample-requests",
) -> None:
    container_id = f"{number:064x}"
    devices = list(range(chip_count))
    host_pids = [1000 + number * 10 + offset for offset in range(chip_count)]
    owned_processes = [
        {
            "host_pid": host_pid,
            "container_id": container_id,
            "physical_npu_id": device,
            "cgroup": f"/system.slice/docker-{container_id}.scope",
            "cmdline": f"VLLMWorker_TP --rank {device}",
        }
        for host_pid, device in zip(host_pids, devices, strict=True)
    ]
    owned_processes_path = repeat / "owned-processes.json"
    _write(owned_processes_path, owned_processes)
    snapshots = []
    for index, second in enumerate((5, 20), start=1):
        npu_path = repeat / f"npu-smi-pre-{index}.txt"
        inspect_path = repeat / f"docker-inspect-pre-{index}.json"
        npu_path.write_text(f"snapshot {index}\n", encoding="utf-8")
        inspect_path.write_text(f'{{"snapshot": {index}}}\n', encoding="utf-8")
        snapshots.append(
            {
                "captured_at": f"2026-08-02T00:00:{second:02d}Z",
                "physical_npu_ids": devices,
                "external_compute_pids": [],
                "external_container_ids": [],
                "lease_conflicts": [],
                "stable": True,
                "npu_smi": {
                    "path": npu_path.name,
                    "sha256": hashlib.sha256(npu_path.read_bytes()).hexdigest(),
                },
                "container_inspect": {
                    "path": inspect_path.name,
                    "sha256": hashlib.sha256(inspect_path.read_bytes()).hexdigest(),
                },
            }
        )

    per_device_peak = peak_hbm_mb // chip_count
    per_device = {str(device): per_device_peak for device in devices}
    hbm_path = repeat / "hbm-samples.jsonl"
    hbm_path.write_text(
        json.dumps(
            {
                "captured_at": "2026-08-02T00:00:25Z",
                "host_pids": host_pids,
                "physical_npu_hbm_mb": per_device,
                "total_hbm_mb": sum(per_device.values()),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cleanup_path = repeat / "cleanup-chain-attestation.json"
    _write(
        cleanup_path,
        {
            "schema_version": "cleanup-chain-attestation/v1",
            "hostname": "host-a",
            "startup_instance_id": f"startup-{number}",
            "container_id": container_id,
            "exit_code": 0,
            "host_pids": host_pids,
            "all_owned_host_pids": host_pids,
            "physical_npu_ids": devices,
            "service_port": 8000,
            "finished_at": "2026-08-02T00:00:30Z",
            "container_stopped_or_removed": True,
            "pids_absent": True,
            "port_released": True,
            "npu_processes_absent": True,
            "lease_released": True,
        },
    )
    immutable_path = repeat / "immutable-input-attestation.json"
    _write(
        immutable_path,
        {
            "schema_version": "immutable-input-attestation/v1",
            "model_id": model_id,
            "model_revision": model_revision,
            "data_identity": data_identity or STATIC_DATA_IDENTITY,
            "resolved_input_kind": resolved_input_kind,
            "resolved_input_sha256": resolved_input_sha256,
        },
    )
    _write(
        repeat / "strict_execution_evidence.json",
        {
            "schema_version": "strict-execution-evidence/v1",
            "hostname": "host-a",
            "startup_instance_id": f"startup-{number}",
            "container_id": container_id,
            "runtime_image_digest": runtime_image_digest,
            "service_port": 8000,
            "immutable_inputs": {
                "path": immutable_path.name,
                "sha256": hashlib.sha256(immutable_path.read_bytes()).hexdigest(),
            },
            "lease": {
                "physical_npu_ids": devices,
                "acquired_at": "2026-08-02T00:00:00Z",
                "released_at": "2026-08-02T00:00:35Z",
            },
            "pre_start_snapshots": snapshots,
            "ownership": [
                {
                    "host_pid": host_pid,
                    "container_id": container_id,
                    "physical_npu_id": device,
                    "cgroup": f"/system.slice/docker-{container_id}.scope",
                }
                for host_pid, device in zip(host_pids, devices, strict=True)
            ],
            "owned_processes": {
                "selection_rule": CANONICAL_WORKER_RULE,
                "raw": {
                    "path": owned_processes_path.name,
                    "sha256": hashlib.sha256(
                        owned_processes_path.read_bytes()
                    ).hexdigest(),
                },
            },
            "peak_hbm_mb": peak_hbm_mb,
            "hbm_samples": {
                "path": hbm_path.name,
                "sha256": hashlib.sha256(hbm_path.read_bytes()).hexdigest(),
            },
            "cleanup": {
                "path": cleanup_path.name,
                "sha256": hashlib.sha256(cleanup_path.read_bytes()).hexdigest(),
            },
        },
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict, dict]:
    repo = tmp_path / "repo"
    spec = {"id": "target-1"}
    spec_path = repo / "docs" / "spec.json"
    _write(spec_path, spec)
    target = {
        "target_id": "target-1",
        "target_version": "1.0.0",
        "profile": "core-text",
        "status": "active",
        "intended_use": "public-leaderboard",
        "baseline_runtime": {
            "engine": "vllm",
            "engine_version": "0.18.0",
            "git_commit": "plugin-sha",
            "runtime_image_digest": TRACE_DIGEST,
        },
        "hardware": {
            "vendor": "Huawei",
            "chip_model": "910B2",
            "chip_count": 1,
            "node_count": 1,
        },
        "model": {
            "id": "Qwen/model",
            "revision": MODEL_REVISION,
            "parameters": "14B",
            "precision": "FP16",
        },
        "server_parameters": {"max_model_len": 32768},
        "workload": {
            "name": "sonnet-throughput",
            "client_parameters": {"num_prompts": 200},
            "data_identity": STATIC_DATA_IDENTITY,
        },
        "source_spec": {
            "path": "docs/spec.json",
            "sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        },
    }
    registry_path = repo / "leaderboard-data" / "official-targets.json"
    _write(registry_path, {"targets": [target]})
    (registry_path.parent / "official-targets.sha256").write_text(
        hashlib.sha256(registry_path.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    entry = {
        "entry_id": "entry-1",
        "engine": "vllm",
        "engine_version": "0.18.0",
        "hardware": {"vendor": "Huawei", "chip_model": "910B2", "chip_count": 1},
        "model": {"repo_id": "Qwen/model", "parameters": "14B", "precision": "FP16"},
        "metrics": {
            "throughput_tps": 100.0,
            "peak_mem_mb": 2048,
            "error_rate": 0,
        },
        "same_spec": {
            "spec_id": "target-1",
            "model": "Qwen/model",
            "model_parameters": "14B",
            "model_precision": "FP16",
            "hardware_vendor": "Huawei",
            "hardware_chip_model": "910B2",
            "chip_count": 1,
            "node_count": 1,
            "resolved_server_parameters": {"max_model_len": 32768},
            "resolved_client_parameters": {"num_prompts": 200},
        },
        "metadata": {
            "idempotency_key": "key-1",
            "runtime_provenance": {
                "engine": {"commit": "core-sha"},
                "plugin": {"commit": "plugin-sha"},
            },
        },
    }
    staged = repo / "staged" / "target-1"
    _write(staged / "run_leaderboard.json", entry)
    _write(
        staged / "leaderboard_manifest.json",
        {
            "entries": [
                {
                    "idempotency_key": "key-1",
                    "leaderboard_artifact": "run_leaderboard.json",
                }
            ]
        },
    )
    results = repo / "results" / "target-1"
    for number in range(1, 4):
        repeat = results / f"repeat-{number:02d}"
        _write(repeat / "raw_benchmark_result.json", {"failed": 0})
        _write(repeat / "submission" / "run_leaderboard.json", entry)
        (repeat / "runner.log").write_text("ok\n", encoding="utf-8")
        _write_strict_execution_evidence(repeat, number)
    return repo, staged, results, entry, target


def _mutate_immutable_input(repeat: Path, **updates: object) -> None:
    immutable_path = repeat / "immutable-input-attestation.json"
    payload = json.loads(immutable_path.read_text(encoding="utf-8"))
    payload.update(updates)
    _write(immutable_path, payload)
    strict_path = repeat / "strict_execution_evidence.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    strict["immutable_inputs"]["sha256"] = hashlib.sha256(
        immutable_path.read_bytes()
    ).hexdigest()
    _write(strict_path, strict)


def _trace_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo, staged, results, entry, target = _fixture(tmp_path)
    target["profile"] = "production-trace"
    target["baseline_runtime"].update(
        {
            "core_commit": "core-sha",
            "backend_commit": "plugin-sha",
            "runtime_packages": TRACE_PACKAGES,
            "runtime_image": TRACE_IMAGE,
            "runtime_image_digest": TRACE_DIGEST,
            "runtime_environment": TRACE_ENVIRONMENT,
        }
    )
    trace_data_identity = {
        "kind": "release-asset",
        "path": "BurstGPT_3.csv",
        "sha256": "f" * 64,
    }
    target["workload"] = {
        "name": "burstgpt-production-replay",
        "client_parameters": {"max_requests": 2},
        "data_identity": trace_data_identity,
    }
    entry["same_spec"]["resolved_client_parameters"] = {"max_requests": 2}
    registry_path = repo / "leaderboard-data" / "official-targets.json"
    _write(registry_path, {"targets": [target]})
    (registry_path.parent / "official-targets.sha256").write_text(
        hashlib.sha256(registry_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    _write(staged / "run_leaderboard.json", entry)

    signature = "cohort-signature"
    model_digest = "a" * 64
    for number in range(1, 4):
        repeat = results / f"repeat-{number:02d}"
        _write_strict_execution_evidence(
            repeat,
            number,
            data_identity=trace_data_identity,
            resolved_input_kind="production-trace-prompt-token-ids",
        )
        repeat_entry = json.loads(json.dumps(entry))
        repeat_entry["entry_id"] = f"entry-{number}"
        repeat_entry["metadata"]["idempotency_key"] = f"key-{number}"
        raw = {
            "completed": 2,
            "failed": 0,
            "repeat": number,
            "cohort_setting_signature": signature,
        }
        plan = {"cohort_setting_signature": signature}
        detail_lines = [
            json.dumps(
                {
                    "type": "metadata",
                    "plan": {"cohort_setting_signature": signature},
                }
            ),
            json.dumps({"request_id": f"{number}-1"}),
            json.dumps({"request_id": f"{number}-2"}),
        ]
        _write(repeat / "raw_benchmark_result.json", raw)
        _write(repeat / "trace_replay_plan.json", plan)
        _write(
            repeat / "model_artifact_provenance.json",
            {"model_artifact_digest": model_digest},
        )
        _write(
            repeat / "runtime_package_provenance.json",
            {
                "runtime_packages": TRACE_PACKAGES,
                "runtime_image": TRACE_IMAGE,
                "runtime_image_digest": TRACE_DIGEST,
                "runtime_environment": TRACE_ENVIRONMENT,
            },
        )
        (repeat / "trace_replay_results.jsonl").write_text(
            "\n".join(detail_lines) + "\n", encoding="utf-8"
        )
        (repeat / "server.stdout.log").write_text("ready\n", encoding="utf-8")
        (repeat / "runner.log").write_text(f"repeat {number}\n", encoding="utf-8")
        _write(repeat / "submission" / "run_leaderboard.json", repeat_entry)
        _write(
            repeat / "startup_evidence.json",
            {
                "startup_instance_id": f"startup-{number}",
                "run_id": f"run-{number}",
                "engine_source_commit": "core-sha",
                "plugin_source_commit": "plugin-sha",
                "model_artifact_digest": model_digest,
                "cohort_setting_signature": signature,
                "runtime_packages": TRACE_PACKAGES,
                "runtime_image": TRACE_IMAGE,
                "runtime_image_digest": TRACE_DIGEST,
                "runtime_environment": TRACE_ENVIRONMENT,
                "finished_at": f"2026-08-02T00:00:0{number}Z",
                "result_hashes": {
                    "raw_sha256": hashlib.sha256(
                        (repeat / "raw_benchmark_result.json").read_bytes()
                    ).hexdigest(),
                    "detail_sha256": hashlib.sha256(
                        (repeat / "trace_replay_results.jsonl").read_bytes()
                    ).hexdigest(),
                },
            },
        )
    return repo, staged, results


def test_attests_three_exact_zero_error_repeats(tmp_path: Path) -> None:
    repo, staged, results, _, target = _fixture(tmp_path)
    output = repo / "submissions" / "target-1"
    attested = attest_completed_baseline(
        repo,
        staged,
        results,
        output,
        verified_by="test-review",
        verified_at="2026-08-02T00:00:00Z",
    )
    assert attested["metadata"]["verified"] is True
    assert attested["metadata"]["target_version"] == target["target_version"]
    assert attested["metadata"]["profile_id"] == target["profile"]
    suite = json.loads((output / "repeat_suite.json").read_text())
    assert suite["successful_repeats"] == 3
    assert suite["selected_repeat"] == "repeat-01"
    assert suite["repeats"][0]["startup_instance_id"] == "startup-1"
    assert suite["repeats"][0]["physical_npu_ids"] == [0]
    assert suite["repeats"][0]["peak_hbm_mb"] == 2048
    assert len(suite["repeats"][0]["pre_start_snapshots"]) == 2
    assert suite["repeats"][0]["model_revision"] == MODEL_REVISION
    assert suite["repeats"][0]["resolved_input_sha256"] == "e" * 64
    assert len(suite["repeats"][0]["immutable_input_attestation_sha256"]) == 64


def test_docker_archive_runtime_uses_config_digest_for_compatibility(
    tmp_path: Path,
) -> None:
    repo, staged, results, _, target = _fixture(tmp_path)
    storage_manifest_digest = "sha256:" + "c" * 64
    target["baseline_runtime"].update(
        {
            "runtime_transport": "docker-archive",
            "runtime_image": None,
            "runtime_config_digest": TRACE_DIGEST,
            "runtime_archive_sha256": "sha256:" + "d" * 64,
            "containerd_storage_manifest_digest": storage_manifest_digest,
        }
    )
    registry_path = repo / "leaderboard-data" / "official-targets.json"
    _write(registry_path, {"targets": [target]})
    (registry_path.parent / "official-targets.sha256").write_text(
        hashlib.sha256(registry_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )

    attest_completed_baseline(
        repo,
        staged,
        results,
        repo / "out",
        verified_by="test-review",
    )

    strict_path = results / "repeat-02" / "strict_execution_evidence.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    strict["runtime_image_digest"] = storage_manifest_digest
    _write(strict_path, strict)
    with pytest.raises(ValueError, match="strict execution runtime image mismatch"):
        attest_completed_baseline(
            repo,
            staged,
            results,
            repo / "out-storage-digest",
            verified_by="test-review",
        )


def test_rejects_missing_strict_execution_evidence(tmp_path: Path) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    (results / "repeat-02" / "strict_execution_evidence.json").unlink()
    with pytest.raises(ValueError, match="strict execution evidence is missing"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_model_revision_mismatch(tmp_path: Path) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    _mutate_immutable_input(
        results / "repeat-02",
        model_revision="b" * 40,
    )
    with pytest.raises(ValueError, match="model revision mismatch"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_data_identity_mismatch(tmp_path: Path) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    _mutate_immutable_input(
        results / "repeat-02",
        data_identity={"kind": "repository-file", "sha256": "c" * 64},
    )
    with pytest.raises(ValueError, match="data identity mismatch"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_resolved_input_kind_mismatch(tmp_path: Path) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    _mutate_immutable_input(
        results / "repeat-02",
        resolved_input_kind="serve-sample-requests",
    )
    with pytest.raises(ValueError, match="resolved input kind mismatch"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_deterministic_input_drift(tmp_path: Path) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    _mutate_immutable_input(
        results / "repeat-02",
        resolved_input_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="different resolved inputs"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_pre_start_snapshots_less_than_15_seconds_apart(
    tmp_path: Path,
) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    evidence_path = results / "repeat-02" / "strict_execution_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["pre_start_snapshots"][1]["captured_at"] = "2026-08-02T00:00:10Z"
    _write(evidence_path, evidence)
    with pytest.raises(ValueError, match="less than 15 seconds apart"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_pid_to_physical_npu_mapping_mismatch(tmp_path: Path) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    evidence_path = results / "repeat-02" / "strict_execution_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["ownership"][0]["physical_npu_id"] = 7
    _write(evidence_path, evidence)
    with pytest.raises(ValueError, match="physical NPU mapping mismatch"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_unmeasured_peak_hbm(tmp_path: Path) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    artifact_path = results / "repeat-02" / "submission" / "run_leaderboard.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["metrics"]["peak_mem_mb"] = 0
    _write(artifact_path, artifact)
    with pytest.raises(ValueError, match="peak HBM mismatch"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_failed_cleanup_chain(tmp_path: Path) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    repeat = results / "repeat-02"
    cleanup_path = repeat / "cleanup-chain-attestation.json"
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
    cleanup["pids_absent"] = False
    _write(cleanup_path, cleanup)
    evidence_path = repeat / "strict_execution_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["cleanup"]["sha256"] = hashlib.sha256(
        cleanup_path.read_bytes()
    ).hexdigest()
    _write(evidence_path, evidence)
    with pytest.raises(ValueError, match="cleanup-chain mismatch for pids_absent"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_fewer_than_three_repeats(tmp_path: Path) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    for path in (results / "repeat-03").glob("**/*"):
        if path.is_file():
            path.unlink()
    with pytest.raises(ValueError, match="insufficient successful repeats"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_attests_three_independent_production_trace_repeats(tmp_path: Path) -> None:
    repo, staged, results = _trace_fixture(tmp_path)
    attested = attest_completed_baseline(
        repo, staged, results, repo / "out", verified_by="test-review"
    )
    assert attested["metadata"]["verified"] is True
    suite = json.loads((repo / "out" / "repeat_suite.json").read_text())
    assert suite["successful_repeats"] == 3
    assert {repeat["startup_instance_id"] for repeat in suite["repeats"]} == {
        "startup-1",
        "startup-2",
        "startup-3",
    }


def test_rejects_cloned_production_trace_repeat(tmp_path: Path) -> None:
    repo, staged, results = _trace_fixture(tmp_path)
    repeat_2 = results / "repeat-02"
    repeat_1 = results / "repeat-01"
    (repeat_2 / "startup_evidence.json").write_bytes(
        (repeat_1 / "startup_evidence.json").read_bytes()
    )
    with pytest.raises(ValueError, match="startup raw result hash mismatch"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_incomplete_production_trace_evidence(tmp_path: Path) -> None:
    repo, staged, results = _trace_fixture(tmp_path)
    (results / "repeat-03" / "trace_replay_plan.json").unlink()
    with pytest.raises(ValueError, match="evidence is missing"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_production_trace_runtime_image_digest_mismatch(
    tmp_path: Path,
) -> None:
    repo, staged, results = _trace_fixture(tmp_path)
    provenance_path = results / "repeat-02" / "runtime_package_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime_image_digest"] = "sha256:" + "c" * 64
    _write(provenance_path, provenance)

    with pytest.raises(ValueError, match="runtime image provenance mismatch"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_production_trace_runtime_environment_mismatch(
    tmp_path: Path,
) -> None:
    repo, staged, results = _trace_fixture(tmp_path)
    provenance_path = results / "repeat-02" / "runtime_package_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime_environment"] = {"VLLM_BATCH_INVARIANT": "0"}
    _write(provenance_path, provenance)

    with pytest.raises(ValueError, match="runtime environment provenance mismatch"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )
