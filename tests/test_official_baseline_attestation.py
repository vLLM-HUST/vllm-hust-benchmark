from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.official_baseline_attestation import (
    attest_completed_baseline,
)
from vllm_hust_benchmark.immutable_input_attestation import resolved_input_sha256
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


def _hashed_record(repeat: Path, path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(repeat)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _ascend_mounts() -> list[dict[str, object]]:
    return [
        {"Source": path, "Destination": path, "RW": False}
        for path in (
            "/usr/local/Ascend/driver",
            "/etc/ascend_install.info",
        )
    ]


def _ascend_mount_argv() -> list[str]:
    argv = []
    for mount in _ascend_mounts():
        path = str(mount["Source"])
        argv.extend(["--mount", f"type=bind,src={path},dst={path},readonly"])
    return argv


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
    resolved_inputs: object | None = None,
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
    captured_inputs = (
        resolved_inputs if resolved_inputs is not None else [{"prompt": "fixed"}]
    )
    _write(
        immutable_path,
        {
            "schema_version": "immutable-input-attestation/v1",
            "model_id": model_id,
            "model_revision": model_revision,
            "data_identity": data_identity or STATIC_DATA_IDENTITY,
            "resolved_input_kind": resolved_input_kind,
            "resolved_inputs": captured_inputs,
            "resolved_input_sha256": resolved_input_sha256(
                input_kind=resolved_input_kind, inputs=captured_inputs
            ),
        },
    )
    inspect_path = repeat / "owned-container-create-inspect.json"
    _write(
        inspect_path,
        [
            {
                "Id": container_id,
                "Image": runtime_image_digest,
                "Config": {
                    "Env": [
                        "ASCEND_VISIBLE_DEVICES=" + ",".join(map(str, devices)),
                        "ASCEND_RT_VISIBLE_DEVICES=" + ",".join(map(str, devices)),
                    ]
                },
                "HostConfig": {
                    "Privileged": False,
                    "Devices": [
                        {
                            "PathOnHost": f"/dev/davinci{device}",
                            "PathInContainer": f"/dev/davinci{device}",
                        }
                        for device in devices
                    ]
                    + [
                        {
                            "PathOnHost": f"/dev/{name}",
                            "PathInContainer": f"/dev/{name}",
                        }
                        for name in ("davinci_manager", "devmm_svm", "hisi_hdc")
                    ],
                },
                "Mounts": _ascend_mounts(),
            }
        ],
    )
    create_argv_path = repeat / "runtime/docker-create-argv.json"
    _write(
        create_argv_path,
        [
            "docker",
            "create",
            *_ascend_mount_argv(),
            *[
                item
                for value in (
                    "ASCEND_VISIBLE_DEVICES=" + ",".join(map(str, devices)),
                    "ASCEND_RT_VISIBLE_DEVICES=" + ",".join(map(str, devices)),
                )
                for item in ("--env", value)
            ],
            *[
                item
                for value in (
                    *(
                        f"/dev/davinci{device}:/dev/davinci{device}"
                        for device in devices
                    ),
                    "/dev/davinci_manager:/dev/davinci_manager",
                    "/dev/devmm_svm:/dev/devmm_svm",
                    "/dev/hisi_hdc:/dev/hisi_hdc",
                )
                for item in ("--device", value)
            ],
            runtime_image_digest,
            "runner",
        ],
    )
    runtime_identity_path = repeat / "owned-container-identity.json"
    _write(
        runtime_identity_path,
        {
            "container_id": container_id,
            "runtime_image_digest": runtime_image_digest,
            "owned_runtime_security": {
                "schema_version": "owned-runtime-security/v1",
                "privileged": False,
                "authorization_source": None,
            },
            "runtime_storage_identity": {
                "transport": "registry",
                "runtime_config_digest": runtime_image_digest,
                "local_create_ref": runtime_image_digest,
            },
            "inspect": {
                "path": inspect_path.name,
                "sha256": hashlib.sha256(inspect_path.read_bytes()).hexdigest(),
            },
            "create_argv": _hashed_record(repeat, create_argv_path),
            "runner_argv": ["runner"],
            "device_node_mapping": {
                str(device): {
                    "host": f"/dev/davinci{device}",
                    "container": f"/dev/davinci{device}",
                }
                for device in devices
            },
            "physical_to_logical_rank": {
                str(device): rank for rank, device in enumerate(devices)
            },
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
            "runtime_storage_identity": {
                "path": runtime_identity_path.name,
                "sha256": hashlib.sha256(
                    runtime_identity_path.read_bytes()
                ).hexdigest(),
            },
            "owned_runtime_security": {
                "schema_version": "owned-runtime-security/v1",
                "privileged": False,
                "authorization_source": None,
            },
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


def _relocate_service_ports(
    repo: Path,
    staged: Path,
    results: Path,
    entry: dict,
    target: dict,
    *,
    port: int,
) -> None:
    target["server_parameters"]["port"] = 8000
    target["workload"]["client_parameters"]["port"] = 8000
    registry_path = repo / "leaderboard-data" / "official-targets.json"
    _write(registry_path, {"targets": [target]})
    (registry_path.parent / "official-targets.sha256").write_text(
        hashlib.sha256(registry_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    entry["same_spec"]["resolved_server_parameters"]["port"] = port
    entry["same_spec"]["resolved_client_parameters"]["port"] = port
    _write(staged / "run_leaderboard.json", entry)
    for repeat in sorted(results.glob("repeat-*")):
        _write(repeat / "submission" / "run_leaderboard.json", entry)
        strict_path = repeat / "strict_execution_evidence.json"
        strict = json.loads(strict_path.read_text(encoding="utf-8"))
        strict["service_port"] = port
        cleanup_path = repeat / strict["cleanup"]["path"]
        cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
        cleanup["service_port"] = port
        _write(cleanup_path, cleanup)
        strict["cleanup"] = _hashed_record(repeat, cleanup_path)
        _write(strict_path, strict)


def _set_privileged_runtime(results: Path, authorization_source: str | None) -> None:
    security = {
        "schema_version": "owned-runtime-security/v1",
        "privileged": True,
        "authorization_source": authorization_source,
    }
    for repeat in sorted(results.glob("repeat-*")):
        strict_path = repeat / "strict_execution_evidence.json"
        strict = json.loads(strict_path.read_text(encoding="utf-8"))
        identity_path = repeat / strict["runtime_storage_identity"]["path"]
        identity = json.loads(identity_path.read_text(encoding="utf-8"))

        inspect_path = repeat / identity["inspect"]["path"]
        inspect = json.loads(inspect_path.read_text(encoding="utf-8"))
        inspect[0]["HostConfig"]["Privileged"] = True
        _write(inspect_path, inspect)
        identity["inspect"] = _hashed_record(repeat, inspect_path)

        create_path = repeat / identity["create_argv"]["path"]
        create_argv = json.loads(create_path.read_text(encoding="utf-8"))
        create_argv.insert(2, "--privileged")
        _write(create_path, create_argv)
        identity["create_argv"] = _hashed_record(repeat, create_path)

        identity["owned_runtime_security"] = security
        _write(identity_path, identity)
        strict["runtime_storage_identity"] = _hashed_record(repeat, identity_path)
        strict["owned_runtime_security"] = security
        _write(strict_path, strict)


def test_accepts_explicitly_authorized_privileged_owned_runtime(
    tmp_path: Path,
) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    _set_privileged_runtime(results, "user-explicit:thread-019fc873:2026-08-05")
    attest_completed_baseline(
        repo, staged, results, repo / "out-privileged", verified_by="test-review"
    )


def test_rejects_privileged_owned_runtime_without_authorization(
    tmp_path: Path,
) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    _set_privileged_runtime(results, None)
    with pytest.raises(ValueError, match="lacks user authorization"):
        attest_completed_baseline(
            repo, staged, results, repo / "out-unauthorized", verified_by="test-review"
        )


def test_rejects_privileged_owned_runtime_argv_drift(tmp_path: Path) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    _set_privileged_runtime(results, "user-explicit:thread-019fc873:2026-08-05")
    repeat = results / "repeat-01"
    strict_path = repeat / "strict_execution_evidence.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    identity_path = repeat / strict["runtime_storage_identity"]["path"]
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    create_path = repeat / identity["create_argv"]["path"]
    create_argv = json.loads(create_path.read_text(encoding="utf-8"))
    create_argv.insert(2, "--privileged")
    _write(create_path, create_argv)
    identity["create_argv"] = _hashed_record(repeat, create_path)
    _write(identity_path, identity)
    strict["runtime_storage_identity"] = _hashed_record(repeat, identity_path)
    _write(strict_path, strict)
    with pytest.raises(ValueError, match="create privilege mismatch"):
        attest_completed_baseline(
            repo, staged, results, repo / "out-argv-drift", verified_by="test-review"
        )


def test_rejects_nonprivileged_runtime_with_alternate_privilege_argv(
    tmp_path: Path,
) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    repeat = results / "repeat-01"
    strict_path = repeat / "strict_execution_evidence.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    identity_path = repeat / strict["runtime_storage_identity"]["path"]
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    create_path = repeat / identity["create_argv"]["path"]
    create_argv = json.loads(create_path.read_text(encoding="utf-8"))
    create_argv.insert(2, "--privileged=false")
    _write(create_path, create_argv)
    identity["create_argv"] = _hashed_record(repeat, create_path)
    _write(identity_path, identity)
    strict["runtime_storage_identity"] = _hashed_record(repeat, identity_path)
    _write(strict_path, strict)
    with pytest.raises(ValueError, match="create privilege mismatch"):
        attest_completed_baseline(
            repo, staged, results, repo / "out-alt-argv", verified_by="test-review"
        )


def _materialize_docker_archive_runtime(results: Path, target: dict) -> tuple[str, str]:
    config_blob = b'{"architecture":"arm64"}'
    config_digest = "sha256:" + hashlib.sha256(config_blob).hexdigest()
    layer_digest = "sha256:" + "1" * 64
    manifest_blob = json.dumps(
        {
            "config": {"digest": config_digest},
            "layers": [{"digest": layer_digest, "size": 123}],
        },
        separators=(",", ":"),
    ).encode()
    storage_digest = "sha256:" + hashlib.sha256(manifest_blob).hexdigest()
    archive_digest = "sha256:" + "d" * 64
    packages = {
        "datasets": "3.3.0",
        "torch": "2.9.0+cpu",
        "torch-npu": "2.9.0.post1",
        "vllm": "0.18.0+empty",
        "vllm-ascend": "0.18.0",
        "xxhash": "3.6.0",
    }
    target["baseline_runtime"].update(
        {
            "runtime_transport": "docker-archive",
            "runtime_image": None,
            "runtime_image_digest": config_digest,
            "runtime_config_digest": config_digest,
            "runtime_archive_sha256": archive_digest,
            "containerd_storage_manifest_digest": storage_digest,
            "core_commit": "core-sha",
            "backend_commit": "plugin-sha",
            "runtime_packages": packages,
        }
    )
    for repeat in sorted(results.glob("repeat-*")):
        strict = json.loads(
            (repeat / "strict_execution_evidence.json").read_text(encoding="utf-8")
        )
        container_id = strict["container_id"]
        startup_id = strict["startup_instance_id"]
        devices = strict["lease"]["physical_npu_ids"]
        manifest_path = repeat / "runtime/containerd-storage-manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(manifest_blob)
        config_path = repeat / "runtime/containerd-config-blob.json"
        config_path.write_bytes(config_blob)
        content_list = repeat / "runtime/containerd-content-list.txt"
        content_list.write_text(f"{layer_digest}\n", encoding="utf-8")
        archive_identity = repeat / "runtime/archive-identity.json"
        _write(
            archive_identity,
            {
                "path": "/var/tmp/vllm-ascend-strict-baseline.tar.zst",
                "size_bytes": 123,
                "sha256": archive_digest,
            },
        )
        docker_image_inspect = repeat / "runtime/docker-image-inspect.json"
        _write(docker_image_inspect, [{"Id": storage_digest}])
        expected_runtime = repeat / "runtime/expected-runtime.json"
        _write(
            expected_runtime,
            {
                "schema_version": "strict-owned-runtime-expected/v1",
                "core_commit": "core-sha",
                "backend_commit": "plugin-sha",
                "packages": packages,
            },
        )
        expected_sha256 = hashlib.sha256(expected_runtime.read_bytes()).hexdigest()
        container_identity = repeat / "runtime/container-identity.json"
        _write(
            container_identity,
            {"startup_instance_id": startup_id, "container_id": container_id},
        )
        identity_host_source = (
            repeat.parents[3] / "root-identities" / f"{repeat.name}.json"
        )
        identity_host_source.parent.mkdir(parents=True, exist_ok=True)
        identity_host_source.write_bytes(container_identity.read_bytes())
        actual_runtime = repeat / "runtime/actual-runtime.json"
        _write(
            actual_runtime,
            {
                "schema_version": "strict-owned-runtime-preflight/v1",
                "startup_instance_id": startup_id,
                "container_id": container_id,
                "expected_contract_sha256": expected_sha256,
                "sources": {
                    "core": {"commit": "core-sha", "clean": True},
                    "backend": {"commit": "plugin-sha", "clean": True},
                },
                "packages": packages,
            },
        )
        repo_root = repeat.parents[2]
        expected_process_argv = [
            "/usr/local/python3.11.14/bin/python",
            "/workspace/vllm-hust-benchmark/scripts/verify-owned-runtime-and-exec.py",
            "--expected",
            "/workspace/vllm-hust-benchmark/"
            + expected_runtime.relative_to(repo_root).as_posix(),
            "--expected-sha256",
            expected_sha256,
            "--container-identity",
            "/run/vllm-hust/container-identity.json",
            "--output",
            "/workspace/vllm-hust-benchmark/"
            + actual_runtime.relative_to(repo_root).as_posix(),
            "--startup-instance-id",
            startup_id,
            "--",
            "runner",
        ]
        container_inspect = repeat / "owned-container-create-inspect.json"
        _write(
            container_inspect,
            [
                {
                    "Id": container_id,
                    "Image": storage_digest,
                    "Path": expected_process_argv[0],
                    "Args": expected_process_argv[1:],
                    "Config": {
                        "Entrypoint": [expected_process_argv[0]],
                        "Cmd": expected_process_argv[1:],
                        "Env": [
                            "ASCEND_VISIBLE_DEVICES=" + ",".join(map(str, devices)),
                            "ASCEND_RT_VISIBLE_DEVICES=" + ",".join(map(str, devices)),
                        ],
                    },
                    "HostConfig": {
                        "Privileged": False,
                        "Devices": [
                            {
                                "PathOnHost": f"/dev/davinci{device}",
                                "PathInContainer": f"/dev/davinci{device}",
                            }
                            for device in devices
                        ]
                        + [
                            {
                                "PathOnHost": f"/dev/{name}",
                                "PathInContainer": f"/dev/{name}",
                            }
                            for name in ("davinci_manager", "devmm_svm", "hisi_hdc")
                        ],
                    },
                    "Mounts": [
                        *_ascend_mounts(),
                        {
                            "Source": str(repo_root),
                            "Destination": "/workspace/vllm-hust-benchmark",
                            "RW": True,
                        },
                        {
                            "Source": str(identity_host_source),
                            "Destination": "/run/vllm-hust/container-identity.json",
                            "RW": False,
                        },
                    ],
                }
            ],
        )
        create_argv_path = repeat / "runtime/docker-create-argv.json"
        _write(
            create_argv_path,
            [
                "docker",
                "create",
                "--pull=never",
                "--entrypoint",
                expected_process_argv[0],
                "--mount",
                (
                    f"type=bind,src={identity_host_source},"
                    "dst=/run/vllm-hust/container-identity.json,readonly"
                ),
                *_ascend_mount_argv(),
                *[
                    item
                    for value in (
                        "ASCEND_VISIBLE_DEVICES=" + ",".join(map(str, devices)),
                        "ASCEND_RT_VISIBLE_DEVICES=" + ",".join(map(str, devices)),
                    )
                    for item in ("--env", value)
                ],
                *[
                    item
                    for value in (
                        *(
                            f"/dev/davinci{device}:/dev/davinci{device}"
                            for device in devices
                        ),
                        "/dev/davinci_manager:/dev/davinci_manager",
                        "/dev/devmm_svm:/dev/devmm_svm",
                        "/dev/hisi_hdc:/dev/hisi_hdc",
                    )
                    for item in ("--device", value)
                ],
                storage_digest,
                *expected_process_argv[1:],
            ],
        )
        runtime_identity = repeat / "owned-container-identity.json"
        _write(
            runtime_identity,
            {
                "container_id": container_id,
                "runtime_image_digest": config_digest,
                "owned_runtime_security": {
                    "schema_version": "owned-runtime-security/v1",
                    "privileged": False,
                    "authorization_source": None,
                },
                "runtime_storage_identity": {
                    "transport": "docker-archive",
                    "runtime_config_digest": config_digest,
                    "containerd_storage_manifest_digest": storage_digest,
                    "local_create_ref": storage_digest,
                    "manifest_config_digest": config_digest,
                    "raw_manifest": _hashed_record(repeat, manifest_path),
                    "raw_config_blob": _hashed_record(repeat, config_path),
                    "layers": [
                        {
                            "digest": layer_digest,
                            "size": 123,
                        }
                    ],
                    "content_list": _hashed_record(repeat, content_list),
                    "archive": _hashed_record(repeat, archive_identity),
                    "docker_image_inspect": _hashed_record(
                        repeat, docker_image_inspect
                    ),
                },
                "inspect": _hashed_record(repeat, container_inspect),
                "create_argv": _hashed_record(repeat, create_argv_path),
                "runner_argv": ["runner"],
                "device_node_mapping": {
                    str(device): {
                        "host": f"/dev/davinci{device}",
                        "container": f"/dev/davinci{device}",
                    }
                    for device in devices
                },
                "physical_to_logical_rank": {
                    str(device): rank for rank, device in enumerate(devices)
                },
                "expected_runtime_contract": _hashed_record(repeat, expected_runtime),
                "container_identity": _hashed_record(repeat, container_identity),
                "container_identity_host_source": str(identity_host_source),
                "actual_runtime_preflight": _hashed_record(repeat, actual_runtime),
            },
        )
        strict_path = repeat / "strict_execution_evidence.json"
        strict["runtime_image_digest"] = config_digest
        strict["runtime_storage_identity"] = _hashed_record(repeat, runtime_identity)
        _write(strict_path, strict)
    return config_digest, storage_digest


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
    assert suite["repeats"][0]["resolved_input_sha256"] == resolved_input_sha256(
        input_kind="throughput-sample-requests", inputs=[{"prompt": "fixed"}]
    )
    assert len(suite["repeats"][0]["immutable_input_attestation_sha256"]) == 64


def test_normalizes_performance_equivalent_loopback_port_relocation(
    tmp_path: Path,
) -> None:
    repo, staged, results, entry, target = _fixture(tmp_path)
    _relocate_service_ports(repo, staged, results, entry, target, port=18123)
    output = repo / "submissions" / "target-1"

    attested = attest_completed_baseline(
        repo,
        staged,
        results,
        output,
        verified_by="test-review",
        verified_at="2026-08-02T00:00:00Z",
    )

    assert attested["same_spec"]["resolved_server_parameters"]["port"] == 8000
    assert attested["same_spec"]["resolved_client_parameters"]["port"] == 8000
    normalization = attested["metadata"]["transport_port_normalization"]
    assert normalization["actual_service_port"] == 18123
    assert normalization["performance_metrics_modified"] is False
    suite = json.loads((output / "repeat_suite.json").read_text())
    assert suite["repeats"][0]["service_port"] == 18123
    assert suite["repeats"][0]["transport_port_relocation"] == {
        "official_port": 8000,
        "actual_service_port": 18123,
    }


def test_rejects_port_relocation_not_bound_to_strict_evidence(
    tmp_path: Path,
) -> None:
    repo, staged, results, entry, target = _fixture(tmp_path)
    _relocate_service_ports(repo, staged, results, entry, target, port=18123)
    strict_path = results / "repeat-02" / "strict_execution_evidence.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    strict["service_port"] = 18124
    cleanup_path = results / "repeat-02" / strict["cleanup"]["path"]
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
    cleanup["service_port"] = 18124
    _write(cleanup_path, cleanup)
    strict["cleanup"] = _hashed_record(results / "repeat-02", cleanup_path)
    _write(strict_path, strict)

    with pytest.raises(ValueError, match="resolved service port"):
        attest_completed_baseline(
            repo,
            staged,
            results,
            repo / "submissions" / "target-1",
            verified_by="test-review",
        )


def test_docker_archive_runtime_uses_config_digest_for_compatibility(
    tmp_path: Path,
) -> None:
    repo, staged, results, _, target = _fixture(tmp_path)
    config_digest, storage_digest = _materialize_docker_archive_runtime(results, target)
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
        repo / "out-docker-archive",
        verified_by="test-review",
    )

    strict_path = results / "repeat-02" / "strict_execution_evidence.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    assert strict["runtime_image_digest"] == config_digest
    strict["runtime_image_digest"] = storage_digest
    _write(strict_path, strict)
    with pytest.raises(ValueError, match="strict execution runtime image mismatch"):
        attest_completed_baseline(
            repo,
            staged,
            results,
            repo / "out-storage-digest",
            verified_by="test-review",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("inspect_only_image", "inspect image mismatch"),
        ("wrapper_bypass", "preflight entrypoint mismatch"),
        ("path_drift", "preflight command mismatch"),
        ("args_drift", "preflight command mismatch"),
        ("cmd_drift", "preflight command mismatch"),
        ("privileged", "privilege mismatch"),
        ("missing_ascend_mount", "Ascend mount mismatch"),
        ("writable_ascend_mount", "Ascend mount mismatch"),
        ("device_node_remap", "physical device scope mismatch"),
        ("create_device_drift", "create physical device scope mismatch"),
        ("visible_env_drift", "visible device env mismatch"),
        ("create_visible_env_drift", "create visible env mismatch"),
        ("logical_rank_drift", "device/rank identity mismatch"),
        ("create_argv_drift", "create/preflight command relationship mismatch"),
        ("writable_alias", "reachable through a writable alias"),
        ("contract_hash_drift", "preflight binding mismatch"),
        ("container_mismatch", "preflight binding mismatch"),
        ("startup_mismatch", "preflight binding mismatch"),
    ),
)
def test_rejects_unbound_docker_archive_preflight(
    tmp_path: Path, mutation: str, message: str
) -> None:
    repo, staged, results, _, target = _fixture(tmp_path)
    _materialize_docker_archive_runtime(results, target)
    registry_path = repo / "leaderboard-data" / "official-targets.json"
    _write(registry_path, {"targets": [target]})
    (registry_path.parent / "official-targets.sha256").write_text(
        hashlib.sha256(registry_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    repeat = results / "repeat-01"
    identity_path = repeat / "owned-container-identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if mutation in {
        "inspect_only_image",
        "wrapper_bypass",
        "path_drift",
        "args_drift",
        "cmd_drift",
        "privileged",
        "missing_ascend_mount",
        "writable_ascend_mount",
        "device_node_remap",
        "visible_env_drift",
        "writable_alias",
    }:
        inspect_path = repeat / identity["inspect"]["path"]
        inspect = json.loads(inspect_path.read_text(encoding="utf-8"))
        if mutation == "inspect_only_image":
            inspect[0] = {"Image": inspect[0]["Image"]}
        elif mutation == "wrapper_bypass":
            inspect[0]["Config"]["Entrypoint"] = ["/bin/bash"]
        elif mutation == "path_drift":
            inspect[0]["Path"] = "/image/default-entrypoint"
        elif mutation == "args_drift":
            inspect[0]["Args"].append("drift")
        elif mutation == "cmd_drift":
            inspect[0]["Config"]["Cmd"].append("drift")
        elif mutation == "privileged":
            inspect[0]["HostConfig"]["Privileged"] = True
        elif mutation == "missing_ascend_mount":
            inspect[0]["Mounts"] = [
                mount
                for mount in inspect[0]["Mounts"]
                if mount.get("Destination") != "/etc/ascend_install.info"
            ]
        elif mutation == "writable_ascend_mount":
            next(
                mount
                for mount in inspect[0]["Mounts"]
                if mount.get("Destination") == "/etc/ascend_install.info"
            )["RW"] = True
        elif mutation == "device_node_remap":
            inspect[0]["HostConfig"]["Devices"][0]["PathInContainer"] = "/dev/davinci7"
        elif mutation == "visible_env_drift":
            inspect[0]["Config"]["Env"][0] = "ASCEND_VISIBLE_DEVICES=7"
        else:
            inspect[0]["Mounts"].append(
                {
                    "Source": identity["container_identity_host_source"],
                    "Destination": "/workspace/identity-alias.json",
                    "RW": True,
                }
            )
        _write(inspect_path, inspect)
        identity["inspect"] = _hashed_record(repeat, inspect_path)
    elif mutation in {
        "create_argv_drift",
        "create_visible_env_drift",
        "create_device_drift",
    }:
        create_path = repeat / identity["create_argv"]["path"]
        create_argv = json.loads(create_path.read_text(encoding="utf-8"))
        if mutation == "create_argv_drift":
            create_argv.append("drift")
        elif mutation == "create_visible_env_drift":
            create_argv[create_argv.index("ASCEND_VISIBLE_DEVICES=0")] = (
                "ASCEND_VISIBLE_DEVICES=7"
            )
        else:
            create_argv[create_argv.index("/dev/davinci0:/dev/davinci0")] = (
                "/dev/davinci0:/dev/davinci7"
            )
        _write(create_path, create_argv)
        identity["create_argv"] = _hashed_record(repeat, create_path)
    elif mutation == "logical_rank_drift":
        identity["physical_to_logical_rank"] = {"0": 1}
    else:
        actual_path = repeat / identity["actual_runtime_preflight"]["path"]
        actual = json.loads(actual_path.read_text(encoding="utf-8"))
        if mutation == "contract_hash_drift":
            actual["expected_contract_sha256"] = "0" * 64
        elif mutation == "container_mismatch":
            actual["container_id"] = "f" * 64
        else:
            actual["startup_instance_id"] = "other-startup"
        _write(actual_path, actual)
        identity["actual_runtime_preflight"] = _hashed_record(repeat, actual_path)
    _write(identity_path, identity)
    strict_path = repeat / "strict_execution_evidence.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    strict["runtime_storage_identity"] = _hashed_record(repeat, identity_path)
    _write(strict_path, strict)
    with pytest.raises(ValueError, match=message):
        attest_completed_baseline(
            repo, staged, results, repo / f"out-{mutation}", verified_by="test-review"
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
        resolved_inputs=[{"prompt": "changed"}],
        resolved_input_sha256=resolved_input_sha256(
            input_kind="throughput-sample-requests",
            inputs=[{"prompt": "changed"}],
        ),
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
