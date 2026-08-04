from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_hust_benchmark import strict_repeat_orchestrator as orchestrator
from vllm_hust_benchmark.official_baseline_attestation import (
    _validate_strict_execution_evidence,
)


NPU_SMI_FIXTURE = """
| NPU   Name  |
| 0     910B2 |
| state | ok | 3451 / 65536 |
| 7     910B2 |
| state | ok | 3419 / 65536 |
| NPU   Chip   Process id   Process name   Process memory(MB) |
| 0     0      3072791      VLLMWorker_TP  28170              |
| 7     0      3074335      VLLMWorker_TP  28170              |
"""


def test_parses_host_npu_smi_fixture() -> None:
    assert orchestrator.parse_hbm_usage(NPU_SMI_FIXTURE) == {
        0: (3451, 65536),
        7: (3419, 65536),
    }
    assert orchestrator.parse_compute_pids(NPU_SMI_FIXTURE) == {
        0: [3072791],
        7: [3074335],
    }


def test_resource_lease_blocks_overlapping_card(tmp_path: Path) -> None:
    first = orchestrator.ResourceLease(
        tmp_path,
        startup_id="first",
        container_id="first-container",
        devices=[0],
        port=18080,
        repeat_dir=tmp_path / "one",
    )
    second = orchestrator.ResourceLease(
        tmp_path,
        startup_id="second",
        container_id="second-container",
        devices=[0],
        port=18081,
        repeat_dir=tmp_path / "two",
    )
    first.acquire()
    try:
        with pytest.raises(orchestrator.GateFailure, match="npu-0"):
            second.acquire()
    finally:
        first.mark_released()


def test_session_filter_excludes_complete_ancestor_chain() -> None:
    conflicts = orchestrator.benchmark_session_conflicts(
        tmux_output="20\ttarget-a:18080\t1\t1\n",
        screen_output="30.target-a-18080\n",
        process_output=(
            "  10 1 root cmd --target target-a --port :18080\n"
            "  20 10 user parent target-a :18080\n"
            "  30 20 user orchestrator target-a :18080\n"
            "  99 1 other external-benchmark target-a :18080\n"
        ),
        target_id="target-a",
        service_port=18080,
        excluded_pids={10, 20, 30},
    )
    assert conflicts == ["  99 1 other external-benchmark target-a :18080"]


def test_process_ancestor_chain_contains_self_and_init() -> None:
    ancestors = orchestrator.process_ancestor_pids()
    assert orchestrator.os.getpid() in ancestors
    assert 1 in ancestors


def test_canonical_worker_rule_prefers_worker_then_lowest_pid() -> None:
    records = [
        {"host_pid": 100, "cmdline": "EngineCore"},
        {"host_pid": 300, "cmdline": "VLLMWorker_TP rank=1"},
        {"host_pid": 200, "cmdline": "VLLMWorker rank=0"},
    ]
    assert min(records, key=orchestrator.canonical_worker_key)["host_pid"] == 200


def _orchestrator_stub(tmp_path: Path) -> orchestrator.StrictRepeatOrchestrator:
    instance = object.__new__(orchestrator.StrictRepeatOrchestrator)
    instance.args = SimpleNamespace(
        repo_host_path=tmp_path / "repo",
        shared_data_host_path=tmp_path / "shared",
        runtime_image_digest="sha256:" + "a" * 64,
        command=["bash", "/workspace/vllm-hust-benchmark/run-one.sh"],
    )
    instance.devices = [7]
    instance.startup_id = "b" * 32
    instance.container_name = "vllm-hust-strict-repeat-" + instance.startup_id
    instance.container_repeat_dir = Path(
        "/workspace/vllm-hust-benchmark/results/repeat-01"
    )
    return instance


def test_docker_create_is_owned_ephemeral_and_scoped(tmp_path: Path) -> None:
    instance = _orchestrator_stub(tmp_path)
    argv = instance._docker_create_argv()
    rendered = " ".join(argv)
    assert argv[:2] == ["docker", "create"]
    assert "--rm" in argv
    assert "--network none" in rendered
    assert "vllm-hust.strict-startup-id=" + "b" * 32 in argv
    assert "/dev/davinci7:/dev/davinci0" in argv
    assert "/dev/davinci0:/dev/davinci0" not in argv
    assert "vllm-hust-shuhao-21rc" not in rendered
    assert "dst=/data/shared_models,readonly" in rendered
    assert "dst=/data/shared_datasets,readonly" in rendered
    assert "VLLM_HUST_STRICT_HOST_ORCHESTRATED=1" in argv
    assert any(
        value.startswith("VLLM_HUST_STRICT_HOST_PEAK_HBM_FILE=") for value in argv
    )


def test_owned_container_inspect_rejects_extra_npu_mapping(tmp_path: Path) -> None:
    instance = _orchestrator_stub(tmp_path)
    instance.container_id = "c" * 64
    devices = [
        {"PathOnHost": "/dev/davinci7", "PathInContainer": "/dev/davinci0"},
        *[
            {"PathOnHost": f"/dev/{name}", "PathInContainer": f"/dev/{name}"}
            for name in ("davinci_manager", "devmm_svm", "hisi_hdc")
        ],
    ]
    record = {
        "Id": instance.container_id,
        "Image": instance.args.runtime_image_digest,
        "Config": {"Labels": {"vllm-hust.strict-startup-id": instance.startup_id}},
        "HostConfig": {
            "AutoRemove": True,
            "NetworkMode": "none",
            "IpcMode": "host",
            "Devices": devices,
        },
        "Mounts": [
            {
                "Source": str(instance.args.repo_host_path.resolve()),
                "Destination": "/workspace/vllm-hust-benchmark",
                "RW": True,
            },
            {
                "Source": str(instance.args.shared_data_host_path.resolve()),
                "Destination": "/data/shared_models",
                "RW": False,
            },
            {
                "Source": str(instance.args.shared_data_host_path.resolve()),
                "Destination": "/data/shared_datasets",
                "RW": False,
            },
            {
                "Source": "/usr/local/Ascend/driver",
                "Destination": "/usr/local/Ascend/driver",
                "RW": False,
            },
        ],
    }
    instance._validate_owned_container_inspect(record)
    record["HostConfig"]["Devices"].append(
        {"PathOnHost": "/dev/davinci0", "PathInContainer": "/dev/davinci1"}
    )
    with pytest.raises(orchestrator.GateFailure, match="device mapping"):
        instance._validate_owned_container_inspect(record)


class _CleanupRoot:
    prefix: list[str] = []

    def __init__(self, inspect_payload: dict) -> None:
        self.inspect_payload = inspect_payload
        self.calls: list[list[str]] = []

    def run(self, argv, *, check=True):
        self.calls.append(list(argv))
        if argv[:2] == ["docker", "inspect"]:
            return orchestrator.CommandResult(json.dumps([self.inspect_payload]), "", 0)
        return orchestrator.CommandResult("", "", 0)


def test_cleanup_refuses_container_without_owned_label(tmp_path: Path) -> None:
    instance = _orchestrator_stub(tmp_path)
    instance.container_id = "c" * 64
    instance.root = _CleanupRoot(
        {
            "Id": instance.container_id,
            "Config": {"Labels": {"vllm-hust.strict-startup-id": "other"}},
            "State": {"Running": True},
        }
    )
    with pytest.raises(orchestrator.GateFailure, match="ownership label"):
        instance._cleanup_owned_container(allow_stop=True)
    assert not any(call[:2] == ["docker", "stop"] for call in instance.root.calls)
    assert not any(call[:2] == ["docker", "rm"] for call in instance.root.calls)


def test_cli_rejects_snapshot_interval_below_fifteen(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        orchestrator.parse_args(
            [
                "--repeat-dir",
                str(tmp_path / "repeat"),
                "--target-id",
                "target",
                "--side",
                "upstream",
                "--physical-npu",
                "0",
                "--service-port",
                "18080",
                "--runtime-image-digest",
                "sha256:" + "a" * 64,
                "--snapshot-interval-seconds",
                "14.99",
                "--",
                "true",
            ]
        )


def test_cli_rejects_relative_host_library_path(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        orchestrator.parse_args(
            [
                "--repeat-dir",
                str(tmp_path / "repeat"),
                "--target-id",
                "target",
                "--side",
                "upstream",
                "--physical-npu",
                "0",
                "--service-port",
                "18080",
                "--runtime-image-digest",
                "sha256:" + "a" * 64,
                "--host-ld-library-path",
                "relative/path",
                "--",
                "true",
            ]
        )


def _trusted_executable_fixture(tmp_path: Path) -> Path:
    executable = tmp_path / "usr" / "local" / "sbin" / "npu-smi"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o555)
    for directory in (
        tmp_path,
        tmp_path / "usr",
        tmp_path / "usr/local",
        executable.parent,
    ):
        directory.chmod(0o755)
    return executable


def _validate_fixture(executable: Path, floor: Path) -> None:
    orchestrator.validate_absolute_executable(
        executable, trusted_uid=executable.stat().st_uid, ancestor_floor=floor
    )


def test_fixed_host_executable_accepts_complete_safe_chain(tmp_path: Path) -> None:
    executable = _trusted_executable_fixture(tmp_path)
    _validate_fixture(executable, tmp_path)


def test_fixed_host_executable_rejects_writable_parent(tmp_path: Path) -> None:
    executable = _trusted_executable_fixture(tmp_path)
    (tmp_path / "usr/local").chmod(0o777)
    with pytest.raises(orchestrator.GateFailure, match="replaceable"):
        _validate_fixture(executable, tmp_path)


def test_fixed_host_executable_rejects_symlink_replacement(tmp_path: Path) -> None:
    executable = _trusted_executable_fixture(tmp_path)
    replacement = tmp_path / "replacement"
    replacement.write_text("#!/bin/sh\n", encoding="utf-8")
    replacement.chmod(0o555)
    executable.unlink()
    executable.symlink_to(replacement)
    with pytest.raises(orchestrator.GateFailure, match="symlink"):
        _validate_fixture(executable, tmp_path)


def test_fixed_host_executable_rejects_writable_file(tmp_path: Path) -> None:
    executable = _trusted_executable_fixture(tmp_path)
    executable.chmod(0o775)
    with pytest.raises(orchestrator.GateFailure, match="replaceable"):
        _validate_fixture(executable, tmp_path)


def test_cli_rejects_removed_host_library_path(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        orchestrator.parse_args(
            [
                "--repeat-dir",
                str(tmp_path / "repeat"),
                "--target-id",
                "target",
                "--side",
                "upstream",
                "--physical-npu",
                "0",
                "--service-port",
                "18080",
                "--runtime-image-digest",
                "sha256:" + "a" * 64,
                "--host-ld-library-path",
                "/tmp",
                "--",
                "true",
            ]
        )


def test_root_commands_scopes_host_library_path_to_npu_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        environments.append(kwargs["env"])
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(orchestrator.os, "geteuid", lambda: 0)
    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    monkeypatch.setattr(
        orchestrator, "secure_host_npu_smi", lambda: "/usr/local/sbin/npu-smi"
    )
    root = orchestrator.RootCommands()
    root.run(["npu-smi", "info"])
    root.run(["docker", "ps", "-aq"])
    assert calls[0] == [
        "/usr/local/sbin/npu-smi",
        "info",
    ]
    assert calls[1] == ["docker", "ps", "-aq"]
    assert environments == [
        {"PATH": orchestrator.SAFE_HOST_PATH, "LANG": "C.UTF-8"},
        {"PATH": orchestrator.SAFE_HOST_PATH, "LANG": "C.UTF-8"},
    ]
    assert all("LD_LIBRARY_PATH" not in environment for environment in environments)


def test_dry_run_never_writes_canonical_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeRoot:
        prefix: list[str] = []

        def __init__(self) -> None:
            pass

        def run(self, argv, *, check=True):
            return orchestrator.CommandResult("", "", 0)

    monkeypatch.setattr(orchestrator, "RootCommands", FakeRoot)
    monkeypatch.setattr(
        orchestrator.StrictRepeatOrchestrator,
        "_validate_target_scope",
        lambda _self: None,
    )
    monkeypatch.setattr(orchestrator.time, "sleep", lambda _seconds: None)
    args = SimpleNamespace(
        repeat_dir=tmp_path / "repo" / "results" / "repeat",
        target_id=(
            "official-ascend-jan-2026-v0.18.0-instructcoder-online-"
            "qwen25-coder-14b-910b2"
        ),
        side="upstream",
        physical_npu=[0],
        service_port=18080,
        runtime_image_digest="sha256:" + "a" * 64,
        container_name_prefix="strict-test",
        repo_host_path=tmp_path / "repo",
        shared_data_host_path=tmp_path / "shared",
        lease_dir=tmp_path / "leases",
        snapshot_interval_seconds=15,
        sample_interval_seconds=1,
        max_idle_hbm_mb=4096,
        max_hbm_drift_mb=256,
        output_uid=1000,
        output_gid=1000,
        dry_run=True,
        command=["true"],
    )
    instance = orchestrator.StrictRepeatOrchestrator(args)

    def fake_create() -> None:
        instance.container_id = "c" * 64
        instance.lease.bind_container_id(instance.container_id)

    snapshot = orchestrator.Snapshot(
        "2026-08-04T00:00:00Z",
        {0: 3451},
        {},
        {"captured_at": "2026-08-04T00:00:00Z", "stable": False},
    )
    monkeypatch.setattr(instance, "_create_owned_container", fake_create)
    monkeypatch.setattr(instance, "_host_snapshot", lambda _number: snapshot)
    monkeypatch.setattr(instance, "_cleanup_owned_container", lambda **_kwargs: None)
    assert instance.run() == 0
    assert (args.repeat_dir / "dry-run-plan.json").is_file()
    assert not (args.repeat_dir / "strict_execution_evidence.json").exists()


def test_official_runner_consumes_atomic_host_peak_file() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "run-official-ascend-goal-baseline.sh"
    ).read_text(encoding="utf-8")
    assert '"${VLLM_HUST_STRICT_HOST_ORCHESTRATED:-}" == "1"' in script
    assert 'PEAK_HBM_EVIDENCE_FILE="$VLLM_HUST_STRICT_HOST_PEAK_HBM_FILE"' in script
    assert "sample_ascend_peak_hbm.py" in script


def test_generated_payload_matches_official_validator(tmp_path: Path) -> None:
    repeat = tmp_path / "repeat-01"
    repeat.mkdir()
    container_id = "c" * 64
    runtime_digest = "sha256:" + "d" * 64
    model_revision = "a" * 40
    data_identity = {
        "kind": "vllm-repository-file",
        "path": "benchmarks/sonnet.txt",
        "sha256": "e" * 64,
    }

    def write_json(name: str, payload: object) -> Path:
        path = repeat / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path

    immutable = write_json(
        "immutable-input-attestation.json",
        {
            "schema_version": "immutable-input-attestation/v1",
            "model_id": "Qwen/model",
            "model_revision": model_revision,
            "data_identity": data_identity,
            "resolved_input_kind": "throughput-sample-requests",
            "resolved_input_sha256": "f" * 64,
        },
    )
    snapshots = []
    for index, second in enumerate((5, 20), start=1):
        npu = repeat / f"npu-{index}.txt"
        inspect = repeat / f"inspect-{index}.json"
        npu.write_text("real host snapshot\n", encoding="utf-8")
        inspect.write_text("{}\n", encoding="utf-8")
        snapshots.append(
            {
                "captured_at": f"2026-08-04T00:00:{second:02d}Z",
                "physical_npu_ids": [0],
                "external_compute_pids": [],
                "external_container_ids": [],
                "lease_conflicts": [],
                "stable": True,
                "npu_smi": orchestrator.evidence_record(repeat, npu),
                "container_inspect": orchestrator.evidence_record(repeat, inspect),
            }
        )
    hbm = repeat / "hbm-samples.jsonl"
    hbm.write_text(
        json.dumps(
            {
                "captured_at": "2026-08-04T00:00:25Z",
                "host_pids": [1234],
                "physical_npu_hbm_mb": {"0": 2048},
                "total_hbm_mb": 2048,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    owned_processes = write_json(
        "runtime/owned-processes.json",
        [
            {
                "host_pid": 1200,
                "physical_npu_id": 0,
                "container_id": container_id,
                "cgroup": f"0::/docker/{container_id}",
                "cmdline": "EngineCore",
            },
            {
                "host_pid": 1234,
                "physical_npu_id": 0,
                "container_id": container_id,
                "cgroup": f"0::/docker/{container_id}",
                "cmdline": "VLLMWorker_TP --rank 0",
            },
        ],
    )
    cleanup = write_json(
        "cleanup-chain-attestation.json",
        {
            "schema_version": "cleanup-chain-attestation/v1",
            "hostname": "host-a",
            "startup_instance_id": "startup-a",
            "container_id": container_id,
            "exit_code": 0,
            "host_pids": [1234],
            "all_owned_host_pids": [1200, 1234],
            "physical_npu_ids": [0],
            "service_port": 18080,
            "container_stopped_or_removed": True,
            "pids_absent": True,
            "port_released": True,
            "npu_processes_absent": True,
            "lease_released": True,
            "finished_at": "2026-08-04T00:00:29Z",
        },
    )
    payload = orchestrator.build_strict_evidence(
        hostname="host-a",
        startup_id="startup-a",
        target_id="target",
        side="upstream",
        container_id=container_id,
        service_port=18080,
        runtime_image_digest=runtime_digest,
        immutable_inputs=orchestrator.evidence_record(repeat, immutable),
        devices=[0],
        acquired_at="2026-08-04T00:00:00Z",
        released_at="2026-08-04T00:00:30Z",
        snapshots=snapshots,
        ownership=[
            {
                "host_pid": 1234,
                "physical_npu_id": 0,
                "container_id": container_id,
                "cgroup": f"0::/docker/{container_id}",
            }
        ],
        owned_processes={
            "selection_rule": orchestrator.CANONICAL_WORKER_RULE,
            "raw": orchestrator.evidence_record(repeat, owned_processes),
        },
        hbm_samples=orchestrator.evidence_record(repeat, hbm),
        peak_hbm_mb=2048,
        cleanup=orchestrator.evidence_record(repeat, cleanup),
    )
    write_json("strict_execution_evidence.json", payload)
    target = {
        "baseline_runtime": {"runtime_image_digest": runtime_digest},
        "hardware": {"chip_count": 1},
        "model": {"id": "Qwen/model", "revision": model_revision},
        "workload": {"name": "sonnet-throughput", "data_identity": data_identity},
    }
    validated = _validate_strict_execution_evidence(
        repeat, target, {"peak_mem_mb": 2048}
    )
    assert validated["container_id"] == container_id
    assert validated["peak_hbm_mb"] == 2048.0

    payload["ownership"][0]["host_pid"] = 1200
    write_json("strict_execution_evidence.json", payload)
    with pytest.raises(ValueError, match="canonical worker selection"):
        _validate_strict_execution_evidence(repeat, target, {"peak_mem_mb": 2048})
