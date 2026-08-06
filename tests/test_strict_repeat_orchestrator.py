from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_hust_benchmark import strict_repeat_orchestrator as orchestrator
from vllm_hust_benchmark.immutable_input_attestation import resolved_input_sha256
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

NPU_SMI_26_PROCESS_FIXTURE = """
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
| 0     910B2               | OK            | 90.0                 44                      0    / 0                |
| 0                         | 0000:C1:00.0  | 0                    0    / 0                41842/ 65536            |
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
| 0       0                 | 1732754       | python             | 150                   | NA                      |
| 0       0                 | 1732755       | python             | 151                   | NA                      |
| 0       0                 | 1732756       | python             | 152                   | NA                      |
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


def test_parses_strict_npu_smi_26_process_columns() -> None:
    assert orchestrator.parse_hbm_usage(NPU_SMI_26_PROCESS_FIXTURE) == {
        0: (41842, 65536)
    }
    assert orchestrator.parse_compute_pids(NPU_SMI_26_PROCESS_FIXTURE) == {
        0: [1732754, 1732755, 1732756]
    }


@pytest.mark.parametrize(
    "row",
    [
        "| 0 0 | 1732754 | python | 150 |",
        "| 0 x | 1732754 | python | 150 | NA |",
        "| 0 0 | not-a-pid | python | 150 | NA |",
        "| 0 0 1732754 | python | 150 | NA |",
        "| 0 0 | 1732754 | python | 150 | 123",
    ],
)
def test_rejects_malformed_npu_smi_process_columns(row: str) -> None:
    output = "| NPU | Chip | Process ID | Process name | Memory |\n" + row
    with pytest.raises(orchestrator.GateFailure, match="malformed.*process row"):
        orchestrator.parse_compute_pids(output)


def test_runtime_sample_writes_owned_pid_and_hbm_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host_pid = orchestrator.os.getpid()
    npu_smi = (
        NPU_SMI_26_PROCESS_FIXTURE.replace("1732754", str(host_pid))
        .replace(
            "| 0       0                 | 1732755       | python             | 151                   | NA                      |\n",
            "",
        )
        .replace(
            "| 0       0                 | 1732756       | python             | 152                   | NA                      |\n",
            "",
        )
    )

    class FakeRoot:
        def run(self, _argv):
            return orchestrator.CommandResult(npu_smi, "", 0)

    container_id = "c" * 64
    instance = object.__new__(orchestrator.StrictRepeatOrchestrator)
    instance.root = FakeRoot()
    instance.repeat_dir = tmp_path
    instance.devices = [0]
    instance.container_id = container_id
    instance.ownership = []
    instance.all_owned_processes = []
    instance.all_owned_host_pids = []
    instance.host_pids = []
    instance.owned_processes_path = tmp_path / "runtime" / "owned-processes.json"
    instance.hbm_path = tmp_path / "hbm-samples.jsonl"
    instance.host_peak_path = tmp_path / "strict-host-peak-hbm.json"
    instance.hbm_sample_count = 0
    instance.per_device_peaks = {0: 0}

    monkeypatch.setattr(
        orchestrator,
        "read_host_process_context",
        lambda pid: (f"0::/docker/{container_id}", f"python worker-{pid}"),
    )
    monkeypatch.setattr(orchestrator, "evidence_record", lambda *_args: {"sha256": "x"})
    assert instance._runtime_sample(1) is True

    owned = json.loads(instance.owned_processes_path.read_text(encoding="utf-8"))
    assert [(item["physical_npu_id"], item["host_pid"]) for item in owned] == [
        (0, host_pid)
    ]
    samples = [
        json.loads(line)
        for line in instance.hbm_path.read_text(encoding="utf-8").splitlines()
    ]
    assert samples[0]["host_pids"] == [host_pid]
    assert samples[0]["physical_npu_hbm_mb"] == {"0": 41842}
    peak = json.loads(instance.host_peak_path.read_text(encoding="utf-8"))
    assert peak["sample_count"] == 1
    assert peak["peak_hbm_mb"] == 41842


def test_runtime_sample_allows_all_owned_pids_to_leave_at_terminal_transition(
    tmp_path: Path,
) -> None:
    class FakeRoot:
        def run(self, _argv):
            output = NPU_SMI_26_PROCESS_FIXTURE
            output = output.replace(
                "| 0       0                 | 1732754       | python             | 150                   | NA                      |\n",
                "",
            )
            output = output.replace(
                "| 0       0                 | 1732755       | python             | 151                   | NA                      |\n",
                "",
            )
            output = output.replace(
                "| 0       0                 | 1732756       | python             | 152                   | NA                      |\n",
                "",
            )
            return orchestrator.CommandResult(output, "", 0)

    instance = object.__new__(orchestrator.StrictRepeatOrchestrator)
    instance.root = FakeRoot()
    instance.repeat_dir = tmp_path
    instance.devices = [0]
    instance.ownership = [{"host_pid": 1732754, "physical_npu_id": 0}]
    instance.all_owned_processes = [{"host_pid": 1732754, "physical_npu_id": 0}]

    assert instance._runtime_sample(2) is False


def test_runtime_sample_tracks_new_pid_from_the_owned_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeRoot:
        def run(self, _argv):
            return orchestrator.CommandResult(NPU_SMI_26_PROCESS_FIXTURE, "", 0)

    container_id = "d" * 64
    instance = object.__new__(orchestrator.StrictRepeatOrchestrator)
    instance.root = FakeRoot()
    instance.repeat_dir = tmp_path
    instance.devices = [0]
    instance.container_id = container_id
    instance.ownership = [{"host_pid": 1732754, "physical_npu_id": 0}]
    instance.all_owned_processes = [
        {
            "host_pid": 1732754,
            "physical_npu_id": 0,
            "container_id": container_id,
            "cgroup": f"0::/docker/{container_id}",
            "cmdline": "VLLM::Worker_TP0",
        }
    ]
    instance.all_owned_host_pids = [1732754]
    instance.host_pids = [1732754]
    instance.owned_processes_path = tmp_path / "runtime" / "owned-processes.json"
    instance.hbm_path = tmp_path / "hbm-samples.jsonl"
    instance.host_peak_path = tmp_path / "strict-host-peak-hbm.json"
    instance.hbm_sample_count = 0
    instance.per_device_peaks = {0: 0}
    monkeypatch.setattr(
        orchestrator,
        "read_host_process_context",
        lambda pid: (f"0::/docker/{container_id}", f"VLLMWorker {pid}"),
    )
    monkeypatch.setattr(orchestrator, "evidence_record", lambda *_args: {"sha256": "x"})

    assert instance._runtime_sample(2) is True
    owned = json.loads(instance.owned_processes_path.read_text(encoding="utf-8"))
    assert [item["host_pid"] for item in owned] == [1732754, 1732755, 1732756]


def test_runtime_sample_rejects_new_pid_outside_owned_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeRoot:
        def run(self, _argv):
            return orchestrator.CommandResult(NPU_SMI_26_PROCESS_FIXTURE, "", 0)

    container_id = "e" * 64
    instance = object.__new__(orchestrator.StrictRepeatOrchestrator)
    instance.root = FakeRoot()
    instance.repeat_dir = tmp_path
    instance.devices = [0]
    instance.container_id = container_id
    instance.ownership = [{"host_pid": 1732754, "physical_npu_id": 0}]
    instance.all_owned_processes = [
        {"host_pid": 1732754, "physical_npu_id": 0}
    ]
    monkeypatch.setattr(
        orchestrator,
        "read_host_process_context",
        lambda pid: ("0::/docker/external", f"python {pid}"),
    )

    with pytest.raises(orchestrator.GateFailure, match="outside the owned container"):
        instance._runtime_sample(2)


def test_terminal_owned_pid_absence_requires_prompt_command_exit() -> None:
    class ExitedProcess:
        returncode = 0

        def wait(self, *, timeout):
            assert timeout == 15.0
            return self.returncode

    instance = object.__new__(orchestrator.StrictRepeatOrchestrator)
    instance.args = SimpleNamespace(sample_interval_seconds=1.0)
    instance._confirm_terminal_owned_pid_absence(ExitedProcess())


def test_terminal_owned_pid_absence_rejects_a_still_active_command() -> None:
    class ActiveProcess:
        def wait(self, *, timeout):
            raise orchestrator.subprocess.TimeoutExpired("docker", timeout)

    instance = object.__new__(orchestrator.StrictRepeatOrchestrator)
    instance.args = SimpleNamespace(sample_interval_seconds=1.0)
    with pytest.raises(
        orchestrator.GateFailure, match="remained active for 15 seconds"
    ):
        instance._confirm_terminal_owned_pid_absence(ActiveProcess())


def test_owned_command_lifecycle_requires_an_identified_compute_pid() -> None:
    instance = object.__new__(orchestrator.StrictRepeatOrchestrator)
    instance.ownership = []
    with pytest.raises(
        orchestrator.GateFailure,
        match="lifecycle ended without an identified owned compute PID",
    ):
        instance._require_owned_process_observation()


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


def test_failure_cleanup_observation_follows_persisted_lease_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = orchestrator.ResourceLease(
        tmp_path / "leases",
        startup_id="failure",
        container_id="owned-container",
        devices=[0],
        port=18080,
        repeat_dir=tmp_path / "repeat-01",
    )
    lease.repeat_dir.mkdir()
    lease.acquire()
    cleanup = {"lease_released": False}
    instance = object.__new__(orchestrator.StrictRepeatOrchestrator)
    instance.lease = lease

    original_write = lease._write_global

    def assert_unlocked_before_persist(*, active: bool) -> None:
        if not active:
            assert lease.handles == []
            assert cleanup["lease_released"] is False
        original_write(active=active)

    monkeypatch.setattr(lease, "_write_global", assert_unlocked_before_persist)
    released_at = instance._release_lease_and_update_cleanup(cleanup)

    persisted = json.loads(
        (lease.repeat_dir / "resource-lease.json").read_text(encoding="utf-8")
    )
    assert persisted["active"] is False
    assert persisted["released_at"] == released_at
    assert cleanup["lease_released"] is True


def test_failure_summary_is_written_after_lease_cleanup() -> None:
    source = inspect.getsource(orchestrator.StrictRepeatOrchestrator.run)
    failure_branch = source.index("except BaseException as error:")
    release = source.index(
        "self._release_lease_and_update_cleanup(cleanup)", failure_branch
    )
    failure_summary = source.index('self.repeat_dir / "strict_execution_failure.json"')
    assert release < failure_summary


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


def test_host_snapshot_excludes_only_the_orchestrator_ssh_sudo_ancestry() -> None:
    process_output = (
        "  1 0 1 /init.scope root init\n"
        "  10 1 10 /system.slice/sshd.service root sshd: shuhao@notty\n"
        "  20 10 10 /user.slice/session-7.scope shuhao bash -lc target-a :18080\n"
        "  30 20 10 /user.slice/session-7.scope root sudo -n python target-a\n"
        "  40 30 10 /user.slice/session-7.scope root python target-a :18080\n"
        "  99 1 99 /system.slice/external.service other external target-a :18080\n"
    )
    ancestors = orchestrator.snapshot_process_ancestor_pids(process_output, 40)
    assert ancestors == {1, 10, 20, 30, 40}

    conflicts = orchestrator.benchmark_session_conflicts(
        tmux_output="",
        screen_output="",
        process_output=process_output,
        target_id="target-a",
        service_port=18080,
        excluded_pids=ancestors,
    )
    assert conflicts == [
        "  99 1 99 /system.slice/external.service other external target-a :18080"
    ]


@pytest.mark.parametrize(
    ("process_output", "message"),
    [
        ("1 0 init\n", "absent"),
        ("40 30 python\n", "incomplete"),
        ("40 30 python\n30 40 sudo\n", "cycle"),
    ],
)
def test_host_snapshot_ancestor_proof_fails_closed(
    process_output: str, message: str
) -> None:
    with pytest.raises(orchestrator.GateFailure, match=message):
        orchestrator.snapshot_process_ancestor_pids(process_output, 40)


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
        owned_runtime_privileged=False,
        privileged_authorization_source=None,
    )
    instance.devices = [7]
    instance.startup_id = "b" * 32
    instance.container_name = "vllm-hust-strict-repeat-" + instance.startup_id
    instance.repeat_dir = tmp_path / "repo" / "results" / "repeat-01"
    instance.container_repeat_dir = Path(
        "/workspace/vllm-hust-benchmark/results/repeat-01"
    )
    instance.runtime_transport = "registry"
    instance.runtime_config_digest = instance.args.runtime_image_digest
    instance.runtime_local_image_ref = instance.args.runtime_image_digest
    instance.runtime_storage_manifest_digest = None
    instance.expected_runtime_contract_sha256 = "e" * 64
    instance.host_container_identity_path = tmp_path / "leases" / "identity.json"
    return instance


def test_docker_create_is_owned_ephemeral_and_scoped(tmp_path: Path) -> None:
    instance = _orchestrator_stub(tmp_path)
    argv = instance._docker_create_argv()
    rendered = " ".join(argv)
    assert argv[:2] == ["docker", "create"]
    assert "--rm" in argv
    assert "--network none" in rendered
    assert "vllm-hust.strict-startup-id=" + "b" * 32 in argv
    assert "/dev/davinci7:/dev/davinci7" in argv
    assert "/dev/davinci7:/dev/davinci0" not in argv
    assert "ASCEND_VISIBLE_DEVICES=7" in argv
    assert "ASCEND_RT_VISIBLE_DEVICES=7" in argv
    assert "vllm-hust-shuhao-21rc" not in rendered
    assert "dst=/data/shared_models,readonly" in rendered
    assert "dst=/data/shared_datasets,readonly" in rendered
    assert "VLLM_HUST_STRICT_HOST_ORCHESTRATED=1" in argv
    assert "VLLM_HUST_STRICT_HOST_GATE_ATTESTED=1" in argv
    assert (
        "type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info,readonly"
    ) in argv
    assert not any("src=/usr/local/bin/npu-smi" in item for item in argv)
    assert any(
        value.startswith("VLLM_HUST_STRICT_HOST_PEAK_HBM_FILE=") for value in argv
    )
    assert "--pull=never" not in argv
    assert "--privileged" not in argv
    image_index = argv.index(instance.runtime_local_image_ref)
    entrypoint_index = argv.index("--entrypoint")
    assert argv[entrypoint_index + 1] == instance.args.command[0]
    assert argv[image_index + 1 :] == instance.args.command[1:]


def test_registry_runtime_prepares_owned_container_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = _orchestrator_stub(tmp_path)
    original_lstat = Path.lstat

    def root_safe_lstat(path: Path) -> os.stat_result:
        values = list(original_lstat(path))
        values[0] &= ~0o022
        values[4] = 0
        return os.stat_result(values)

    monkeypatch.setattr(Path, "lstat", root_safe_lstat)

    instance._write_runtime_preflight_contract()

    assert json.loads(instance.host_container_identity_path.read_text()) == {
        "startup_instance_id": instance.startup_id,
        "container_id": None,
    }


def test_docker_create_preserves_runner_argument_boundaries(tmp_path: Path) -> None:
    instance = _orchestrator_stub(tmp_path)
    instance.args.command = [
        "/usr/bin/env",
        "KEY=value with spaces",
        "bash",
        "-c",
        "literal;not-host-shell",
        "--",
        "$(not-expanded)",
    ]
    argv = instance._docker_create_argv()
    image_index = argv.index(instance.runtime_local_image_ref)
    assert argv[argv.index("--entrypoint") + 1] == "/usr/bin/env"
    assert argv[image_index + 1 :] == instance.args.command[1:]


def test_privileged_owned_runtime_is_explicit_and_preserves_argv(
    tmp_path: Path,
) -> None:
    instance = _orchestrator_stub(tmp_path)
    instance.args.owned_runtime_privileged = True
    instance.args.privileged_authorization_source = "user-explicit:thread-123"
    argv = instance._docker_create_argv()
    assert argv.count("--privileged") == 1
    image_index = argv.index(instance.runtime_local_image_ref)
    assert argv[image_index + 1 :] == instance.args.command[1:]


def test_docker_create_keeps_physical_nodes_and_separate_logical_rank_order(
    tmp_path: Path,
) -> None:
    instance = _orchestrator_stub(tmp_path)
    instance.devices = [7, 6]
    argv = instance._docker_create_argv()
    assert "/dev/davinci7:/dev/davinci7" in argv
    assert "/dev/davinci6:/dev/davinci6" in argv
    assert not any("/dev/davinci7:/dev/davinci0" == item for item in argv)
    assert "ASCEND_VISIBLE_DEVICES=7,6" in argv
    assert "ASCEND_RT_VISIBLE_DEVICES=7,6" in argv


def _docker29_runtime() -> dict[str, object]:
    return {
        "runtime_transport": "docker-archive",
        "runtime_image_digest": "sha256:" + "9" * 64,
        "runtime_config_digest": "sha256:" + "9" * 64,
        "containerd_storage_manifest_digest": "sha256:" + "5" * 64,
        "runtime_archive_sha256": "sha256:" + "d" * 64,
    }


def test_docker29_contract_maps_storage_to_config_and_create_ref(
    tmp_path: Path,
) -> None:
    runtime = _docker29_runtime()
    contract = orchestrator.resolve_runtime_storage_contract(
        runtime, runtime["runtime_image_digest"]
    )
    assert contract["local_image_ref"] == runtime["containerd_storage_manifest_digest"]
    assert contract["config_digest"] == runtime["runtime_config_digest"]
    instance = _orchestrator_stub(tmp_path)
    instance.runtime_transport = "docker-archive"
    instance.runtime_local_image_ref = runtime["containerd_storage_manifest_digest"]
    argv = instance._docker_create_argv()
    assert runtime["containerd_storage_manifest_digest"] in argv
    assert runtime["runtime_config_digest"] not in argv
    assert "--pull=never" in argv
    image_index = argv.index(runtime["containerd_storage_manifest_digest"])
    assert argv[argv.index("--entrypoint") + 1] == (
        "/usr/local/python3.11.14/bin/python"
    )
    assert argv[image_index + 1] == (
        "/workspace/vllm-hust-benchmark/scripts/verify-owned-runtime-and-exec.py"
    )
    assert argv[image_index + 1 :][-len(instance.args.command) :] == (
        instance.args.command
    )


def test_docker29_contract_rejects_wrong_config_and_masquerading_storage() -> None:
    runtime = _docker29_runtime()
    runtime["runtime_config_digest"] = "sha256:" + "8" * 64
    with pytest.raises(orchestrator.GateFailure, match="config digest differs"):
        orchestrator.resolve_runtime_storage_contract(
            runtime, runtime["runtime_image_digest"]
        )
    runtime = _docker29_runtime()
    runtime["containerd_storage_manifest_digest"] = runtime["runtime_config_digest"]
    with pytest.raises(orchestrator.GateFailure, match="masquerade"):
        orchestrator.resolve_runtime_storage_contract(
            runtime, runtime["runtime_image_digest"]
        )


class _Docker29Root:
    def __init__(self, manifest: bytes, config: bytes, storage: str) -> None:
        self.manifest = manifest
        self.config = config
        self.storage = storage
        self.calls: list[list[str]] = []

    def run_bytes(self, argv, *, check=True):
        self.calls.append(list(argv))
        digest = argv[-1]
        if argv[-2] == "get":
            payload = self.manifest if digest == self.storage else self.config
        else:
            manifest = json.loads(self.manifest)
            payload = (
                "\n".join(layer["digest"] for layer in manifest["layers"]) + "\n"
            ).encode()
        return orchestrator.CommandBytesResult(payload, b"", 0)

    def run(self, argv, *, check=True):
        return orchestrator.CommandResult(
            json.dumps([{"Id": self.storage}]) + "\n", "", 0
        )


def test_docker29_content_attestation_hashes_manifest_config_layers_and_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = b'{"architecture":"arm64"}'
    config_digest = "sha256:" + orchestrator.hashlib.sha256(config).hexdigest()
    layer_digest = "sha256:" + "1" * 64
    manifest = json.dumps(
        {
            "config": {"digest": config_digest},
            "layers": [{"digest": layer_digest, "size": 123}],
        },
        separators=(",", ":"),
    ).encode()
    storage_digest = "sha256:" + orchestrator.hashlib.sha256(manifest).hexdigest()
    archive = tmp_path / "runtime.tar.zst"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(orchestrator, "STRICT_V018_ARCHIVE", archive)
    instance = _orchestrator_stub(tmp_path)
    instance.repeat_dir.mkdir(parents=True)
    instance.runtime_transport = "docker-archive"
    instance.runtime_config_digest = config_digest
    instance.runtime_storage_manifest_digest = storage_digest
    instance.runtime_local_image_ref = storage_digest
    instance.runtime_archive_sha256 = (
        "sha256:" + orchestrator.hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    instance.root = _Docker29Root(manifest, config, storage_digest)
    identity = instance._attest_runtime_storage()
    assert identity["manifest_config_digest"] == config_digest
    assert identity["containerd_storage_manifest_digest"] == storage_digest
    assert identity["layers"][0]["digest"] == layer_digest
    assert sum(call[-2:] == ["ls", "-q"] for call in instance.root.calls) == 1
    assert not any("info" in call for call in instance.root.calls)


@pytest.mark.parametrize(
    "output",
    (
        b"",
        b"sha256:not-a-digest\n",
        ("sha256:" + "1" * 64 + " extra\n").encode(),
        ("sha256:" + "1" * 64 + "\n" + "sha256:" + "1" * 64 + "\n").encode(),
        b"\xff\n",
    ),
)
def test_ctr_content_list_rejects_malformed_output(output: bytes) -> None:
    with pytest.raises(orchestrator.GateFailure, match="content list"):
        orchestrator.parse_ctr_content_digests(output)


def test_ctr_content_list_uses_exact_digest_not_prefix() -> None:
    required = "sha256:" + "1" * 64
    prefixed = required + "0"
    with pytest.raises(orchestrator.GateFailure, match="malformed"):
        orchestrator.parse_ctr_content_digests((prefixed + "\n").encode())


class _MissingLayerRoot(_Docker29Root):
    def run_bytes(self, argv, *, check=True):
        if argv[-2:] == ["ls", "-q"]:
            return orchestrator.CommandBytesResult(
                ("sha256:" + "2" * 64 + "\n").encode(), b"", 0
            )
        return super().run_bytes(argv, check=check)


class _FailedContentListRoot(_Docker29Root):
    def run_bytes(self, argv, *, check=True):
        if argv[-2:] == ["ls", "-q"]:
            raise orchestrator.GateFailure("host command failed: ctr content ls")
        return super().run_bytes(argv, check=check)


@pytest.mark.parametrize("root_type", (_MissingLayerRoot, _FailedContentListRoot))
def test_docker29_rejects_missing_or_failed_content_list(
    tmp_path: Path, root_type: type[_Docker29Root]
) -> None:
    config = b"config"
    config_digest = "sha256:" + orchestrator.hashlib.sha256(config).hexdigest()
    manifest = json.dumps(
        {
            "config": {"digest": config_digest},
            "layers": [{"digest": "sha256:" + "1" * 64, "size": 1}],
        },
        separators=(",", ":"),
    ).encode()
    storage = "sha256:" + orchestrator.hashlib.sha256(manifest).hexdigest()
    instance = _orchestrator_stub(tmp_path)
    instance.repeat_dir.mkdir(parents=True)
    instance.runtime_transport = "docker-archive"
    instance.runtime_config_digest = config_digest
    instance.runtime_storage_manifest_digest = storage
    instance.root = root_type(manifest, config, storage)
    with pytest.raises(
        orchestrator.GateFailure, match="content list|content ls|omitted layer"
    ):
        instance._attest_runtime_storage()


def test_docker29_content_attestation_rejects_manifest_hash_mismatch(
    tmp_path: Path,
) -> None:
    instance = _orchestrator_stub(tmp_path)
    instance.repeat_dir.mkdir(parents=True)
    instance.runtime_transport = "docker-archive"
    instance.runtime_storage_manifest_digest = "sha256:" + "5" * 64
    instance.root = _Docker29Root(
        b"{}", b"{}", instance.runtime_storage_manifest_digest
    )
    with pytest.raises(orchestrator.GateFailure, match="manifest hash mismatch"):
        instance._attest_runtime_storage()


def test_docker29_content_attestation_rejects_config_blob_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_config = b"expected-config"
    config_digest = "sha256:" + orchestrator.hashlib.sha256(expected_config).hexdigest()
    manifest = json.dumps(
        {
            "config": {"digest": config_digest},
            "layers": [{"digest": "sha256:" + "1" * 64, "size": 1}],
        },
        separators=(",", ":"),
    ).encode()
    storage = "sha256:" + orchestrator.hashlib.sha256(manifest).hexdigest()
    archive = tmp_path / "runtime.tar.zst"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(orchestrator, "STRICT_V018_ARCHIVE", archive)
    instance = _orchestrator_stub(tmp_path)
    instance.repeat_dir.mkdir(parents=True)
    instance.runtime_transport = "docker-archive"
    instance.runtime_config_digest = config_digest
    instance.runtime_storage_manifest_digest = storage
    instance.runtime_archive_sha256 = (
        "sha256:" + orchestrator.hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    instance.root = _Docker29Root(manifest, b"wrong-config", storage)
    with pytest.raises(orchestrator.GateFailure, match="config blob hash mismatch"):
        instance._attest_runtime_storage()


def test_docker29_content_attestation_requires_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = b"config"
    config_digest = "sha256:" + orchestrator.hashlib.sha256(config).hexdigest()
    manifest = json.dumps(
        {
            "config": {"digest": config_digest},
            "layers": [{"digest": "sha256:" + "1" * 64, "size": 1}],
        },
        separators=(",", ":"),
    ).encode()
    storage = "sha256:" + orchestrator.hashlib.sha256(manifest).hexdigest()
    monkeypatch.setattr(orchestrator, "STRICT_V018_ARCHIVE", tmp_path / "missing")
    instance = _orchestrator_stub(tmp_path)
    instance.repeat_dir.mkdir(parents=True)
    instance.runtime_transport = "docker-archive"
    instance.runtime_config_digest = config_digest
    instance.runtime_storage_manifest_digest = storage
    instance.runtime_archive_sha256 = "sha256:" + "d" * 64
    instance.root = _Docker29Root(manifest, config, storage)
    with pytest.raises(orchestrator.GateFailure, match="archive is missing"):
        instance._attest_runtime_storage()


def test_owned_container_inspect_rejects_extra_npu_mapping(tmp_path: Path) -> None:
    instance = _orchestrator_stub(tmp_path)
    instance.container_id = "c" * 64
    devices = [
        {"PathOnHost": "/dev/davinci7", "PathInContainer": "/dev/davinci7"},
        *[
            {"PathOnHost": f"/dev/{name}", "PathInContainer": f"/dev/{name}"}
            for name in ("davinci_manager", "devmm_svm", "hisi_hdc")
        ],
    ]
    record = {
        "Id": instance.container_id,
        "Image": instance.args.runtime_image_digest,
        "Path": instance.args.command[0],
        "Args": instance.args.command[1:],
        "Config": {
            "Labels": {"vllm-hust.strict-startup-id": instance.startup_id},
            "Entrypoint": [instance.args.command[0]],
            "Cmd": instance.args.command[1:],
            "Env": ["ASCEND_VISIBLE_DEVICES=7", "ASCEND_RT_VISIBLE_DEVICES=7"],
        },
        "HostConfig": {
            "AutoRemove": True,
            "NetworkMode": "none",
            "IpcMode": "host",
            "Privileged": False,
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
            *[
                {"Source": path, "Destination": path, "RW": False}
                for path in ("/etc/ascend_install.info",)
            ],
        ],
    }
    instance._validate_owned_container_inspect(record)
    record["Config"]["Env"][0] = "ASCEND_VISIBLE_DEVICES=0"
    with pytest.raises(orchestrator.GateFailure, match="visible NPU environment"):
        instance._validate_owned_container_inspect(record)
    record["Config"]["Env"][0] = "ASCEND_VISIBLE_DEVICES=7"
    record["HostConfig"]["Privileged"] = True
    with pytest.raises(orchestrator.GateFailure, match="privilege contract"):
        instance._validate_owned_container_inspect(record)
    record["HostConfig"]["Privileged"] = False
    record["Mounts"][-1]["RW"] = True
    with pytest.raises(orchestrator.GateFailure, match="Ascend mount"):
        instance._validate_owned_container_inspect(record)
    record["Mounts"][-1]["RW"] = False
    record["Mounts"].append(dict(record["Mounts"][-1]))
    with pytest.raises(orchestrator.GateFailure, match="Ascend mount"):
        instance._validate_owned_container_inspect(record)
    record["Mounts"].pop()
    record["Config"]["Entrypoint"] = ["/image/default-entrypoint"]
    with pytest.raises(orchestrator.GateFailure, match="process argv"):
        instance._validate_owned_container_inspect(record)
    record["Config"]["Entrypoint"] = [instance.args.command[0]]
    record["Path"] = "/image/default-entrypoint"
    with pytest.raises(orchestrator.GateFailure, match="process argv"):
        instance._validate_owned_container_inspect(record)
    record["Path"] = instance.args.command[0]
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


class _CleanupRaceRoot:
    prefix: list[str] = []

    def __init__(
        self,
        inspect_results: list[orchestrator.CommandResult],
        removal: orchestrator.CommandResult,
    ) -> None:
        self.inspect_results = iter(inspect_results)
        self.removal = removal
        self.calls: list[list[str]] = []

    def run(self, argv, *, check=True):
        self.calls.append(list(argv))
        if argv[:2] == ["docker", "inspect"]:
            return next(self.inspect_results)
        if argv[:2] == ["docker", "rm"]:
            return self.removal
        return orchestrator.CommandResult("", "", 0)


def _stopped_owned_container(instance) -> orchestrator.CommandResult:
    return orchestrator.CommandResult(
        json.dumps(
            [
                {
                    "Id": instance.container_id,
                    "Config": {
                        "Labels": {"vllm-hust.strict-startup-id": instance.startup_id}
                    },
                    "State": {"Running": False},
                }
            ]
        ),
        "",
        0,
    )


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


def test_cleanup_accepts_auto_remove_race_after_confirming_absence(
    tmp_path: Path,
) -> None:
    instance = _orchestrator_stub(tmp_path)
    instance.container_id = "c" * 64
    present = _stopped_owned_container(instance)
    absent = orchestrator.CommandResult("", "No such container", 1)
    instance.root = _CleanupRaceRoot(
        [present, present, absent],
        orchestrator.CommandResult("", "No such container", 1),
    )

    instance._cleanup_owned_container(allow_stop=True)

    assert sum(call[:2] == ["docker", "inspect"] for call in instance.root.calls) == 3
    assert sum(call[:2] == ["docker", "rm"] for call in instance.root.calls) == 1


def test_cleanup_rejects_failed_remove_when_owned_container_still_exists(
    tmp_path: Path,
) -> None:
    instance = _orchestrator_stub(tmp_path)
    instance.container_id = "c" * 64
    present = _stopped_owned_container(instance)
    instance.root = _CleanupRaceRoot(
        [present, present, present],
        orchestrator.CommandResult("", "remove failed", 1),
    )

    with pytest.raises(
        orchestrator.GateFailure,
        match="removal failed and container still exists",
    ):
        instance._cleanup_owned_container(allow_stop=True)


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


def _minimal_cli_argv(tmp_path: Path) -> list[str]:
    return [
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
        "--",
        "true",
    ]


def test_cli_defaults_to_nonprivileged_owned_runtime(tmp_path: Path) -> None:
    args = orchestrator.parse_args(_minimal_cli_argv(tmp_path))
    assert args.owned_runtime_privileged is False
    assert args.privileged_authorization_source is None


def test_cli_requires_authorization_for_privileged_owned_runtime(
    tmp_path: Path,
) -> None:
    argv = _minimal_cli_argv(tmp_path)
    argv[0:0] = ["--owned-runtime-privileged"]
    with pytest.raises(SystemExit):
        orchestrator.parse_args(argv)


def test_cli_rejects_privileged_authorization_without_opt_in(tmp_path: Path) -> None:
    argv = _minimal_cli_argv(tmp_path)
    argv[0:0] = [
        "--privileged-authorization-source",
        "user-explicit:thread-019fc873:2026-08-05",
    ]
    with pytest.raises(SystemExit):
        orchestrator.parse_args(argv)


def test_cli_accepts_explicit_privileged_authorization(tmp_path: Path) -> None:
    argv = _minimal_cli_argv(tmp_path)
    argv[0:0] = [
        "--owned-runtime-privileged",
        "--privileged-authorization-source",
        "user-explicit:thread-019fc873:2026-08-05",
    ]
    args = orchestrator.parse_args(argv)
    assert args.owned_runtime_privileged is True
    assert args.privileged_authorization_source == (
        "user-explicit:thread-019fc873:2026-08-05"
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


def test_trusted_mount_source_accepts_safe_file_and_directory(tmp_path: Path) -> None:
    directory = tmp_path / "etc"
    directory.mkdir()
    source = directory / "ascend_install.info"
    source.write_text("Install_Path_Param=/usr/local/Ascend\n", encoding="utf-8")
    tmp_path.chmod(0o755)
    directory.chmod(0o755)
    source.chmod(0o444)
    uid = source.stat().st_uid
    orchestrator.validate_trusted_mount_source(
        directory, "directory", trusted_uid=uid, ancestor_floor=tmp_path
    )
    orchestrator.validate_trusted_mount_source(
        source, "file", trusted_uid=uid, ancestor_floor=tmp_path
    )


@pytest.mark.parametrize("failure", ("missing", "writable", "symlink", "wrong-type"))
def test_trusted_mount_source_fails_closed(tmp_path: Path, failure: str) -> None:
    directory = tmp_path / "etc"
    directory.mkdir()
    source = directory / "ascend_install.info"
    source.write_text("Install_Path_Param=/usr/local/Ascend\n", encoding="utf-8")
    tmp_path.chmod(0o755)
    directory.chmod(0o755)
    source.chmod(0o444)
    uid = source.stat().st_uid
    if failure == "missing":
        source.unlink()
    elif failure == "writable":
        directory.chmod(0o777)
    elif failure == "symlink":
        source.unlink()
        source.symlink_to(directory)
    else:
        source.unlink()
        source.mkdir()
    with pytest.raises(orchestrator.GateFailure):
        orchestrator.validate_trusted_mount_source(
            source, "file", trusted_uid=uid, ancestor_floor=tmp_path
        )


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

    captured_inputs = [{"prompt_token_ids": [1, 2, 3], "output_len": 4}]
    immutable = write_json(
        "immutable-input-attestation.json",
        {
            "schema_version": "immutable-input-attestation/v1",
            "model_id": "Qwen/model",
            "model_revision": model_revision,
            "data_identity": data_identity,
            "resolved_input_kind": "throughput-sample-requests",
            "resolved_input_sha256": resolved_input_sha256(
                input_kind="throughput-sample-requests", inputs=captured_inputs
            ),
            "resolved_inputs": captured_inputs,
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
    container_inspect = write_json(
        "owned-container-create-inspect.json",
        [
            {
                "Id": container_id,
                "Image": runtime_digest,
                "Config": {
                    "Env": ["ASCEND_VISIBLE_DEVICES=0", "ASCEND_RT_VISIBLE_DEVICES=0"]
                },
                "HostConfig": {
                    "Privileged": False,
                    "Devices": [
                        {
                            "PathOnHost": "/dev/davinci0",
                            "PathInContainer": "/dev/davinci0",
                        },
                        *[
                            {
                                "PathOnHost": f"/dev/{name}",
                                "PathInContainer": f"/dev/{name}",
                            }
                            for name in ("davinci_manager", "devmm_svm", "hisi_hdc")
                        ],
                    ],
                },
                "Mounts": [
                    {"Source": path, "Destination": path, "RW": False}
                    for path in (
                        "/usr/local/Ascend/driver",
                        "/etc/ascend_install.info",
                    )
                ],
            }
        ],
    )
    create_argv = write_json(
        "runtime/docker-create-argv.json",
        [
            "docker",
            "create",
            *[
                item
                for path in (
                    "/usr/local/Ascend/driver",
                    "/etc/ascend_install.info",
                )
                for item in (
                    "--mount",
                    f"type=bind,src={path},dst={path},readonly",
                )
            ],
            "--env",
            "ASCEND_VISIBLE_DEVICES=0",
            "--env",
            "ASCEND_RT_VISIBLE_DEVICES=0",
            *[
                item
                for value in (
                    "/dev/davinci0:/dev/davinci0",
                    "/dev/davinci_manager:/dev/davinci_manager",
                    "/dev/devmm_svm:/dev/devmm_svm",
                    "/dev/hisi_hdc:/dev/hisi_hdc",
                )
                for item in ("--device", value)
            ],
            runtime_digest,
            "runner",
        ],
    )
    runtime_identity = write_json(
        "owned-container-identity.json",
        {
            "container_id": container_id,
            "runtime_image_digest": runtime_digest,
            "runtime_storage_identity": {
                "transport": "registry",
                "runtime_config_digest": runtime_digest,
                "local_create_ref": runtime_digest,
            },
            "inspect": orchestrator.evidence_record(repeat, container_inspect),
            "create_argv": orchestrator.evidence_record(repeat, create_argv),
            "runner_argv": ["runner"],
            "device_node_mapping": {
                "0": {
                    "host": "/dev/davinci0",
                    "container": "/dev/davinci0",
                }
            },
            "physical_to_logical_rank": {"0": 0},
            "owned_runtime_security": {
                "schema_version": "owned-runtime-security/v1",
                "privileged": False,
                "authorization_source": None,
            },
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
        runtime_storage_identity=orchestrator.evidence_record(repeat, runtime_identity),
        owned_runtime_security={
            "schema_version": "owned-runtime-security/v1",
            "privileged": False,
            "authorization_source": None,
        },
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
