"""Tests for the poy-180 watchdog ownership contract (issue #127).

Covers:
- Runner name → physical device resolution.
- ``docker create`` command carries labels, device mapping, env, and rejects
  forbidden volumes / reserved env overrides.
- ``validate_container_inspect`` accepts a matching contract and rejects
  wrong labels, wrong device mapping, forbidden bind mounts, privileged
  mode, dangerous capabilities, and shared/slave propagation.
"""

from __future__ import annotations

import pytest

from vllm_hust_benchmark.runner_ownership import (
    LOGICAL_DEVICE_LABEL,
    PHYSICAL_DEVICE_LABEL,
    RUNNER_LABEL,
    build_docker_create_command,
    resolve_runner_device,
    validate_container_inspect,
)


def _inspect_payload(runner_name: str, physical_device: int) -> dict:
    """Return a minimal valid docker inspect payload for the given assignment."""
    return {
        "Config": {
            "Labels": {
                RUNNER_LABEL: runner_name,
                PHYSICAL_DEVICE_LABEL: str(physical_device),
                LOGICAL_DEVICE_LABEL: "0",
            },
            "Env": ["ASCEND_RT_VISIBLE_DEVICES=0", "ASCEND_VISIBLE_DEVICES=0"],
        },
        "HostConfig": {
            "Devices": [
                {
                    "PathOnHost": f"/dev/davinci{physical_device}",
                    "PathInContainer": "/dev/davinci0",
                }
            ]
        },
    }


# ---------------------------------------------------------------------------
# resolve_runner_device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("physical_device", range(4))
def test_resolve_runner_device(physical_device: int) -> None:
    assignment = resolve_runner_device(f"poy-180-21rc-npu{physical_device}")
    assert assignment.physical_device == physical_device
    assert assignment.logical_device == 0


@pytest.mark.parametrize("runner_name", ["", "npu4", "poy-180", "npu-2"])
def test_resolve_runner_device_rejects_unknown_runner(runner_name: str) -> None:
    with pytest.raises(ValueError, match="runner name must end"):
        resolve_runner_device(runner_name)


# ---------------------------------------------------------------------------
# build_docker_create_command
# ---------------------------------------------------------------------------


def test_docker_command_carries_watchdog_contract() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu2")
    command = build_docker_create_command(
        assignment=assignment,
        container_name="benchmark-job",
        image="example/ascend:latest",
        command=["python", "run.py"],
        volumes=["/workspace:/workspace:ro"],
        extra_env=["BENCHMARK_SCENARIO=sharegpt-online"],  # pragma: allowlist secret
    )
    joined = " ".join(command)
    assert "org.vllm-hust.runner=poy-180-21rc-npu2" in joined
    assert "org.vllm-hust.npu-physical=2" in joined
    assert "/dev/davinci2:/dev/davinci0" in joined
    assert "ASCEND_RT_VISIBLE_DEVICES=0" in joined
    assert command[-3:] == ["example/ascend:latest", "python", "run.py"]


def test_docker_command_includes_control_devices() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    command = build_docker_create_command(
        assignment=assignment,
        container_name="benchmark-job",
        image="example/ascend:latest",
        command=[],
    )
    joined = " ".join(command)
    assert "/dev/davinci_manager:/dev/davinci_manager" in joined
    assert "/dev/devmm_svm:/dev/devmm_svm" in joined
    assert "/dev/hisi_hdc:/dev/hisi_hdc" in joined


def test_docker_command_rejects_logical_device_override() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="managed by the launcher"):
        build_docker_create_command(
            assignment=assignment,
            container_name="benchmark-job",
            image="example/ascend:latest",
            command=[],
            extra_env=["ASCEND_RT_VISIBLE_DEVICES=3"],
        )


def test_docker_command_rejects_empty_name() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="container name is required"):
        build_docker_create_command(
            assignment=assignment,
            container_name="  ",
            image="example/ascend:latest",
            command=[],
        )


def test_docker_command_rejects_empty_image() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="container image is required"):
        build_docker_create_command(
            assignment=assignment,
            container_name="benchmark-job",
            image="",
            command=[],
        )


# ---------------------------------------------------------------------------
# validate_container_inspect — acceptance
# ---------------------------------------------------------------------------


def test_validate_container_inspect_accepts_matching_contract() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu3")
    validate_container_inspect(_inspect_payload(assignment.runner_name, 3), assignment)


def test_validate_container_inspect_rejects_wrong_device() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu3")
    payload = _inspect_payload(assignment.runner_name, 3)
    payload["HostConfig"]["Devices"][0]["PathOnHost"] = "/dev/davinci2"
    with pytest.raises(ValueError, match="mapping is missing or incorrect"):
        validate_container_inspect(payload, assignment)


def test_validate_container_inspect_rejects_wrong_runner_label() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu3")
    payload = _inspect_payload(assignment.runner_name, 3)
    payload["Config"]["Labels"][RUNNER_LABEL] = "poy-180-21rc-npu0"
    with pytest.raises(ValueError, match="container label"):
        validate_container_inspect(payload, assignment)


def test_validate_container_inspect_rejects_missing_env() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    payload["Config"]["Env"] = ["ASCEND_RT_VISIBLE_DEVICES=0"]
    with pytest.raises(ValueError, match="logical-device environment"):
        validate_container_inspect(payload, assignment)


# ---------------------------------------------------------------------------
# mount constraint tests (issue #127)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "volume_spec",
    [
        "/dev:/dev",  # exposes all device nodes
        "/dev/davinci1:/dev/davinci1",  # bypasses single-card isolation
        "/:/host",  # exposes host root filesystem
        "/sys:/sys",  # exposes NPU driver sysfs
        "/var/run/docker.sock:/var/run/docker.sock",  # container escape
        "/usr/local/Ascend:/usr/local/Ascend",  # exposes CANN stack
    ],
)
def test_docker_command_rejects_bypass_volume(volume_spec: str) -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="forbidden host path"):
        build_docker_create_command(
            assignment=assignment,
            container_name="benchmark-job",
            image="example/ascend:latest",
            command=[],
            volumes=[volume_spec],
        )


@pytest.mark.parametrize(
    "volume_spec",
    [
        "/host/data:/etc",  # non-allowlisted destination
        "/host/data:/usr/bin",  # non-allowlisted destination
    ],
)
def test_docker_command_rejects_non_allowlisted_destination(volume_spec: str) -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="forbidden container destination"):
        build_docker_create_command(
            assignment=assignment,
            container_name="benchmark-job",
            image="example/ascend:latest",
            command=[],
            volumes=[volume_spec],
        )


def test_docker_command_rejects_invalid_volume_format() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="invalid volume spec"):
        build_docker_create_command(
            assignment=assignment,
            container_name="benchmark-job",
            image="example/ascend:latest",
            command=[],
            volumes=["a:b:c:d"],
        )


def test_docker_command_accepts_workspace_volume() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    command = build_docker_create_command(
        assignment=assignment,
        container_name="benchmark-job",
        image="example/ascend:latest",
        command=[],
        volumes=["/host/ws:/workspace", "/host/cache:/root/.cache"],
    )
    joined = " ".join(command)
    assert "/host/ws:/workspace" in joined
    assert "/host/cache:/root/.cache" in joined


def test_docker_command_accepts_named_and_anonymous_volumes() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    command = build_docker_create_command(
        assignment=assignment,
        container_name="benchmark-job",
        image="example/ascend:latest",
        command=[],
        volumes=["my_vol:/workspace", "/tmp"],
    )
    joined = " ".join(command)
    assert "my_vol:/workspace" in joined
    assert "/tmp" in joined


def test_docker_command_accepts_readonly_option() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    command = build_docker_create_command(
        assignment=assignment,
        container_name="benchmark-job",
        image="example/ascend:latest",
        command=[],
        volumes=["/host/ws:/workspace:ro"],
    )
    assert "/host/ws:/workspace:ro" in " ".join(command)


def test_docker_command_rejects_shared_propagation() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="forbidden mount option"):
        build_docker_create_command(
            assignment=assignment,
            container_name="benchmark-job",
            image="example/ascend:latest",
            command=[],
            volumes=["/host/ws:/workspace:shared"],
        )


# ---------------------------------------------------------------------------
# validate_container_inspect — mount / capability rejection
# ---------------------------------------------------------------------------


def test_inspect_rejects_dev_bind_mount() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    payload["Mounts"] = [
        {
            "Type": "bind",
            "Source": "/dev",
            "Destination": "/dev",
            "Mode": "rw",
            "RW": True,
            "Propagation": "rprivate",
        }
    ]
    with pytest.raises(ValueError, match="forbidden bind mount source"):
        validate_container_inspect(payload, assignment)


def test_inspect_rejects_root_bind_mount() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    payload["Mounts"] = [
        {
            "Type": "bind",
            "Source": "/",
            "Destination": "/host",
            "Mode": "rw",
            "RW": True,
            "Propagation": "rprivate",
        }
    ]
    with pytest.raises(ValueError, match="forbidden"):
        validate_container_inspect(payload, assignment)


def test_inspect_rejects_privileged_mode() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    payload["HostConfig"]["Privileged"] = True
    with pytest.raises(ValueError, match="privileged"):
        validate_container_inspect(payload, assignment)


@pytest.mark.parametrize("cap", ["SYS_ADMIN", "SYS_PTRACE", "SYS_MODULE"])
def test_inspect_rejects_dangerous_capabilities(cap: str) -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    payload["HostConfig"]["CapAdd"] = [cap]
    with pytest.raises(ValueError, match="dangerous capabilities"):
        validate_container_inspect(payload, assignment)


def test_inspect_rejects_legacy_binds() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    payload["HostConfig"]["Binds"] = ["/dev:/dev"]
    with pytest.raises(ValueError, match="forbidden host path"):
        validate_container_inspect(payload, assignment)


def test_inspect_rejects_shared_propagation() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    payload["Mounts"] = [
        {
            "Type": "bind",
            "Source": "/host/data",
            "Destination": "/workspace",
            "Mode": "rw",
            "RW": True,
            "Propagation": "shared",
        }
    ]
    with pytest.raises(ValueError, match="propagation"):
        validate_container_inspect(payload, assignment)


def test_inspect_accepts_workspace_bind_mount() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    payload["Mounts"] = [
        {
            "Type": "bind",
            "Source": "/home/user/workspace",
            "Destination": "/workspace",
            "Mode": "rw",
            "RW": True,
            "Propagation": "rprivate",
        }
    ]
    validate_container_inspect(payload, assignment)


def test_inspect_accepts_tmpfs_mount() -> None:
    """tmpfs mounts have no host source and should be accepted."""
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    payload["Mounts"] = [
        {
            "Type": "tmpfs",
            "Destination": "/tmp",
        }
    ]
    validate_container_inspect(payload, assignment)
