"""Tests for the poy-180 watchdog ownership contract (issue #127).

Covers:
- Runner name → physical device resolution.
- ``docker create`` command carries labels, device mapping, env, and rejects
  forbidden volumes / reserved env overrides / argument injection.
- ``validate_container_inspect`` accepts a matching contract and rejects
  wrong labels, wrong/extra device mapping, forbidden bind mounts, named
  volumes, privileged mode, dangerous capabilities, and shared/slave
  propagation.

PR #150 review fixes:
- Argument injection: image/name/command starting with ``-`` is rejected; a
  ``--`` separator is emitted before the image.
- Host path blacklist covers sub-paths (``/proc/1/root``, ``/sys/kernel``).
- Named volumes are rejected (local driver can bind ``/`` to a named volume).
- Device mapping is an exact set: extra ``davinci*`` devices are rejected.
"""

from __future__ import annotations

import pytest

from vllm_hust_benchmark.runner_ownership import (
    CONTROL_DEVICES,
    LOGICAL_DEVICE_LABEL,
    PHYSICAL_DEVICE_LABEL,
    RUNNER_LABEL,
    build_docker_create_command,
    resolve_runner_device,
    validate_container_inspect,
)


def _inspect_payload(runner_name: str, physical_device: int) -> dict:
    """Return a valid docker inspect payload with the exact expected devices."""
    devices = [
        {
            "PathOnHost": f"/dev/davinci{physical_device}",
            "PathInContainer": "/dev/davinci0",
        }
    ]
    for device in CONTROL_DEVICES:
        devices.append({"PathOnHost": device, "PathInContainer": device})
    return {
        "Config": {
            "Labels": {
                RUNNER_LABEL: runner_name,
                PHYSICAL_DEVICE_LABEL: str(physical_device),
                LOGICAL_DEVICE_LABEL: "0",
            },
            "Env": ["ASCEND_RT_VISIBLE_DEVICES=0", "ASCEND_VISIBLE_DEVICES=0"],
        },
        "HostConfig": {"Devices": devices},
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
    # ``--`` separator must precede image so Docker treats it as positional.
    sep_index = command.index("--")
    assert command[sep_index + 1] == "example/ascend:latest"
    assert command[sep_index + 2 :] == ["python", "run.py"]


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


# --- argument injection guards (PR #150 review) ---


@pytest.mark.parametrize(
    "image",
    [
        "--device=/dev/davinci1:/dev/davinci1",
        "-v",
        "--privileged",
    ],
)
def test_docker_command_rejects_option_like_image(image: str) -> None:
    """A caller must not inject docker options via the image field."""
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="must not start with '-'"):
        build_docker_create_command(
            assignment=assignment,
            container_name="benchmark-job",
            image=image,
            command=["ubuntu"],
        )


def test_docker_command_rejects_option_like_name() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="must not start with '-'"):
        build_docker_create_command(
            assignment=assignment,
            container_name="--label=evil",
            image="example/ascend:latest",
            command=[],
        )


@pytest.mark.parametrize("cmd_element", ["--device=/dev/davinci1:/dev/davinci1", "-v"])
def test_docker_command_rejects_option_like_command_element(cmd_element: str) -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="must not start with '-'"):
        build_docker_create_command(
            assignment=assignment,
            container_name="benchmark-job",
            image="example/ascend:latest",
            command=[cmd_element, "ubuntu"],
        )


def test_docker_command_emits_separator_before_image() -> None:
    """``--`` must appear before the image token."""
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    command = build_docker_create_command(
        assignment=assignment,
        container_name="benchmark-job",
        image="example/ascend:latest",
        command=["python", "run.py"],
    )
    assert "--" in command
    sep_index = command.index("--")
    # Image and command must all be after the separator.
    assert command[sep_index + 1] == "example/ascend:latest"
    assert command[sep_index + 2 :] == ["python", "run.py"]


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
    with pytest.raises(ValueError, match="extra device mappings"):
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


# --- extra device rejection (PR #150 review) ---


def test_inspect_rejects_extra_davinci_device() -> None:
    """A container mapping an extra NPU card must be rejected (exact set)."""
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    payload["HostConfig"]["Devices"].append(
        {"PathOnHost": "/dev/davinci1", "PathInContainer": "/dev/davinci1"}
    )
    with pytest.raises(ValueError, match="extra device mappings"):
        validate_container_inspect(payload, assignment)


def test_inspect_rejects_missing_control_device() -> None:
    """A container missing a control device must be rejected (exact set)."""
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    # Remove the last control device.
    payload["HostConfig"]["Devices"].pop()
    with pytest.raises(ValueError, match="missing or incorrect"):
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


# --- sub-path blacklist (PR #150 review) ---


@pytest.mark.parametrize(
    "volume_spec",
    [
        "/proc/1/root:/workspace",  # procfs sub-path escape
        "/proc/self:/workspace",  # procfs sub-path
        "/sys/kernel:/workspace",  # sysfs sub-path
        "/sys/class/devdrv0:/workspace",  # NPU driver sysfs sub-path
        "/dev/shm:/workspace",  # /dev sub-path
        "/usr/local/Ascend/driver:/workspace",  # CANN sub-path
    ],
)
def test_docker_command_rejects_forbidden_subpath(volume_spec: str) -> None:
    """Forbidden host paths must also cover their sub-paths."""
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


# --- named volume rejection (PR #150 review) ---


@pytest.mark.parametrize(
    "volume_spec",
    [
        "my_vol:/workspace",  # named volume
        "data_vol:/data",  # named volume to allowed destination
    ],
)
def test_docker_command_rejects_named_volume(volume_spec: str) -> None:
    """Named volumes are rejected (local driver can bind / to a named volume)."""
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="named volumes are not allowed"):
        build_docker_create_command(
            assignment=assignment,
            container_name="benchmark-job",
            image="example/ascend:latest",
            command=[],
            volumes=[volume_spec],
        )


def test_docker_command_accepts_anonymous_volume() -> None:
    """Anonymous volumes (container path only) are safe and accepted."""
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    command = build_docker_create_command(
        assignment=assignment,
        container_name="benchmark-job",
        image="example/ascend:latest",
        command=[],
        volumes=["/tmp"],
    )
    assert "/tmp" in " ".join(command)


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


# --- named volume rejection in inspect (PR #150 review) ---


def test_inspect_rejects_named_volume() -> None:
    """Named volumes in inspect payload must be rejected."""
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    payload["Mounts"] = [
        {
            "Type": "volume",
            "Name": "my_vol",
            "Source": "/var/lib/docker/volumes/my_vol/_data",
            "Destination": "/workspace",
            "Mode": "rw",
            "RW": True,
            "Propagation": "rprivate",
        }
    ]
    with pytest.raises(ValueError, match="named volume"):
        validate_container_inspect(payload, assignment)


# --- sub-path rejection in inspect (PR #150 review) ---


@pytest.mark.parametrize(
    "source",
    [
        "/proc/1/root",  # procfs sub-path escape
        "/sys/kernel",  # sysfs sub-path
        "/dev/shm",  # /dev sub-path
        "/usr/local/Ascend/driver",  # CANN sub-path
    ],
)
def test_inspect_rejects_forbidden_subpath(source: str) -> None:
    """Forbidden host path sub-paths must be rejected in inspect too."""
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    payload["Mounts"] = [
        {
            "Type": "bind",
            "Source": source,
            "Destination": "/workspace",
            "Mode": "rw",
            "RW": True,
            "Propagation": "rprivate",
        }
    ]
    with pytest.raises(ValueError, match="forbidden bind mount source"):
        validate_container_inspect(payload, assignment)
