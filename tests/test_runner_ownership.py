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
- Argument injection: image/name starting with ``-`` is rejected; a
  ``--`` separator is emitted before the image.  ``command`` elements
  after ``--`` are NOT rejected (normal commands use ``-`` flags).
- Host path blacklist covers sub-paths (``/proc/1/root``, ``/sys/kernel``)
  and resolves symlinks via ``realpath`` so ``link -> /proc`` is caught.
- Named volumes are rejected (local driver can bind ``/`` to a named volume).
- Anonymous volumes are rejected (build/inspect contract consistency).
- Device mapping is an exact set: extra ``davinci*`` devices are rejected.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

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
def test_docker_command_accepts_option_like_command_element(cmd_element: str) -> None:
    """Command elements after ``--`` are positional args, NOT docker options.

    Per PR #150 review round 2: '请保留选项区边界保护，但不要禁止镜像后的
    正常参数'.  Normal commands like ``python -m vllm ... --model ...`` use
    ``-``-prefixed flags and must be accepted.
    """
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    command = build_docker_create_command(
        assignment=assignment,
        container_name="benchmark-job",
        image="example/ascend:latest",
        command=[cmd_element, "ubuntu"],
    )
    # The element must appear after the ``--`` separator.
    sep_index = command.index("--")
    assert cmd_element in command[sep_index + 1 :]


def test_docker_command_accepts_normal_vllm_invocation() -> None:
    """A real-world vLLM command with ``-`` flags must be accepted."""
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    command = build_docker_create_command(
        assignment=assignment,
        container_name="benchmark-job",
        image="example/ascend:latest",
        command=[
            "python",
            "-m",
            "vllm.entrypoints.cli",
            "--model",
            "/data/model",
            "--port",
            "8000",
        ],
    )
    sep_index = command.index("--")
    # After `--` comes the image, then the command elements.
    assert command[sep_index + 1] == "example/ascend:latest"
    assert command[sep_index + 2 :] == [
        "python",
        "-m",
        "vllm.entrypoints.cli",
        "--model",
        "/data/model",
        "--port",
        "8000",
    ]


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


def test_docker_command_rejects_anonymous_volume() -> None:
    """Anonymous volumes are rejected to keep build/inspect contracts consistent.

    Per PR #150 review round 2: 'build 仍接受只有 /tmp 的匿名 volume，docker
    inspect 会把它表示为 Type=volume，而 inspect validator 现在拒绝所有
    volume，因此这类命令必然 create 成功后再被拒绝。请让 create 前后的
    契约一致'.
    """
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="anonymous volumes are not allowed"):
        build_docker_create_command(
            assignment=assignment,
            container_name="benchmark-job",
            image="example/ascend:latest",
            command=[],
            volumes=["/tmp"],
        )


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


# --- symlink resolution (PR #150 review round 2) ---


def test_docker_command_rejects_symlink_to_proc(tmp_path: Path) -> None:
    """A symlink to a forbidden path must be caught via realpath.

    Per PR #150 review round 2: '_normalize_path 仍然只做 rstrip，没有解析
    bind source 的符号链接。我本地验证 /tmp/safe/link -> /proc 时
    _is_forbidden_host_path(link) 返回 false'.
    """
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    link = safe_dir / "link"
    link.symlink_to("/proc")
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="forbidden host path"):
        build_docker_create_command(
            assignment=assignment,
            container_name="benchmark-job",
            image="example/ascend:latest",
            command=[],
            volumes=[f"{link}:/workspace"],
        )


def test_docker_command_rejects_symlink_to_dev(tmp_path: Path) -> None:
    """A symlink to /dev must be caught via realpath."""
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    link = safe_dir / "devlink"
    link.symlink_to("/dev")
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    with pytest.raises(ValueError, match="forbidden host path"):
        build_docker_create_command(
            assignment=assignment,
            container_name="benchmark-job",
            image="example/ascend:latest",
            command=[],
            volumes=[f"{link}:/workspace"],
        )


def test_inspect_rejects_symlink_to_proc(tmp_path: Path) -> None:
    """Inspect path must also resolve symlinks (same contract as build).

    Per PR #150 review round 2: '在 inspect 路径上保持同一契约'.
    """
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    link = safe_dir / "link"
    link.symlink_to("/proc")
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    payload = _inspect_payload(assignment.runner_name, 0)
    payload["Mounts"] = [
        {
            "Type": "bind",
            "Source": str(link),
            "Destination": "/workspace",
            "Mode": "rw",
            "RW": True,
            "Propagation": "rprivate",
        }
    ]
    with pytest.raises(ValueError, match="forbidden bind mount source"):
        validate_container_inspect(payload, assignment)


def test_docker_command_accepts_symlink_to_safe_path(tmp_path: Path) -> None:
    """A symlink to a non-forbidden path must be accepted."""
    real_data = tmp_path / "real_data"
    real_data.mkdir()
    link = tmp_path / "link_to_data"
    link.symlink_to(real_data)
    assignment = resolve_runner_device("poy-180-21rc-npu0")
    command = build_docker_create_command(
        assignment=assignment,
        container_name="benchmark-job",
        image="example/ascend:latest",
        command=[],
        volumes=[f"{link}:/workspace"],
    )
    assert str(link) in " ".join(command)


# ---------------------------------------------------------------------------
# recursive mount propagation rejection (PR #150 review round 3)
# ---------------------------------------------------------------------------


def test_recursive_mount_propagation_rejected():
    """rshared and rslave must be rejected just like shared and slave."""
    from vllm_hust_benchmark.runner_ownership import _validate_volume_spec

    for option in ("shared", "slave", "rshared", "rslave"):
        with pytest.raises(ValueError, match="forbidden mount option"):
            _validate_volume_spec(f"/tmp:/workspace:{option}")


def test_inspect_rejects_recursive_mount_propagation():
    """Inspect must reject rshared and rslave propagation."""
    from vllm_hust_benchmark.runner_ownership import _validate_mount_entry

    for prop in ("rshared", "rslave"):
        with pytest.raises(ValueError, match="forbidden mount propagation"):
            _validate_mount_entry(
                {
                    "Type": "bind",
                    "Source": "/tmp",
                    "Destination": "/workspace",
                    "Propagation": prop,
                }
            )


def test_dangerous_capabilities_is_module_constant():
    """DANGEROUS_CAPABILITIES must be a module-level frozenset for consistency."""
    import vllm_hust_benchmark.runner_ownership as mod

    assert hasattr(mod, "DANGEROUS_CAPABILITIES")
    assert isinstance(mod.DANGEROUS_CAPABILITIES, frozenset)
    assert "SYS_ADMIN" in mod.DANGEROUS_CAPABILITIES
    assert "SYS_PTRACE" in mod.DANGEROUS_CAPABILITIES
    assert "SYS_MODULE" in mod.DANGEROUS_CAPABILITIES


# ---------------------------------------------------------------------------
# launcher script contract (PR #150 review round 3)
# ---------------------------------------------------------------------------


def test_launcher_does_not_accept_runner_name_arg():
    """The launcher must not allow --runner-name to be overridden by the caller.

    Runner identity must come from the RUNNER_NAME env var only.
    """
    import pathlib

    script_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "run-watchdog-owned-npu-container.py"
    )
    assert script_path.exists()
    text = script_path.read_text(encoding="utf-8")
    # --runner-name must NOT be a CLI argument
    assert '"--runner-name"' not in text
    assert "'--runner-name'" not in text
    # RUNNER_NAME must be read from environment
    assert "os.environ" in text
    assert "RUNNER_NAME" in text


def test_launcher_requires_runner_name_env(monkeypatch):
    """The launcher must fail with exit code 2 when RUNNER_NAME is not set."""
    import importlib.util
    import pathlib

    script_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "run-watchdog-owned-npu-container.py"
    )
    spec = importlib.util.spec_from_file_location("run_launcher", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.delenv("RUNNER_NAME", raising=False)
    rc = mod.run(["--name", "test", "--image", "ubuntu", "--", "echo", "hi"])
    assert rc == 2


def test_launcher_uses_runner_name_env(monkeypatch):
    """The launcher resolves the runner from RUNNER_NAME env var."""
    import importlib.util
    import pathlib

    script_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "run-watchdog-owned-npu-container.py"
    )
    spec = importlib.util.spec_from_file_location("run_launcher", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setenv("RUNNER_NAME", "poy-180-21rc-npu2")

    calls = []

    class FakeResult:
        def __init__(self, stdout="", stderr="", returncode=0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["docker", "create"]:
            return FakeResult(stdout="abc123\n")
        if cmd[:2] == ["docker", "inspect"]:
            return FakeResult(
                stdout='[{"Config": {"Labels": {"org.vllm-hust.runner": "poy-180-21rc-npu2", "org.vllm-hust.npu-physical": "2", "org.vllm-hust.npu-logical": "0"}, "Env": ["ASCEND_RT_VISIBLE_DEVICES=0", "ASCEND_VISIBLE_DEVICES=0"]}, "HostConfig": {"Privileged": false, "CapAdd": [], "Devices": [{"PathOnHost": "/dev/davinci2", "PathInContainer": "/dev/davinci0"}, {"PathOnHost": "/dev/davinci_manager", "PathInContainer": "/dev/davinci_manager"}, {"PathOnHost": "/dev/devmm_svm", "PathInContainer": "/dev/devmm_svm"}, {"PathOnHost": "/dev/hisi_hdc", "PathInContainer": "/dev/hisi_hdc"}], "Binds": []}, "Mounts": []}]'
            )
        if cmd[:2] == ["docker", "start"]:
            return FakeResult()
        if cmd[:2] == ["docker", "wait"]:
            return FakeResult(stdout="0\n")
        if cmd[:2] == ["docker", "logs"]:
            return FakeResult()
        if cmd[:2] == ["docker", "rm"]:
            return FakeResult()
        return FakeResult()

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = mod.run(
        ["--name", "test-container", "--image", "ubuntu", "--", "echo", "hello"]
    )
    assert rc == 0
    # Verify docker create was called
    assert any(c[:2] == ["docker", "create"] for c in calls)
