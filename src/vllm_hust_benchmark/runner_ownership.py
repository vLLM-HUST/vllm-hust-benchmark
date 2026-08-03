from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

RUNNER_LABEL = "org.vllm-hust.runner"
PHYSICAL_DEVICE_LABEL = "org.vllm-hust.npu-physical"
LOGICAL_DEVICE_LABEL = "org.vllm-hust.npu-logical"
RUNNER_PATTERN = re.compile(r"^.+-npu(?P<device>[0-3])$")
CONTROL_DEVICES = (
    "/dev/davinci_manager",
    "/dev/devmm_svm",
    "/dev/hisi_hdc",
)
RESERVED_ENV_NAMES = {
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
}

# Host paths that must never be bind-mounted into a container, because doing
# so would bypass single-card NPU device isolation or expose host resources.
FORBIDDEN_HOST_PATH_PATTERNS = (
    re.compile(r"^/+$"),  # root filesystem
    re.compile(r"^/dev/?$"),  # /dev exposes all device nodes
    re.compile(r"^/dev/davinci\d+$"),  # NPU device nodes
    re.compile(r"^/dev/davinci_manager$"),  # NPU management device
    re.compile(r"^/dev/devmm_svm$"),  # NPU memory management device
    re.compile(r"^/dev/hisi_hdc$"),  # NPU debug device
    re.compile(r"^/sys/?$"),  # sysfs (NPU driver sysfs)
    re.compile(r"^/sys/class/devdrv.*"),  # NPU driver sysfs entries
    re.compile(r"^/proc/?$"),  # procfs
    re.compile(r"^/(var/)?run/docker\.sock$"),  # Docker socket (container escape)
    re.compile(r"^/usr/local/Ascend/?$"),  # CANN installation
    re.compile(r"^/usr/local/slog/?$"),  # NPU log directory
    re.compile(r"^/etc/dcmi.*"),  # NPU DCMI configuration
)

# Container destinations allowed for bind mounts. Keeps the attack surface
# small by only permitting work-related paths inside the container.
ALLOWED_CONTAINER_DESTINATIONS = (
    "/workspace",
    "/tmp",
    "/data",
    "/root/.cache",
    "/home",
    "/opt/models",
)

# Mount options that weaken isolation.
FORBIDDEN_MOUNT_OPTIONS = {"shared", "slave"}


def _normalize_path(path: str) -> str:
    """Normalize a POSIX path for comparison (strip trailing slash)."""
    if path == "/":
        return "/"
    return path.rstrip("/")


def _is_forbidden_host_path(host_path: str) -> bool:
    """Return True if bind-mounting host_path would bypass isolation."""
    normalized = _normalize_path(host_path)
    return any(pattern.match(normalized) for pattern in FORBIDDEN_HOST_PATH_PATTERNS)


def _is_allowed_container_destination(container_path: str) -> bool:
    """Return True if container_path is an allowed mount destination."""
    normalized = _normalize_path(container_path)
    for allowed in ALLOWED_CONTAINER_DESTINATIONS:
        allowed_norm = _normalize_path(allowed)
        if normalized == allowed_norm or normalized.startswith(allowed_norm + "/"):
            return True
    return False


def _validate_volume_spec(volume_spec: str) -> None:
    """Validate a Docker volume spec before it reaches docker create.

    Format: [HOST_PATH:]CONTAINER_PATH[:OPTIONS].

    Raises ValueError when the spec would bypass single-card device isolation
    or mount into a non-allowlisted container path.
    """
    parts = volume_spec.split(":")
    if len(parts) == 1:
        # Anonymous volume (container path only, Docker-managed) — safe.
        return
    if len(parts) not in (2, 3):
        raise ValueError(
            f"invalid volume spec (expected 1-3 colon-separated parts): {volume_spec!r}"
        )
    host_path, container_path = parts[0], parts[1]
    options = parts[2] if len(parts) == 3 else ""

    # A leading "/" marks a bind mount (host path). Named volumes (e.g.
    # "my_vol:/workspace") are Docker-managed and safe.
    if host_path.startswith("/"):
        if _is_forbidden_host_path(host_path):
            raise ValueError(
                f"forbidden host path in volume spec {volume_spec!r}: "
                f"mounting {host_path!r} would bypass device isolation or "
                f"expose host resources; use --device for device access"
            )

    if not _is_allowed_container_destination(container_path):
        raise ValueError(
            f"forbidden container destination in volume spec {volume_spec!r}: "
            f"only {ALLOWED_CONTAINER_DESTINATIONS} are allowed, "
            f"got {container_path!r}"
        )

    if options:
        for opt in options.split(","):
            if opt.strip() in FORBIDDEN_MOUNT_OPTIONS:
                raise ValueError(
                    f"forbidden mount option in volume spec {volume_spec!r}: {opt!r}"
                )


@dataclass(frozen=True)
class RunnerDeviceAssignment:
    runner_name: str
    physical_device: int
    logical_device: int = 0

    @property
    def host_device_path(self) -> str:
        return f"/dev/davinci{self.physical_device}"

    @property
    def container_device_path(self) -> str:
        return f"/dev/davinci{self.logical_device}"


def resolve_runner_device(runner_name: str) -> RunnerDeviceAssignment:
    normalized = runner_name.strip()
    match = RUNNER_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"runner name must end in npu0, npu1, npu2, or npu3; got {runner_name!r}"
        )
    return RunnerDeviceAssignment(
        runner_name=normalized,
        physical_device=int(match.group("device")),
    )


def _validate_extra_env(extra_env: Iterable[str]) -> list[str]:
    validated: list[str] = []
    for item in extra_env:
        name, separator, _ = item.partition("=")
        if not separator or not name:
            raise ValueError(f"container environment must use NAME=value: {item!r}")
        if name in RESERVED_ENV_NAMES:
            raise ValueError(f"container environment {name} is managed by the launcher")
        validated.append(item)
    return validated


def build_docker_create_command(
    *,
    assignment: RunnerDeviceAssignment,
    container_name: str,
    image: str,
    command: Sequence[str],
    volumes: Iterable[str] = (),
    extra_env: Iterable[str] = (),
    docker_bin: str = "docker",
) -> list[str]:
    if not container_name.strip():
        raise ValueError("container name is required")
    if not image.strip():
        raise ValueError("container image is required")

    docker_command = [
        docker_bin,
        "create",
        "--name",
        container_name,
        "--label",
        f"{RUNNER_LABEL}={assignment.runner_name}",
        "--label",
        f"{PHYSICAL_DEVICE_LABEL}={assignment.physical_device}",
        "--label",
        f"{LOGICAL_DEVICE_LABEL}={assignment.logical_device}",
        "--device",
        f"{assignment.host_device_path}:{assignment.container_device_path}",
    ]
    for device in CONTROL_DEVICES:
        docker_command.extend(("--device", f"{device}:{device}"))
    docker_command.extend(
        ("--env", f"ASCEND_RT_VISIBLE_DEVICES={assignment.logical_device}")
    )
    docker_command.extend(
        ("--env", f"ASCEND_VISIBLE_DEVICES={assignment.logical_device}")
    )
    for volume in volumes:
        _validate_volume_spec(volume)
        docker_command.extend(("--volume", volume))
    for environment in _validate_extra_env(extra_env):
        docker_command.extend(("--env", environment))
    docker_command.append(image)
    docker_command.extend(command)
    return docker_command


def _validate_mount_entry(mount: Mapping[str, Any]) -> None:
    """Validate a single Mounts entry from docker inspect."""
    mount_type = mount.get("Type", "")
    # tmpfs mounts have no host source and are safe.
    if mount_type == "tmpfs":
        return

    source = mount.get("Source", "")
    destination = mount.get("Destination", "")
    propagation = mount.get("Propagation", "")

    if mount_type == "bind" and source.startswith("/"):
        if _is_forbidden_host_path(source):
            raise ValueError(
                f"forbidden bind mount source {source!r} -> {destination!r}: "
                f"this path would bypass NPU device isolation; "
                f"use --device for device access"
            )

    if destination and not _is_allowed_container_destination(destination):
        raise ValueError(
            f"forbidden mount destination {destination!r}: "
            f"only {ALLOWED_CONTAINER_DESTINATIONS} are allowed"
        )

    if propagation in FORBIDDEN_MOUNT_OPTIONS:
        raise ValueError(
            f"forbidden mount propagation {propagation!r} for "
            f"{source!r} -> {destination!r}"
        )


def validate_container_inspect(
    inspect_payload: Mapping[str, Any], assignment: RunnerDeviceAssignment
) -> None:
    config = inspect_payload.get("Config") or {}
    labels = config.get("Labels") or {}
    expected_labels = {
        RUNNER_LABEL: assignment.runner_name,
        PHYSICAL_DEVICE_LABEL: str(assignment.physical_device),
        LOGICAL_DEVICE_LABEL: str(assignment.logical_device),
    }
    for name, expected in expected_labels.items():
        actual = labels.get(name)
        if actual != expected:
            raise ValueError(
                f"container label {name}={actual!r}, expected {expected!r}"
            )

    host_config = inspect_payload.get("HostConfig") or {}

    # Reject bind mounts that would bypass device isolation. Docker inspect
    # exposes both the modern Mounts array and the legacy Binds list.
    mounts = inspect_payload.get("Mounts") or []
    for mount in mounts:
        _validate_mount_entry(mount)

    binds = host_config.get("Binds") or []
    for bind_spec in binds:
        _validate_volume_spec(bind_spec)

    # Privileged mode and dangerous capabilities bypass all isolation.
    if host_config.get("Privileged"):
        raise ValueError("container must not run in privileged mode")
    cap_add = host_config.get("CapAdd") or []
    dangerous_caps = {"SYS_ADMIN", "SYS_PTRACE", "SYS_MODULE"}
    present = dangerous_caps.intersection(cap_add)
    if present:
        raise ValueError(
            f"container must not have dangerous capabilities: {sorted(present)}"
        )

    devices = host_config.get("Devices") or []
    mapped_devices = {
        (device.get("PathOnHost"), device.get("PathInContainer")) for device in devices
    }
    expected_mapping = (
        assignment.host_device_path,
        assignment.container_device_path,
    )
    if expected_mapping not in mapped_devices:
        raise ValueError(
            "container NPU mapping is missing or incorrect: "
            f"expected {expected_mapping[0]} -> {expected_mapping[1]}"
        )

    environment = config.get("Env") or []
    expected_environment = {
        f"ASCEND_RT_VISIBLE_DEVICES={assignment.logical_device}",
        f"ASCEND_VISIBLE_DEVICES={assignment.logical_device}",
    }
    missing_environment = expected_environment.difference(environment)
    if missing_environment:
        raise ValueError(
            "container logical-device environment is incomplete: "
            + ", ".join(sorted(missing_environment))
        )
