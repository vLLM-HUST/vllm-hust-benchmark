"""Watchdog ownership contract for Docker NPU jobs on poy-180 runners.

Issue #127: the poy-180 NPU watchdog identifies runner-owned processes by
container cgroup. Docker containers launched via the host Docker socket are
sibling cgroups on the host; without a runner ownership label the watchdog
treats their NPU processes as unowned and reclaims them.

This module resolves the runner name (``poy-180-21rc-npu0`` … ``npu3``) to
a single physical NPU device, builds a ``docker create`` command that carries
the ownership labels and device mapping, and validates the resulting
container's inspect payload so the contract holds before the container
starts.

Key contracts enforced:
- ``org.vllm-hust.runner=<runner-name>`` label on every NPU container.
- Only the runner's own ``/dev/davinciN`` is mapped, always to container
  logical device ``/dev/davinci0``.
- ``ASCEND_RT_VISIBLE_DEVICES=0`` / ``ASCEND_VISIBLE_DEVICES=0`` are set by
  the launcher and cannot be overridden.
- Volume/mount constraints: host path blacklist (``/dev``, ``/sys``, ``/proc``,
  Docker socket, CANN stack, …) and container destination whitelist
  (``/workspace``, ``/tmp``, ``/data``, …). Named volumes are rejected because
  the local driver can bind ``/`` or ``/dev`` to a named volume.
- Privileged mode and ``SYS_ADMIN``/``SYS_PTRACE``/``SYS_MODULE`` capabilities
  are rejected.
- Argument injection: ``image``, ``container_name`` and ``command`` elements
  starting with ``-`` are rejected, and a ``--`` separator is emitted before
  the image so Docker cannot reinterpret them as options.
- NPU device mapping is validated as an exact set: no extra ``davinci*``
  devices are allowed beyond the runner's own card and the control devices.
"""

from __future__ import annotations

import os
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
# Patterns use a trailing ``/?`` plus ``($|/)`` so that both the directory
# itself and any sub-path underneath it are rejected (e.g. ``/proc``,
# ``/proc/1/root``, ``/sys/kernel``, …).
FORBIDDEN_HOST_PATH_PATTERNS = (
    re.compile(r"^/+$"),  # root filesystem
    re.compile(r"^/dev/?($|/)"),  # /dev exposes all device nodes (+ subdirs)
    re.compile(r"^/dev/davinci\d+/?$"),  # NPU device nodes
    re.compile(r"^/dev/davinci_manager/?$"),  # NPU management device
    re.compile(r"^/dev/devmm_svm/?$"),  # NPU memory management device
    re.compile(r"^/dev/hisi_hdc/?$"),  # NPU debug device
    re.compile(r"^/sys/?($|/)"),  # sysfs (NPU driver sysfs) + subdirs
    re.compile(r"^/proc/?($|/)"),  # procfs (+ /proc/1/root escape)
    re.compile(r"^/(var/)?run/docker\.sock$"),  # Docker socket (container escape)
    re.compile(r"^/usr/local/Ascend/?($|/)"),  # CANN installation + subdirs
    re.compile(r"^/usr/local/slog/?($|/)"),  # NPU log directory + subdirs
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
    """Return True if bind-mounting host_path would bypass isolation.

    Checks BOTH the original path and the realpath-resolved path against
    the blacklist.  This is necessary because:

    1. ``realpath`` resolves symlinks like ``link -> /proc``, catching
       indirect references to forbidden paths (PR #150 review round 2).
    2. On macOS, ``/var`` is a symlink to ``/private/var``, so
       ``realpath("/var/run/docker.sock")`` returns
       ``/private/var/run/docker.sock`` — the original path must also be
       checked so direct patterns like ``/(var/)?run/docker.sock`` still
       match.

    ``os.path.realpath`` can raise ``PermissionError`` on Linux for paths
    like ``/proc/1/root`` (requires ``CAP_SYS_PTRACE``).  In that case the
    original-path check above has already matched, so we return False
    (not forbidden) — the symlink resolution is best-effort.

    Matches both the path itself and any sub-path underneath it (e.g.
    ``/proc`` and ``/proc/1/root``).
    """
    # Check the original path first — catches direct references.
    normalized = _normalize_path(host_path)
    if any(pattern.match(normalized) for pattern in FORBIDDEN_HOST_PATH_PATTERNS):
        return True
    # Then try realpath to catch symlinks like link -> /proc.
    try:
        resolved = os.path.realpath(host_path)
    except (OSError, PermissionError):
        # realpath can fail on /proc/1/root (Linux CAP_SYS_PTRACE required).
        # The original-path check above is sufficient in that case.
        return False
    normalized_resolved = _normalize_path(resolved)
    return any(
        pattern.match(normalized_resolved) for pattern in FORBIDDEN_HOST_PATH_PATTERNS
    )


def _is_allowed_container_destination(container_path: str) -> bool:
    """Return True if container_path is an allowed mount destination."""
    normalized = _normalize_path(container_path)
    for allowed in ALLOWED_CONTAINER_DESTINATIONS:
        allowed_norm = _normalize_path(allowed)
        if normalized == allowed_norm or normalized.startswith(allowed_norm + "/"):
            return True
    return False


def _reject_option_like(value: str, field_name: str) -> None:
    """Reject values that start with ``-`` to prevent argument injection.

    Docker parses tokens starting with ``-`` as options even after a value
    position. A caller passing ``--image=--device=/dev/davinci1:...`` could
    inject an extra device option. Rejecting any ``-``-prefixed value closes
    this entry.
    """
    if value.startswith("-"):
        raise ValueError(
            f"{field_name} must not start with '-': {value!r} "
            f"(would be interpreted as a docker option)"
        )


def _validate_volume_spec(volume_spec: str) -> None:
    """Validate a Docker volume spec before it reaches docker create.

    Format: ``[HOST_PATH:]CONTAINER_PATH[:OPTIONS]``.

    Only bind mounts (host path starting with ``/``) are allowed. Named
    volumes are rejected because the local driver can bind ``/`` or ``/dev``
    to a named volume, which would bypass the host-path blacklist.

    Raises ``ValueError`` when the spec would bypass single-card device
    isolation or mount into a non-allowlisted container path.
    """
    parts = volume_spec.split(":")
    if len(parts) == 1:
        # Anonymous volume (container path only, Docker-managed).
        # Rejected per PR #150 review round 2: 'build 仍接受只有 /tmp 的匿名
        # volume，docker inspect 会把它表示为 Type=volume，而 inspect
        # validator 现在拒绝所有 volume，因此这类命令必然 create 成功后再被
        # 拒绝。请让 create 前后的契约一致'.
        raise ValueError(
            f"anonymous volumes are not allowed (build/inspect contract "
            f"inconsistency): {volume_spec!r}; use an explicit bind mount "
            f"or tmpfs instead"
        )
    if len(parts) not in (2, 3):
        raise ValueError(
            f"invalid volume spec (expected 1-3 colon-separated parts): {volume_spec!r}"
        )
    host_path, container_path = parts[0], parts[1]
    options = parts[2] if len(parts) == 3 else ""

    # Reject named volumes: a local-driver named volume can bind ``/`` or
    # ``/dev`` to the volume, bypassing the host-path blacklist. Only bind
    # mounts (host path starting with ``/``) are allowed, and they are
    # checked against the blacklist below.
    if not host_path.startswith("/"):
        raise ValueError(
            f"named volumes are not allowed (use a bind mount instead): {volume_spec!r}"
        )

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
    """Resolved runner → physical NPU device mapping."""

    runner_name: str
    physical_device: int
    logical_device: int = 0

    @property
    def host_device_path(self) -> str:
        return f"/dev/davinci{self.physical_device}"

    @property
    def container_device_path(self) -> str:
        return f"/dev/davinci{self.logical_device}"

    @property
    def expected_device_mappings(self) -> frozenset[tuple[str, str]]:
        """Exact set of host→container device mappings the container must have."""
        mappings = {(self.host_device_path, self.container_device_path)}
        for device in CONTROL_DEVICES:
            mappings.add((device, device))
        return frozenset(mappings)


def resolve_runner_device(runner_name: str) -> RunnerDeviceAssignment:
    """Resolve a poy-180 runner name to its physical NPU device.

    Runner names follow ``poy-180-21rc-npu0`` … ``poy-180-21rc-npu3``. The
    trailing digit maps directly to the physical ``/dev/davinciN`` device.

    Raises ``ValueError`` if the name does not match the expected pattern.
    """
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
    """Validate user-supplied container environment variables.

    Reserved env names (``ASCEND_*VISIBLE_DEVICES``) are managed by the
    launcher and cannot be overridden.
    """
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
    command: Sequence[str] = (),
    volumes: Iterable[str] = (),
    extra_env: Iterable[str] = (),
    docker_bin: str = "docker",
) -> list[str]:
    """Build a ``docker create`` command carrying the watchdog ownership contract.

    The command includes:
    - ``org.vllm-hust.runner`` / ``npu-physical`` / ``npu-logical`` labels.
    - The runner's own ``/dev/davinciN`` mapped to container ``/dev/davinci0``.
    - Control devices (``davinci_manager``, ``devmm_svm``, ``hisi_hdc``).
    - ``ASCEND_RT_VISIBLE_DEVICES=0`` / ``ASCEND_VISIBLE_DEVICES=0``.
    - Validated volume mounts (no forbidden host paths or destinations).
    - Validated environment variables (no reserved names).
    - A ``--`` separator before the image so Docker cannot reinterpret
      ``image`` or ``command`` elements as options.

    ``image`` and ``container_name`` must not start with ``-``
    (argument-injection guard for the option region).  ``command`` elements
    after the ``--`` separator are positional args and are NOT rejected —
    normal commands like ``python -m vllm ... --model ...`` must work.

    Per PR #150 review round 2: '当前逐项拒绝 command 中以 - 开头的值会让
    正常的 python -m vllm ... --model ... 直接报错。请保留选项区边界保护，
    但不要禁止镜像后的正常参数'.
    """
    if not container_name.strip():
        raise ValueError("container name is required")
    if not image.strip():
        raise ValueError("container image is required")

    # Argument-injection guard: reject values that start with '-' for
    # fields in the Docker OPTION region (before ``--``).  ``image`` and
    # ``container_name`` are in this region, so they are checked.  ``command``
    # elements come AFTER ``--`` and are positional args — they must NOT be
    # rejected because normal commands use ``-``-prefixed flags.
    _reject_option_like(container_name, "container_name")
    _reject_option_like(image, "image")

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
    # ``--`` ends the option region so Docker treats the following tokens as
    # positional (image + command), never as options.
    docker_command.append("--")
    docker_command.append(image)
    docker_command.extend(command)
    return docker_command


def _validate_mount_entry(mount: Mapping[str, Any]) -> None:
    """Validate a single ``Mounts`` entry from docker inspect."""
    mount_type = mount.get("Type", "")
    # tmpfs mounts have no host source and are safe.
    if mount_type == "tmpfs":
        return

    source = mount.get("Source", "")
    destination = mount.get("Destination", "")
    propagation = mount.get("Propagation", "")

    # Reject named volumes: a local-driver named volume can bind ``/`` or
    # ``/dev`` to the volume, bypassing the host-path blacklist.
    if mount_type == "volume":
        raise ValueError(
            f"named volume {source!r} -> {destination!r} is not allowed: "
            f"local driver can bind forbidden host paths to a named volume; "
            f"use a bind mount instead"
        )

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
    """Validate a container's docker inspect payload against the ownership contract.

    Checks:
    - Runner / physical / logical labels match the assignment.
    - No forbidden bind mounts (both ``Mounts`` array and legacy ``Binds``).
    - No named volumes (local driver can bypass host-path blacklist).
    - No privileged mode or dangerous capabilities.
    - NPU device mapping is an exact set: only the runner's own ``davinciN``
      and the control devices are allowed; any extra ``davinci*`` device is
      rejected.
    - Logical-device environment variables are set.

    Raises ``ValueError`` on any mismatch (fail-closed).
    """
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

    # Exact-set device validation: the container must have exactly the
    # expected device mappings and no extra NPU devices. A container that
    # also maps /dev/davinci1 would bypass single-card isolation even though
    # the expected mapping is present.
    devices = host_config.get("Devices") or []
    mapped_devices = {
        (device.get("PathOnHost"), device.get("PathInContainer")) for device in devices
    }
    expected_mappings = assignment.expected_device_mappings
    extra_devices = mapped_devices - expected_mappings
    if extra_devices:
        raise ValueError(
            f"container has extra device mappings beyond the runner's own card "
            f"and control devices: {sorted(extra_devices)}; "
            f"only {sorted(expected_mappings)} are allowed"
        )
    missing_devices = expected_mappings - mapped_devices
    if missing_devices:
        raise ValueError(
            "container NPU mapping is missing or incorrect: "
            f"expected {sorted(expected_mappings)}, "
            f"missing {sorted(missing_devices)}"
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
