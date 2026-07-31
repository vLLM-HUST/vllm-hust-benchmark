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
        docker_command.extend(("--volume", volume))
    for environment in _validate_extra_env(extra_env):
        docker_command.extend(("--env", environment))
    docker_command.append(image)
    docker_command.extend(command)
    return docker_command


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

    devices = (inspect_payload.get("HostConfig") or {}).get("Devices") or []
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
