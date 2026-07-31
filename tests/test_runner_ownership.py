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


@pytest.mark.parametrize("physical_device", range(4))
def test_resolve_runner_device(physical_device: int) -> None:
    assignment = resolve_runner_device(f"poy-180-21rc-npu{physical_device}")
    assert assignment.physical_device == physical_device
    assert assignment.logical_device == 0


@pytest.mark.parametrize("runner_name", ["", "npu4", "poy-180", "npu-2"])
def test_resolve_runner_device_rejects_unknown_runner(runner_name: str) -> None:
    with pytest.raises(ValueError, match="runner name must end"):
        resolve_runner_device(runner_name)


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


def test_validate_container_inspect_accepts_matching_contract() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu3")
    validate_container_inspect(_inspect_payload(assignment.runner_name, 3), assignment)


def test_validate_container_inspect_rejects_wrong_device() -> None:
    assignment = resolve_runner_device("poy-180-21rc-npu3")
    payload = _inspect_payload(assignment.runner_name, 3)
    payload["HostConfig"]["Devices"][0]["PathOnHost"] = "/dev/davinci2"
    with pytest.raises(ValueError, match="mapping is missing or incorrect"):
        validate_container_inspect(payload, assignment)
