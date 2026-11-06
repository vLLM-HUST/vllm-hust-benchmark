"""Docker-call fixtures only; no native task or actual image qualification."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bind_naturebench_task_images as runner

IDENTITY = "sha256:" + "a" * 64


def test_offline_plan_has_no_docker_calls():
    def forbidden(*args, **kwargs):
        raise AssertionError("offline plan must not call Docker")

    report = runner.bind({runner.TASKS[0]: IDENTITY}, call=forbidden)
    assert report["bindings"][0]["tag"] == f"naturebench-task-{runner.TASKS[0]}:base"
    assert report["native_tasks"] == 0 and not report["source_checkbox_eligible"]


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"invented": IDENTITY},
        {runner.TASKS[0]: "mutable:tag"},
        {runner.TASKS[0]: 1},
    ],
)
def test_bad_manifest_rejected(value):
    with pytest.raises(ValueError):
        runner.bind(value)


@pytest.mark.parametrize("conflict", [False, True])
def test_identity_checked_and_conflicting_tag_never_written(conflict):
    calls = []

    def call(argv, **kwargs):
        calls.append(argv)
        if argv[2] == "inspect":
            return SimpleNamespace(stdout=json.dumps([{"Id": IDENTITY}]))
        if argv[2] == "ls":
            return SimpleNamespace(
                stdout=("sha256:" + "b" * 64) if conflict else IDENTITY
            )
        return SimpleNamespace(stdout="")

    if conflict:
        with pytest.raises(ValueError, match="no overwrite"):
            runner.bind({runner.TASKS[0]: IDENTITY}, execute=True, call=call)
        assert not any(argv[2] == "tag" for argv in calls)
    else:
        report = runner.bind({runner.TASKS[0]: IDENTITY}, execute=True, call=call)
        assert report["status"] == "IMAGE_ALIASES_BOUND_NOT_NATIVE_EXECUTION"
        assert sum(argv[2] == "tag" for argv in calls) == 1
