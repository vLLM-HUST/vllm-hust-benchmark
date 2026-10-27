"""Inert software fixtures; no container or generated-code execution."""

from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


launcher = load("bcb_container_launcher", "run_bigcodebench_container_scoring.py")
worker = load("bcb_container_worker", "bigcodebench_container_worker.py")


def test_container_flags_are_offline_nonroot_bounded_and_private(tmp_path):
    command = launcher.container_command(
        "docker",
        "sha256:" + "a" * 64,
        "fixture",
        tmp_path,
        65534,
        {"HF_HUB_OFFLINE": "1"},
    )
    assert "--pull=never" in command and "--network=none" in command
    assert "--read-only" in command and "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges:true" in command
    assert command[command.index("--user") + 1] == "65534:65534"
    assert command[command.index("--memory") + 1] == "32g"
    assert command[command.index("--pids-limit") + 1] == "1024"
    assert "as=32212254720:32212254720" in command
    mounts = [command[i + 1] for i, x in enumerate(command) if x == "--mount"]
    assert len(mounts) == 3
    assert sum(m.endswith(",readonly") for m in mounts) == 2
    assert not any(
        x in command
        for x in ("--privileged", "--device", "--gpus", "--pid=host", "--rm")
    )
    assert command[-1] == "--evaluate"


@pytest.mark.parametrize(
    "image,uid",
    [("latest", 65534), ("sha256:" + "a" * 64, 0), ("sha256:" + "z" * 64, 65534)],
)
def test_mutable_images_or_root_uid_rejected(tmp_path, image, uid):
    with pytest.raises(ValueError):
        launcher.container_command("docker", image, "fixture", tmp_path, uid, {})


def result():
    ids = ["BigCodeBench/5", "BigCodeBench/8", "BigCodeBench/11"]
    evaluation = {
        "eval": {
            task: [{"task_id": task, "status": status}]
            for task, status in zip(ids, ["pass", "fail", "timeout"])
        }
    }
    metrics = {"gt_pass_rate": 1.0, "failed_tasks": [], "pass@1": 1 / 3}
    return evaluation, metrics, ids


def test_wrong_answers_remain_valid_scored_results():
    evaluation, metrics, ids = result()
    assert (
        worker.validate_results(evaluation, metrics, ids)["official_pass_at_1"] == 1 / 3
    )


@pytest.mark.parametrize(
    "failure", ["reference", "missing", "metric", "status", "duplicate"]
)
def test_environment_or_coverage_failure_cannot_be_accepted(failure):
    evaluation, metrics, ids = deepcopy(result())
    if failure == "reference":
        metrics["gt_pass_rate"] = 2 / 3
    elif failure == "missing":
        evaluation["eval"].pop(ids[0])
    elif failure == "metric":
        metrics["pass@1"] = 1.0
    elif failure == "status":
        evaluation["eval"][ids[0]][0]["status"] = "environment_error"
    else:
        evaluation["eval"][ids[0]].append(evaluation["eval"][ids[0]][0].copy())
    with pytest.raises(RuntimeError):
        worker.validate_results(evaluation, metrics, ids)


def test_worker_refuses_current_unqualified_host_before_import():
    with pytest.raises(RuntimeError, match="private nonroot PID1"):
        worker.assert_isolation({})


def test_cli_import_probe_is_separate_from_candidate_execution(monkeypatch):
    calls = []
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    worker.qualify_author_entrypoints()
    command, options = calls[0]
    assert command[:2] == [sys.executable, "-c"]
    assert "bigcodebench.sanitize" in command[2]
    assert "bigcodebench.evaluate" in command[2]
    assert "--samples" not in command and "--execution" not in command
    assert options == {"check": True, "timeout": 120}


@pytest.mark.parametrize("failure", ["missing", "reordered", "duplicate", "nonobject"])
def test_sanitize_selection_checked_before_evaluation(tmp_path, monkeypatch, failure):
    ids = ["BigCodeBench/0", "BigCodeBench/1", "BigCodeBench/2"]
    records = [{"task_id": task, "solution": ""} for task in ids]
    if failure == "missing":
        records.pop()
    elif failure == "reordered":
        records.reverse()
    elif failure == "duplicate":
        records.append(records[0])
    else:
        records[0] = None
    sanitized = tmp_path / "sanitized.jsonl"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        sanitized.write_text("\n".join(json.dumps(record) for record in records))

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    commands = [
        ["python", "-m", "bigcodebench.sanitize"],
        ["python", "-m", "bigcodebench.evaluate"],
    ]
    with pytest.raises(RuntimeError, match="sanitizer"):
        worker.run_official_steps(commands, ids, sanitized)
    assert len(calls) == 1
    assert calls[0][2] == "bigcodebench.sanitize"


def test_empty_sanitized_solution_is_not_replaced_or_rejected(tmp_path, monkeypatch):
    sanitized = tmp_path / "sanitized.jsonl"
    sanitized.write_text(json.dumps({"task_id": "BigCodeBench/0", "solution": ""}))
    calls = []
    monkeypatch.setattr(
        worker.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    commands = [
        ["python", "-m", "bigcodebench.sanitize"],
        ["python", "-m", "bigcodebench.evaluate"],
    ]
    worker.run_official_steps(commands, ["BigCodeBench/0"], sanitized)
    assert len(calls) == 2
    assert calls[1][2] == "bigcodebench.evaluate"


def test_dependency_probe_keeps_pins_and_pip_check(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==1.21.2\nrequests==2.31.0\n")
    monkeypatch.setattr(worker.sys, "version_info", (3, 10, 0))
    versions = {"numpy": "1.21.2", "requests": "2.31.0"}
    monkeypatch.setattr(worker.metadata, "version", lambda name: versions[name])
    calls = []
    monkeypatch.setattr(
        worker.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    assert worker.qualify_environment(requirements) == versions
    assert calls[0][0][0][-2:] == ["pip", "check"]
    assert calls[0][1]["check"] is True
    versions["numpy"] = "newer-unqualified"
    with pytest.raises(RuntimeError, match="dependency version mismatch"):
        worker.qualify_environment(requirements)
