"""Synthetic contract fixtures; no model, task or container execution."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "arm_plan", ROOT / "scripts/prepare_pyramidkv_arm.py"
)
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


@pytest.fixture
def settings():
    return json.loads(
        (ROOT / "docs/examples/swe-bench-pro/single-agent.json").read_text()
    )


def test_swe_pair_only_changes_run_identity(settings):
    original = copy.deepcopy(settings)
    jobs = plan.swe_pair(
        [{"instance_id": "task-a", "repo": "owner/repo"}], ["task-a"], settings
    )
    assert settings == original
    assert jobs["b0"].pop("run_id") != jobs["b1"].pop("run_id")
    assert jobs["b0"] == jobs["b1"]
    assert jobs["b0"]["agent"]["seed"] == 17
    assert (
        jobs["b0"]["agent"]["wall_budget_seconds"]
        == settings["agent"]["wall_budget_seconds"]
    )
    assert jobs["b0"]["model"]["model_name"] == "openai/qwen35-pyramidkv"


@pytest.mark.parametrize("ids", [[], ["task-a", "task-a"], ["missing"], ["../escape"]])
def test_swe_rejects_selection_drift(settings, ids):
    with pytest.raises(ValueError):
        plan.swe_pair([{"instance_id": "task-a", "repo": "owner/repo"}], ids, settings)


def test_swe_requires_separate_repository_plan(settings):
    rows = [
        {"instance_id": "a", "repo": "one/repo"},
        {"instance_id": "b", "repo": "two/repo"},
    ]
    with pytest.raises(ValueError, match="separate plan"):
        plan.swe_pair(rows, ["a", "b"], settings)


@pytest.mark.parametrize(
    "field", ["integration", "task_instructions", "api_key", "endpoint"]
)
def test_swe_rejects_extra_hooks_and_credentials(settings, field):
    if field == "integration":
        settings[field] = {"hook": "synthetic:factory"}
    elif field == "task_instructions":
        settings["agent"][field] = "Synthetic extra instruction"
    elif field == "api_key":
        settings["model"][field] = "synthetic"
    else:
        settings["model"]["api_base"] = "https://example.invalid/v1?key=synthetic"
    with pytest.raises(ValueError):
        plan.swe_pair([{"instance_id": "a", "repo": "one/repo"}], ["a"], settings)


def test_reject_wrong_dataset_before_creating_output(tmp_path):
    data = tmp_path / "wrong.parquet"
    data.write_bytes(b"synthetic non-V1 bytes")
    with pytest.raises(ValueError, match="V1 parquet"):
        plan.prepare_swe(data, ["a"], tmp_path / "missing-template", tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.fixture
def terminal_fixture(tmp_path):
    assets = tmp_path / "assets"
    task = assets / "snapshot/tasks/synthetic-task"
    files = {
        "instruction.md": "synthetic task, not benchmark data\n",
        "solution/solve.sh": "#!/bin/sh\ntrue\n",
        "tests/test.sh": "#!/bin/sh\ntrue\n",
        "tests/test_outputs.py": "# synthetic verifier\n",
        "environment/Dockerfile": "FROM synthetic:base\n",
        "task.toml": '[environment]\ndocker_image="synthetic:original"\ncpus=1\nmemory_mb=2048\n[agent]\ntimeout_sec=900\n',
    }
    for name, text in files.items():
        p = task / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    reference = {
        "image_id": "sha256:" + "a" * 64,
        "task_files": {name: plan.sha(task / name) for name in files},
    }
    ref = tmp_path / "arm-reference.json"
    ref.write_text(json.dumps(reference))
    (assets / "tree.json").write_text(
        json.dumps(
            {
                "tree": [
                    {
                        "path": "tasks/synthetic-task/" + name,
                        "type": "blob",
                        "mode": "100755" if name.endswith(".sh") else "100644",
                    }
                    for name in files
                ]
            }
        )
    )
    return assets, task, ref, reference, files


def test_arm_reference_requires_immutable_image(terminal_fixture):
    _, task, _, reference, _ = terminal_fixture
    reference["image_id"] = "mutable:tag"
    with pytest.raises(ValueError, match="config digest"):
        plan.validate_arm_reference(reference, task)


def test_arm_reference_rejects_changed_verifier(terminal_fixture):
    _, task, _, reference, _ = terminal_fixture
    (task / "tests/test_outputs.py").write_text("# altered synthetic verifier\n")
    with pytest.raises(ValueError, match="verifier bytes"):
        plan.validate_arm_reference(reference, task)


def test_terminal_stages_image_only_and_stays_unqualified(
    terminal_fixture, tmp_path, monkeypatch
):
    assets, task, reference_path, reference, files = terminal_fixture

    def audited_fixture(_assets, output):
        output.mkdir(parents=True)
        (output / "inventory.json").write_text(
            json.dumps(
                [
                    {
                        "task_id": "synthetic-task",
                        "image": "synthetic:original",
                        "files": list(files),
                    }
                ]
            )
        )

    monkeypatch.setattr(plan.terminal.source_audit, "audit", audited_fixture)
    output = tmp_path / "out"
    result = plan.prepare_terminal(
        assets, "synthetic-task", reference_path, "http://localhost:8000/v1", output
    )
    assert result["ready_for_scored_execution"] is False
    assert result["historical_control_reused_as_current_qualification"] is False
    assert result["model_calls"] == result["tasks_executed"] == 0
    assert (
        output / "tasks/synthetic-task/solution/solve.sh"
    ).stat().st_mode & 0o777 == 0o755
    for name in files:
        if name != "task.toml":
            assert (output / "tasks/synthetic-task" / name).read_bytes() == (
                task / name
            ).read_bytes()
    staged = (output / "tasks/synthetic-task/task.toml").read_text()
    assert reference["image_id"] in staged and "timeout_sec=900" in staged
    jobs = [
        json.loads((output / f"job-{arm}.json").read_text()) for arm in ("b0", "b1")
    ]
    for job in jobs:
        assert job["environment"]["force_build"] is False
        del job["job_name"], job["jobs_dir"]
    assert jobs[0] == jobs[1]
    with pytest.raises(ValueError, match="overwrite"):
        plan.prepare_terminal(
            assets, "synthetic-task", reference_path, "http://localhost:8000/v1", output
        )
