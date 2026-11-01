"""Tiny protocol/source fixtures; no original task data, network or model calls."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "naturebench_contract", ROOT / "scripts/verify_naturebench_original_contract.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_original_pins_and_release_keep_task_data_license_separate():
    pins = json.loads(runner.PINS.read_text("utf-8"))
    release = json.loads(runner.RELEASE.read_text("utf-8"))
    assert pins["source_id"] == release["source_id"] == "NATUREBENCH"
    assert pins["revision"] == release["code_revision"]
    assert len(pins["files"]) == 56
    assert sum(item["bytes"] for item in pins["files"]) == 514543
    assert {item["path"] for item in pins["files"]} >= {
        "NOTICE",
        "LICENSE",
        "run_naturebench.py",
        "solve.py",
        "conda_env.yml",
        "conda_env_eval.yml",
        "task-set/all.txt",
        "task-set/naturebench-25/all.txt",
    }
    assert release["full_task_groups"] == {
        "all": 90,
        "cpu": 3,
        "gpu_high": 17,
        "gpu_low": 70,
    }
    assert "Third-party" in release["license_boundary"]
    assert "hidden ground truth" in release["agent_visibility"]


def test_source_plan_is_offline_and_does_not_create_author_root(tmp_path):
    target = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_jsonschemabench_author_source.py"),
            "--source",
            "naturebench",
            "--author-root",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["files"] == 56 and receipt["model_calls"] == 0
    assert not receipt["source_checkbox_eligible"] and not target.exists()


@pytest.mark.parametrize("fault", [None, "duplicate", "overlap"])
def test_original_group_partition_rejects_duplicates_and_overlap(tmp_path, fault):
    rows = {"all": ["a", "b", "c"], "cpu": ["a"], "gpu_high": ["b"], "gpu_low": ["c"]}
    if fault == "duplicate":
        rows["all"] = ["a", "a", "c"]
    if fault == "overlap":
        rows["gpu_low"] = ["b"]
    for name, values in rows.items():
        (tmp_path / f"{name}.txt").write_text("\n".join(values), "utf-8")
    counts = {key: len(value) for key, value in rows.items()}
    if fault:
        with pytest.raises(ValueError, match="uniqueness|partition"):
            runner.task_groups(tmp_path, counts)
    else:
        assert runner.task_groups(tmp_path, counts) == rows
