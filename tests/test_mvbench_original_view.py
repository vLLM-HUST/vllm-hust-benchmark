import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/prepare_mvbench_original_view.py"
SPEC = importlib.util.spec_from_file_location("mvbench_original_view", ENTRY)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_plan_does_not_require_files_or_optional_video_packages(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ENTRY),
            "--official-checkout",
            str(tmp_path / "absent-code"),
            "--snapshot",
            str(tmp_path / "absent-data"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLAN_ONLY"
    assert plan["original_tasks"] == 4000
    assert plan["model_calls"] == 0
    assert list(tmp_path.iterdir()) == []


def test_unverified_source_cannot_execute(tmp_path):
    marker = tmp_path / "unexpected"
    source = f"open({str(marker)!r}, 'w').write('unsafe')".encode()
    with pytest.raises(ValueError, match="Unverified original"):
        MODULE.original_generator(source)
    assert not marker.exists()


def test_fixed_sample_covers_all_twenty_original_types():
    profile = json.loads(
        (ROOT / "configs/workload-native/mvbench-original-protocol.json").read_text(
            encoding="utf-8"
        )
    )
    chosen = profile["fixed_selection"]
    assert len(chosen) == 60
    tasks = {row["task"] for row in chosen}
    assert len(tasks) == 20
    for task in tasks:
        assert [row["original_offset"] for row in chosen if row["task"] == task] == [
            0,
            1,
            2,
        ]
    assert profile["dataset_revision"] == MODULE.REVISION


def test_native_runner_plan_without_video_or_model_access(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_mvbench_original.py"),
            "--official-checkout",
            str(tmp_path / "missing"),
            "--snapshot",
            str(tmp_path / "missing"),
            "--frame-cache",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "output"),
            "--model",
            "example",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["selected_original_tasks"] == 60
    assert plan["frames_each"] == 16
    assert plan["model_calls"] == 0
    assert list(tmp_path.iterdir()) == []


def test_native_fixed_positions_preserves_all_groups_and_source_indices(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_mvbench_original as runner

    data = [{"task_type": f"task-{task}"} for task in range(20) for _ in range(4)]
    expected = [task * 4 + offset for task in range(20) for offset in range(3)]
    assert runner.fixed_positions(data) == expected
    with pytest.raises(ValueError, match="twenty-type"):
        runner.fixed_positions(data[:76])
