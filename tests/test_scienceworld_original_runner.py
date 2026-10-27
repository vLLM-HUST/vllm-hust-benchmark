"""Check native author protocol without model, Java, or source downloads."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "scienceworld_runner", scripts / "run_scienceworld_original.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_author_arguments_preserved(runner, tmp_path):
    args = runner.native_args(tmp_path / "out", tmp_path / "source")
    assert args["env_step_limit"] == 100
    assert args["set"] == "test_mini"
    assert args["simplification_str"] == "easy"
    assert args["model_name"] == "gpt-4"
    assert args["no_stop"] is True


def test_source_identity(runner, tmp_path, monkeypatch):
    data = b"author original\n"
    path = tmp_path / "original.py"
    path.write_bytes(data)
    item = {
        "path": path.name,
        "size": len(data),
        "sha": hashlib.sha1(
            f"blob {len(data)}".encode() + bytes([0]) + data
        ).hexdigest(),
    }
    monkeypatch.setattr(runner, "PINNED_FILES", [item])
    assert runner.verify_source(tmp_path) == tmp_path.resolve()
    path.write_bytes(b"changed prompt\n")
    with pytest.raises(ValueError, match="identity"):
        runner.verify_source(tmp_path)


def test_remote_endpoint_rejected(runner):
    with pytest.raises(ValueError, match="loopback"):
        runner.local_endpoint("https://example.com/v1")


def test_all_thirty_original_task_indices_available(runner):
    assert runner.task_indices(",".join(str(i) for i in range(30))) == list(range(30))
    assert runner.task_indices("0,1,2") == [0, 1, 2]
    for value in ("-1", "30", "0,0"):
        with pytest.raises(ValueError):
            runner.task_indices(value)


def test_full_native_test_split_is_available_without_cutoff(runner, tmp_path):
    config = runner.native_args(tmp_path / "out", tmp_path / "source", "test")
    assert config["set"] == "test"
    assert config["cut_off"] is False
    assert config["env_step_limit"] == 100


def test_prompt_and_native_dependencies_pinned(runner):
    assert len(runner.PINNED_FILES) == 6
    assert any(
        item["path"] == "baselines/ReAct_baseline/prompt.jsonl"
        for item in runner.PINNED_FILES
    )
    assert any(item["path"] == "eval_utils.py" for item in runner.PINNED_FILES)
