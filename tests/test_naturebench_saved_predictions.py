"""Fixture-only isolation and provenance checks, not native science results."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "saved_scoring",
    Path(__file__).parents[1] / "scripts/score_naturebench_saved_predictions.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def args(tmp_path, monkeypatch):
    data = tmp_path / "data"
    evaluator = data / "tasks/original-id/evaluation"
    evaluator.mkdir(parents=True)
    (evaluator / "evaluator.py").write_text("# original fixture")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    monkeypatch.setattr(
        runner,
        "original_cpu_packages",
        lambda *a, **k: iter([{"source": {"revision": "fixture"}}]),
    )
    return SimpleNamespace(
        data_root=data,
        predictions=predictions,
        output=tmp_path / "fresh",
        task_id="original-id",
        image="sha256:" + "a" * 64,
    )


def test_plan_isolated_and_does_not_run_generated_code(args):
    result = runner.plan(args)
    command = result["command"]
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert not any("docker.sock" in value or "/dev/" in value for value in command)
    assert command[-2:] == ["python", "/evaluation/evaluator.py"]
    assert result["new_model_calls"] == 0
    assert result["source_checkbox_eligible"] is False
    assert not args.output.exists()


def test_mutable_image_rejected(args):
    args.image = "latest"
    with pytest.raises(ValueError, match="immutable"):
        runner.plan(args)


def test_existing_output_not_overwritten(args):
    args.output.mkdir()
    with pytest.raises(FileExistsError):
        runner.plan(args)


def test_output_cannot_overlap_predictions(args):
    args.output = args.predictions / "new"
    with pytest.raises(ValueError, match="separate"):
        runner.plan(args)
