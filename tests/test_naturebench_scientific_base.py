"""Software-only recipe fixtures, not native scientific task qualification."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_naturebench_scientific_base as runner


ORIGINAL = b"FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04\nRUN python -m pip install numpy==1.26.4\nRUN python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118\n"


@pytest.mark.parametrize("cached", [False, True])
def test_disclosed_recipe_preserves_fixed_versions(cached):
    raw = runner.derive(ORIGINAL, cached)
    assert runner.BASE.encode() in raw
    assert b"torch==2.6.0" in raw and b"numpy==1.26.4" in raw
    assert b"https://pypi.org/simple" in raw
    assert (b"COPY wheelhouse" in raw) == cached
    assert raw.count(b"RUN python -m pip install") == 2


@pytest.mark.parametrize("raw", [b"changed source", ORIGINAL + ORIGINAL])
def test_unknown_or_duplicate_original_rejected(raw):
    with pytest.raises(ValueError, match="no heuristic"):
        runner.derive(raw)


def test_second_task_preserves_original_dependencies():
    original = b"FROM naturebench-base:v3\nRUN python -m pip install gudhi==3.10.1\n"
    result = runner.second_task_recipe(original)
    assert b"FROM naturebench-gudhi:3.10.1-arm64" in result
    assert b"gudhi==3.10.1" in result
    assert b"PIP_CONSTRAINT=" in result and b"pip check" in result
    assert b"no model forward" in result
    with pytest.raises(ValueError, match="base differs"):
        runner.second_task_recipe(b"FROM changed-base\n")


def test_plan_no_output_and_write_preserves_original(tmp_path, monkeypatch):
    source = tmp_path / "original"
    (source / "docker").mkdir(parents=True)
    (source / "docker/Dockerfile.base").write_bytes(ORIGINAL)
    for name in ["LICENSE", "NOTICE"]:
        (source / name).write_text("fixture " + name)
    monkeypatch.setattr(
        runner, "original_files", lambda _: (source, {"revision": "fixture"})
    )
    output = tmp_path / "derived"
    plan = runner.prepare(source, output)
    assert plan["output_created"] is False and not output.exists()
    result = runner.prepare(source, output, write=True)
    assert result["source_checkbox_eligible"] is False
    assert result["model_calls"] == 0
    assert (output / "Dockerfile.base.original").read_bytes() == ORIGINAL
    assert (source / "docker/Dockerfile.base").read_bytes() == ORIGINAL
    with pytest.raises(FileExistsError):
        runner.prepare(source, output, write=True)
    with pytest.raises(ValueError, match="separate"):
        runner.prepare(source, source / "nested", write=True)
