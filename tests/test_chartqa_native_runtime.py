"""CPU setup safety fixtures only; no compilation or native benchmark calls."""

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "chartqa_runtime", ROOT / "scripts/prepare_chartqa_native_runtime.py"
)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def test_pins_include_actual_original_submodule_versions():
    assert set(runtime.PINS) == {"ffmpeg", "decord", "dlpack", "dmlc-core"}
    for repository, revision, tree in runtime.PINS.values():
        assert "/" in repository
        assert re.fullmatch("[0-9a-f]{40}", revision)
        assert re.fullmatch("[0-9a-f]{40}", tree)


def test_default_plan_makes_no_output(tmp_path):
    target = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_chartqa_native_runtime.py"),
            "--output",
            str(target),
            "--python",
            sys.executable,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["status"] == "PLAN_ONLY_NO_BUILD_OR_MODEL_CALL"
    assert not target.exists()


def test_existing_directory_is_not_overwritten(tmp_path):
    with pytest.raises(ValueError, match="Fresh"):
        runtime.run(tmp_path, Path(sys.executable), 4)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("jobs", [0, -1, 17])
def test_no_unbounded_build_parallelism(tmp_path, jobs):
    target = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="CPU build jobs"):
        runtime.run(target, Path(sys.executable), jobs)
    assert not target.exists()
