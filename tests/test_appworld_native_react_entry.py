"""Public entry safety/contracts only; not native execution evidence."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ENTRY = Path(__file__).resolve().parents[1] / "scripts/run_appworld_original_react.py"


def module():
    spec = importlib.util.spec_from_file_location("appworld_native_entry", ENTRY)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_plan_needs_no_runtime_or_data(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "--profile",
            str(tmp_path / "absent"),
            "--model",
            "openai/actual-model",
            "--api-base",
            "http://127.0.0.1:1234/v1",
            "--environment-url",
            "http+unix://%2Fio%2Fenvironment.sock",
            "--experiment",
            "test-original",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY"
    assert report["model_calls"] == 0
    assert not list(tmp_path.iterdir())


def test_modified_original_profile_rejected(tmp_path):
    path = tmp_path / "changed.jsonnet"
    path.write_text("{}")
    with pytest.raises(ValueError, match="unchanged pinned original"):
        module().validate_profile(path)


@pytest.mark.parametrize("name", ["..", ".", "../escape", "a/b", "a\\b", ""])
def test_experiment_path_escape_rejected(name):
    with pytest.raises(ValueError, match="simple experiment"):
        module().run(SimpleNamespace(limit=3, experiment=name, model="actual"))


def test_experiment_image_keeps_original_dependency_version():
    text = (
        ENTRY.parents[1] / "configs/workload-native/appworld-react-cpu.Dockerfile"
    ).read_text()
    assert "APPWORLD_CORE_IMAGE" in text
    assert "appworld[experiments]==0.1.3.post1" in text
    assert "=='1.37.19'" in text
    assert "requests-unixsocket==0.4.1" in text
    assert "USER 65534:65534" in text
    assert "COPY data" not in text and "COPY models" not in text
