"""No real Docker or builds: test explicit architecture and trusted build scope."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "swe_images", SCRIPTS / "prepare_swebench_original_images.py"
)
images = importlib.util.module_from_spec(spec)
spec.loader.exec_module(images)


class FakeAPI:
    def __init__(self):
        self.calls = []

    def build(self, **kwargs):
        self.calls.append(kwargs)
        return iter([{"stream": "inert build fixture"}])


@pytest.mark.parametrize("network", ["bridge", "host"])
def test_network_override_is_per_trusted_build_only(tmp_path, network):
    original = FakeAPI()
    controlled = images.BuildAPI(original, ["inert-image"], tmp_path, network)
    assert list(
        controlled.build(tag="inert-image", path=str(tmp_path / "context"), decode=True)
    )
    assert controlled.calls == 1
    assert original.calls[0].get("network_mode") == (
        "host" if network == "host" else None
    )
    assert "privileged" not in original.calls[0]


def test_build_rejects_unselected_image_and_external_context(tmp_path):
    original = FakeAPI()
    controlled = images.BuildAPI(original, ["inert-image"], tmp_path, "host")
    with pytest.raises(ValueError):
        controlled.build(tag="different-image", path=str(tmp_path))
    with pytest.raises(ValueError):
        controlled.build(tag="inert-image", path=str(tmp_path.parent / "outside"))
    assert not original.calls


def test_default_plan_retains_arm_and_does_not_start_missing_runtime(tmp_path):
    record = {
        "instance_id": "fixture__repo-1",
        "repo": "fixture/repo",
        "base_commit": "inert",
        "version": "inert",
        "problem_statement": "inert issue",
        "patch": "inert",
        "test_patch": "inert",
        "FAIL_TO_PASS": "[]",
        "PASS_TO_PASS": "[]",
    }
    path = tmp_path / "inert.json"
    path.write_text(json.dumps([record]), encoding="utf-8")
    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_swebench_original_images.py"),
            "--source-id",
            "SWE_VERIFIED",
            "--records-file",
            str(path),
            "--harness-root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--docker-url",
            "unix:///inert.sock",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY" and report["architecture"] == "arm64"
    assert report["model_calls"] == 0 and not report["source_checkbox_eligible"]
    assert not output.exists()
