"""Original scoring adapter controls using inert records and fake Docker only."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "swe_grade", SCRIPTS / "run_swebench_original_grade.py"
)
grade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grade)


def rows():
    return [
        {
            "instance_id": "fixture__repo-1",
            "repo": "fixture/repo",
            "base_commit": "inert",
            "version": "inert",
            "problem_statement": "inert issue",
            "patch": "inert reference patch",
            "test_patch": "inert original tests",
            "FAIL_TO_PASS": '["test_failure"]',
            "PASS_TO_PASS": '["test_pass"]',
        }
    ]


def prediction(patch="inert model patch"):
    return {
        "fixture__repo-1": {
            "instance_id": "fixture__repo-1",
            "model_name_or_path": "openai/fixture-model",
            "model_patch": patch,
        }
    }


def test_native_predictions_and_reference_are_not_rewritten_or_interchanged():
    original = prediction()
    before = copy.deepcopy(original)
    selected = grade.predictions_for(rows(), original)
    assert selected == original == before
    assert selected["fixture__repo-1"]["model_patch"] != rows()[0]["patch"]
    control = grade.predictions_for(rows(), gold=True)
    assert control["fixture__repo-1"]["model_patch"] == rows()[0]["patch"]
    assert (
        control["fixture__repo-1"]["model_name_or_path"]
        == "gold-original-reference-control"
    )


@pytest.mark.parametrize("patch", ["", None])
def test_explicit_empty_native_outcome_preserved_without_fabricating_patch(patch):
    assert (
        grade.predictions_for(rows(), prediction(patch))["fixture__repo-1"][
            "model_patch"
        ]
        is patch
    )


def test_missing_prediction_is_not_filled_with_reference_or_empty_answer():
    with pytest.raises(ValueError, match="missing"):
        grade.predictions_for(rows(), {})


@pytest.mark.parametrize(
    "key,value",
    [
        ("instance_id", "different"),
        ("model_name_or_path", "../../escape"),
        ("model_name_or_path", "C:escape"),
        ("model_patch", {"bad": "type"}),
    ],
)
def test_mismatched_or_path_escaping_prediction_rejected(key, value):
    data = prediction()
    data["fixture__repo-1"][key] = value
    with pytest.raises(ValueError):
        grade.predictions_for(rows(), data)


class FakeContainer:
    id = "inert-container"

    def __init__(self, kwargs, *, unsafe=False):
        self.removed = False
        self.attrs = {
            "HostConfig": {
                "Privileged": unsafe,
                "CapDrop": kwargs["cap_drop"],
                "NetworkMode": kwargs["network_mode"],
                "Memory": kwargs["mem_limit"],
                "NanoCpus": kwargs["nano_cpus"],
                "PidsLimit": kwargs["pids_limit"],
                "SecurityOpt": kwargs["security_opt"],
            },
            "Mounts": [],
        }

    def reload(self):
        pass

    def remove(self, *, force):
        assert force
        self.removed = True


class FakeCollection:
    def __init__(self, *, unsafe=False):
        self.calls = []
        self.unsafe = unsafe

    def create(self, **kwargs):
        self.calls.append(kwargs)
        self.container = FakeContainer(kwargs, unsafe=self.unsafe)
        return self.container


def test_original_create_is_wrapped_with_limits_not_scoring_changes():
    original = FakeCollection()
    observed = []
    controlled = grade.RestrictedContainers(original, ["fixture-image"], observed)
    controlled.create(
        image="fixture-image",
        name="inert",
        user="root",
        detach=True,
        command="tail -f /dev/null",
        platform="linux/arm64/v8",
        cap_add=[],
    )
    actual = original.calls[0]
    for key, value in grade.LIMITS.items():
        assert actual[key] == value
    assert observed[0]["host_mounts"] == 0 and observed[0]["network"] == "none"


@pytest.mark.parametrize(
    "extra",
    [
        {"cap_add": ["SYS_ADMIN"]},
        {"privileged": True},
        {"volumes": {"/": "/host"}},
        {"devices": ["/dev/davinci0"]},
        {"image": "unapproved"},
    ],
)
def test_original_grader_cannot_expand_container_privileges(extra):
    original = FakeCollection()
    controlled = grade.RestrictedContainers(original, ["fixture-image"], [])
    kwargs = {"image": "fixture-image", "cap_add": []}
    kwargs.update(extra)
    with pytest.raises(ValueError):
        controlled.create(**kwargs)
    assert not original.calls


def test_failed_new_container_isolation_removes_only_new_owned_container():
    original = FakeCollection(unsafe=True)
    controlled = grade.RestrictedContainers(original, ["fixture-image"], [])
    with pytest.raises(RuntimeError):
        controlled.create(image="fixture-image", cap_add=[])
    assert original.container.removed


def test_gold_plan_never_imports_missing_harness_or_creates_results(tmp_path):
    path = tmp_path / "inert.json"
    path.write_text(json.dumps(rows()), encoding="utf-8")
    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_swebench_original_grade.py"),
            "--source-id",
            "SWE_VERIFIED",
            "--records-json",
            str(path),
            "--harness-root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--docker-url",
            "unix:///inert.sock",
            "--gold-control",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY" and report["gold_control"]
    assert report["original_test_timeout_seconds"] == 1800
    assert report["model_calls"] == 0 and not report["source_checkbox_eligible"]
    assert not output.exists()
