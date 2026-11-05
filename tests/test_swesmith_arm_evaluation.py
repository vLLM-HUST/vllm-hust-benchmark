import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "swesmith_arm_eval",
    Path(__file__).resolve().parents[1] / "scripts/run_swesmith_arm_evaluation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def arguments():
    return {
        "image": "task-image",
        "name": "swesmith.eval.workload-native-fixture.task",
        "platform": "linux/x86_64",
        "mem_limit": "10g",
    }


def test_binding_preserves_memory_and_disables_network_without_host_access():
    result = MODULE.deployment_kwargs(
        arguments(), "task-image", "workload-native-fixture"
    )
    assert result["mem_limit"] == "10g"
    assert result["platform"] == "linux/arm64/v8"
    assert result["network_disabled"] is True
    assert result["cap_drop"] == ["ALL"]
    assert result["security_opt"] == ["no-new-privileges"]


@pytest.mark.parametrize(
    "change",
    [
        {"privileged": True},
        {"devices": ["/dev/davinci0"]},
        {"volumes": {"/": {"bind": "/host"}}},
        {"network_mode": "host"},
        {"name": "unrelated-container"},
        {"image": "different-image"},
    ],
)
def test_binding_rejects_privileges_and_unrelated_targets(change):
    with pytest.raises(ValueError):
        MODULE.deployment_kwargs(
            {**arguments(), **change}, "task-image", "workload-native-fixture"
        )


def test_author_archive_identity_required(tmp_path):
    archive = tmp_path / "wrong.tar.gz"
    archive.write_bytes(b"not the original publisher archive")
    with pytest.raises(ValueError, match="original author archive"):
        MODULE.verify_author(archive)


def test_predictions_reject_external_task_paths_without_rewriting_input(tmp_path):
    identity = next(iter(MODULE.COMMITS))
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps({identity: {"instance_id": identity, "model_patch": ""}})
    )
    original = path.read_bytes()
    MODULE.validate_predictions(path)
    assert path.read_bytes() == original
    path.write_text(
        json.dumps(
            {"../../outside": {"instance_id": "../../outside", "model_patch": ""}}
        )
    )
    with pytest.raises(ValueError, match="frozen task IDs"):
        MODULE.validate_predictions(path)


def test_native_null_submission_is_preserved_not_fabricated_as_empty_patch(tmp_path):
    identity = next(iter(MODULE.COMMITS))
    path = tmp_path / "native-predictions.json"
    path.write_text(
        json.dumps({identity: {"instance_id": identity, "model_patch": None}})
    )
    original = path.read_bytes()
    MODULE.validate_predictions(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("patch_field", [{}, {"model_patch": 0}, {"model_patch": []}])
def test_missing_or_invalid_patch_is_not_a_native_null_submission(
    tmp_path, patch_field
):
    identity = next(iter(MODULE.COMMITS))
    path = tmp_path / "invalid-predictions.json"
    path.write_text(json.dumps({identity: {"instance_id": identity, **patch_field}}))
    with pytest.raises(ValueError, match="original string/null"):
        MODULE.validate_predictions(path)
