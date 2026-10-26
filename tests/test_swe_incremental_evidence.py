import importlib.util
import json
from pathlib import Path
import shutil

import pytest

FOLDER = (
    Path(__file__).resolve().parents[1] / "reports/swe-native018-incremental-20261026"
)


def verifier():
    spec = importlib.util.spec_from_file_location(
        "swe_incremental_verifier", FOLDER / "verify.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify


def test_public_observation_verifies():
    result = verifier()(FOLDER)
    assert result["status"] == "passed"
    assert result["official_result"] is False
    assert result["patch_application_attested"] is False


@pytest.mark.parametrize("field", ["application", "grade", "scope", "calls"])
def test_public_observation_rejects_false_claims(tmp_path, field):
    target = tmp_path / "packet"
    shutil.copytree(FOLDER, target)
    path = target / "summary.json"
    value = json.loads(path.read_text())
    if field == "application":
        value["grading"]["patch_application_attested"] = True
    elif field == "grade":
        value["grading"]["official_result"] = True
    elif field == "scope":
        value["full_dataset_acceptance"] = True
    else:
        value["agent"]["model_calls"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        verifier()(target)


@pytest.mark.parametrize("field", ["raw_count", "dedup", "grade", "release", "patch"])
def test_cached_review_rejects_false_claims(tmp_path, field):
    target = tmp_path / "packet"
    shutil.copytree(FOLDER, target)
    path = target / (
        "cached-release-receipt.json"
        if field == "release"
        else "cached-independent-review.json"
    )
    value = json.loads(path.read_text())
    if field == "raw_count":
        value["official_grading"]["mocha_raw_counts"]["tests"] = 173
    elif field == "dedup":
        value["official_grading"]["duplicate_names_collapsed"] = 0
    elif field == "grade":
        value["official_grading"]["resolved"] = True
    elif field == "release":
        value["all_devices_released"] = False
    else:
        value["agent"]["patch_sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        verifier()(target)
