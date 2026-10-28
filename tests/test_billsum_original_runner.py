"""Model-free contracts; do not confuse fixtures with native model evidence."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "billsum_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_billsum_original.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate original source/profile key")
        result[key] = value
    return result


def test_publisher_data_and_model_not_bound_to_other_summarization_sources():
    pins = json.loads(runner.PINS.read_text("utf-8"), object_pairs_hook=unique_object)
    assert pins["source_id"] == "BILLSUM"
    assert pins["dataset"]["repo_id"] == "FiscalNote/billsum"
    assert pins["model"]["repo_id"] == "google/pegasus-billsum"
    assert {
        item["split"]: item["rows"]
        for item in pins["dataset"]["files"]
        if item.get("role") == "data"
    } == {"train": 18949, "test": 3269, "ca_test": 1237}
    assert pins["generation_config"]["length_penalty"] == 0.6
    assert pins["generation_config"]["num_beams"] == 8
    assert len(pins["model"]["files"]) == 7
    manifests = json.loads(
        (
            runner.PINS.parents[2]
            / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text("utf-8"),
        object_pairs_hook=unique_object,
    )
    assert manifests["sources"]["BILLSUM"]["revision"] == pins["dataset"]["revision"]


def test_saved_response_requires_exact_original_bill_reference_title_and_index():
    row = {
        "text": " Original bill. ",
        "summary": "Reference",
        "title": "Original title",
    }
    case = {
        "original_index": 2,
        "original_row": row,
        "original_text_utf8_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
        "native_result": [{"summary_text": "Prediction"}],
    }
    assert runner.canonical_row(row) == row
    assert runner.validate_saved_case(case, row, 2) == "Prediction"
    for field in row:
        changed = dict(row, **{field: "Changed"})
        with pytest.raises(ValueError, match="original bill"):
            runner.validate_saved_case(case, changed, 2)
    with pytest.raises(ValueError, match="original bill"):
        runner.validate_saved_case(case, row, 3)
    with pytest.raises(ValueError, match="original bill"):
        runner.validate_saved_case(dict(case, native_result=[None]), row, 2)


def test_saved_run_requires_original_model_generation_and_input_policy():
    pins = json.loads(runner.PINS.read_text("utf-8"))
    receipt = {
        "source_id": "BILLSUM",
        "status": "ORIGINAL_BILLSUM_GENERATION_COMPLETED",
        "model_revision": pins["model"]["revision"],
        "dataset_revision": pins["dataset"]["revision"],
        "input_policy": runner.INPUT_POLICY,
        "native_generation_config": deepcopy(pins["generation_config"]),
        "versions": {"transformers": pins["framework_version"]},
        "device": "npu:1",
        "model_parameters_verified_on_npu": True,
        "native_encoder_maximum_tokens": 1024,
    }
    runner.validate_saved_run(receipt, pins)
    for field, value in [
        ("input_policy", "include gold"),
        ("source_id", "XSUM"),
        ("device", "cpu"),
    ]:
        with pytest.raises(ValueError, match="model/protocol"):
            runner.validate_saved_run(dict(receipt, **{field: value}), pins)
    receipt["native_generation_config"]["length_penalty"] = 0.8
    with pytest.raises(ValueError, match="model/protocol"):
        runner.validate_saved_run(receipt, pins)
