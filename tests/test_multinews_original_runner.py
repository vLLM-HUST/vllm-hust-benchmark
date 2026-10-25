"""Model-free Multi-News protocol fixtures; real evidence is separately recorded."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "multinews_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_multinews_original.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate frozen profile key")
        result[key] = value
    return result


def test_original_pegasus_and_parallel_asset_profile():
    pins = json.loads(runner.PINS.read_text("utf-8"), object_pairs_hook=unique_object)
    assert pins["source_id"] == "MULTINEWS"
    assert pins["dataset"]["repo_id"] == "alexfabbri/multi_news"
    assert pins["model"]["repo_id"] == "google/pegasus-multi_news"
    assert len(pins["dataset"]["files"]) == 8
    assert len(pins["model"]["files"]) == 7
    sources = [item for item in pins["dataset"]["files"] if item["role"] == "data"]
    assert {item["split"]: item["rows"] for item in sources} == {
        "train": 44972,
        "validation": 5622,
        "test": 5622,
    }
    assert all(item["paired_target"].endswith(".tgt") for item in sources)
    assert pins["generation_config"]["max_length"] == 256
    assert pins["generation_config"]["min_length"] == 32
    assert pins["generation_config"]["num_beams"] == 8
    assert pins["generation_config"]["length_penalty"] == 0.8
    assert pins["generation_config"]["no_repeat_ngram_size"] == 0
    assert pins["protocol"]["encoder_maximum_tokens"] == 1024


def test_summary_utilities_do_not_rebind_multinews_to_xsum():
    common = runner.shared_utilities()
    assert common.selected_indices(5622) == [0, 1, 2]
    assert runner.canonical_row(
        {"document": "Bundle ||||| Next", "summary": "Gold", "target_file": "original"}
    ) == {"document": "Bundle ||||| Next", "summary": "Gold"}


def test_saved_responses_require_original_model_window_and_input_policy():
    pins = json.loads(runner.PINS.read_text("utf-8"))
    receipt = {
        "source_id": "MULTINEWS",
        "status": "ORIGINAL_MULTINEWS_GENERATION_COMPLETED",
        "dataset_revision": pins["dataset"]["revision"],
        "model_revision": pins["model"]["revision"],
        "versions": {"transformers": pins["framework_version"]},
        "device": "npu:1",
        "native_generation_config": deepcopy(pins["generation_config"]),
        "native_encoder_maximum_tokens": 1024,
        "model_parameters_verified_on_npu": True,
        "input_policy": runner.INPUT_POLICY,
    }
    runner.validate_saved_run(receipt, pins)
    changed = deepcopy(receipt)
    changed["input_policy"] = "Modified question or reduced budget"
    with pytest.raises(ValueError, match="model/protocol"):
        runner.validate_saved_run(changed, pins)
    changed = deepcopy(receipt)
    changed["native_generation_config"]["num_beams"] = 1
    with pytest.raises(ValueError, match="model/protocol"):
        runner.validate_saved_run(changed, pins)
