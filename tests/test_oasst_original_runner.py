"""Runner contract checks only, no real requests or publisher payloads."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PATH = (
    Path(__file__).resolve().parents[1] / "scripts/run_oasst_original_continuation.py"
)
SPEC = importlib.util.spec_from_file_location("oasst_original_runner", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_corpus_delivery_is_not_quality_or_live_episode_claim():
    root = PATH.parents[1]
    docs = root / "docs/workload-data"
    ledger = json.loads((docs / "dataset-validation-status.json").read_text("utf-8"))
    item = ledger["delivery_inventory"]["OASST1"]
    receipt = json.loads((root / item["delivery_receipt"]).read_text("utf-8"))
    assert item["checked"] and item["status"] == "AVAILABLE"
    assert receipt["available_original_rows"] == 84437 + 4401
    assert receipt["verified_unique_continuation_points"] == 6
    assert not receipt["quality_scored"]
    assert not receipt["live_episode_validated"]
    assert not receipt["full_source_inference_validated"]
    for name in receipt["installed_ci_handoff_receipts"]:
        evidence = json.loads((docs / "validation" / name).read_text("utf-8"))
        assert evidence["new_model_calls"] == 0


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/v1",
        "http://example.com/v1",
        "http://user:password@localhost/v1",  # pragma: allowlist secret - invented rejected URL fixture
        "http://localhost/v1?x=1",
        "http://localhost/v1#fragment",
        "http://localhost/other",
    ],
)
def test_remote_or_credential_endpoints_rejected(url):
    with pytest.raises(ValueError):
        RUNNER.validate_endpoint(url)


def test_local_endpoint_and_explicit_three_original_targets():
    assert RUNNER.validate_endpoint("http://127.0.0.1:19466/v1")
    assert len(set(RUNNER.TARGETS)) == 3


def test_explicit_original_targets_are_not_replaced_filtered_or_deduplicated():
    assert RUNNER.selected_targets(None) == RUNNER.TARGETS
    assert RUNNER.selected_targets(["original-assistant-id"]) == [
        "original-assistant-id"
    ]
    assert RUNNER.selected_targets(["second", "first"]) == ["second", "first"]
    for targets in [[], [""], [" "], ["same", "same"]]:
        with pytest.raises(ValueError):
            RUNNER.selected_targets(targets)


def test_train_never_silently_uses_validation_ids():
    with pytest.raises(ValueError, match="explicit"):
        RUNNER.selected_targets(None, "train")
    assert RUNNER.selected_targets(["train-original-id"], "train") == [
        "train-original-id"
    ]
    assert RUNNER.SPLITS["train"][1] == 84437
    assert RUNNER.SPLITS["validation"] == (RUNNER.DATA_SHA, 4401)


def test_help_needs_no_optional_data_sdk():
    result = subprocess.run(
        [sys.executable, "-I", str(PATH), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--data-file" in result.stdout
