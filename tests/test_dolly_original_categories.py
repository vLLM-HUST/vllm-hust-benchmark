"""Software contracts, not real source or model execution evidence."""

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_dolly_original_categories.py"
spec = importlib.util.spec_from_file_location("dolly_categories", PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_category_selection_uses_first_original_rows_and_full_is_explicit():
    rows = [{"category": category} for category in ["a", "b", "a", "b", "a", "b", "a"]]
    indices, groups = runner.selected_indices(rows)
    assert indices == [0, 1, 2, 3, 4, 5]
    assert groups == {"a": [0, 2, 4], "b": [1, 3, 5]}
    assert runner.selected_indices(rows, full=True)[0] == list(range(7))
    assert runner.selected_indices([{"category": "small"}])[0] == [0]
    with pytest.raises(ValueError, match="category"):
        runner.selected_indices([{"category": ""}])


def test_reuse_requires_matching_protocol_and_complete_existing_integrity():
    upstream = SimpleNamespace(
        REVISION="revision", DATA_SHA="dataset", CONSTS_SHA="constants"
    )
    receipt = {
        "source_id": "DOLLY",
        "revision": "revision",
        "dataset_sha256": "dataset",
        "publisher_constants_sha256": "constants",
        "model": "model",
        "generation_config": {"temperature": 0, "max_tokens": 2048},
        "status": "ORIGINAL_FIELDS_GENERATED_AND_INTEGRITY_VERIFIED_NO_QUALITY_SCORE",
        "selected_row_indices": [0],
        "samples": [
            {
                "row_index": 0,
                "full_prompt_tokens": 10,
                "preflight_prompt_tokens": 10,
                "finish_reasons": ["stop"],
                "input_fields_roundtrip": True,
                "response_validation": {"status": "RESPONSE_INTEGRITY_VERIFIED"},
            }
        ],
    }
    assert runner.reusable_indices(receipt, runner=upstream, model="model") == {0}
    with pytest.raises(ValueError, match="same source"):
        runner.reusable_indices(receipt, runner=upstream, model="another-model")
    receipt["samples"][0]["full_prompt_tokens"] = 9
    with pytest.raises(ValueError, match="incomplete"):
        runner.reusable_indices(receipt, runner=upstream, model="model")


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/v1",
        "http://user:fake@localhost/v1",  # pragma: allowlist secret - invalid URL fixture
        "http://localhost/v1?x=1",
        "http://localhost/other",
    ],
)
def test_external_or_credential_endpoint_is_rejected(url):
    with pytest.raises(ValueError, match="local"):
        runner.validate_endpoint(url)


def test_original_shared_runner_is_reused_and_private_paths_are_external(tmp_path):
    original = runner.original_runner()
    assert original.DATA_SHA
    assert (
        original.generation_prefix(
            {"instruction": "input", "context": "", "response": "REFERENCE_SENTINEL"},
            SimpleNamespace(PROMPT_NO_INPUT_FORMAT="{instruction}--{response}"),
        )
        == "input--"
    )
    assert runner.external(tmp_path) == tmp_path.resolve()
    with pytest.raises(ValueError, match="outside"):
        runner.external(runner.ROOT / "outputs")


def test_redacted_category_receipt_preserves_request_corpus_boundaries():
    receipt = json.loads(
        (
            runner.ROOT
            / "docs/workload-data/validation/dolly-original-categories-20261024.json"
        ).read_text("utf-8")
    )
    groups = receipt["selected_indices_by_category"]
    assert len(groups) == 8
    assert all(len(indices) == 3 for indices in groups.values())
    assert len({index for indices in groups.values() for index in indices}) == 24
    assert receipt["new_unique_original_requests"] == len(receipt["new_samples"]) == 21
    assert receipt["reused_unique_original_requests"] == 3
    assert (
        sum(receipt["category_population_counts"].values())
        == receipt["original_dataset_rows"]
        == 15011
    )
    assert all(
        sample["finish_reasons"] == ["stop"] for sample in receipt["new_samples"]
    )
    assert receipt["quality_score"] == "NOT_APPLICABLE_REQUEST_CORPUS"
    assert not receipt["full_dataset_validated"]
    assert not receipt["source_checkbox_eligible"]
    assert not receipt["raw_data_published"]
    assert (
        hashlib.sha256(PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == receipt["public_runner_sha256"]
    )
    ledger = json.loads(
        (runner.ROOT / "docs/workload-data/dataset-validation-status.json").read_text(
            "utf-8"
        )
    )
    item = ledger["delivery_inventory"]["DOLLY"]
    assert item["status"] == "AVAILABLE" and item["checked"]
    delivery = json.loads((runner.ROOT / item["delivery_receipt"]).read_text("utf-8"))
    assert delivery["delivery_kind"] == "request_corpus"
    assert delivery["source_checkbox_eligible"] is True
    assert delivery["fixed_ci_provider_public_entry_recheck"]["model_calls"] == 0
    assert not delivery["full_dataset_validated"]
    assert not delivery["quality_claim"]
