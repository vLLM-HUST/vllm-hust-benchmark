"""CPU protocol checks only; no source download, model, or scoring calls."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "schema_runner",
    Path(__file__).parents[1] / "scripts/run_jsonschemabench_original_schema.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_research_asset_handoff_does_not_complete_original_official_bench():
    root = Path(__file__).parents[1]
    ledger = json.loads(
        (root / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["JSONSCHEMABENCH"]
    handoff = json.loads(
        (root / row["research_exact_artifact_handoff_receipt"]).read_text("utf-8")
    )
    workflow = json.loads(
        (root / row["official_workflow_readiness_receipt"]).read_text("utf-8")
    )
    # Old three-schema handoff remains insufficient; the later native closure
    # is independent evidence and must not rewrite the frozen old receipt.
    closure = workflow.get("native_delivery_closure_20261026")
    assert row["checked"] == bool(closure and closure["source_checkbox_eligible"])
    if row["checked"]:
        assert len(closure["protocols"]) == 11
        assert closure["official_formatter_bench_and_evaluator_used"]
        assert closure["original_messages_schemas_and_saved_outputs_equal"]
    assert handoff["public_prepare_resolve_rows"] == 3
    assert handoff["research_source_index_import_rows"] == 3
    assert handoff["research_direct_prepare_resolve_rows"] == 3
    assert handoff["whole_original_schema_and_source_identity_equal"]
    assert not handoff["schema_keywords_added_removed_or_rewritten"]
    assert not handoff["native_official_bench_executed"]
    assert workflow["original_entry"] == "core.bench.bench"
    assert workflow["original_dataset_repository"] == "epfl-dlab/JSONSchemaBench"
    assert workflow["original_named_partitions_excluding_default"] == 10
    assert not workflow["new_original_dataset_revision_verified"]
    assert not workflow["source_completion_checked"]
    assert workflow["public_metadata_fetch"]["downloaded_dataset_files"] == 0


@pytest.mark.parametrize(
    "origin",
    [
        "https://example.com",
        "http://user:secret@localhost",  # pragma: allowlist secret - invented rejected credential fixture
        "http://localhost/v1",
        "http://localhost/?key=x",
    ],
)
def test_external_and_credential_endpoints_rejected(origin):
    with pytest.raises(ValueError):
        runner.endpoint(origin)


def test_original_schema_not_adapted_and_budget_explicit():
    schema = {"type": "object", "properties": {"fixture": {"maxLength": 3}}}
    before = deepcopy(schema)
    body = runner.request_body(
        schema, [{"role": "user", "content": "fixture"}], "fixture-model"
    )
    assert body["response_format"]["json_schema"]["schema"] == before
    assert schema == before
    assert "required" not in schema and "additionalProperties" not in schema
    assert body["max_tokens"] == 1024 and body["stream"] is False
    assert runner.endpoint("http://127.0.0.1:19466/") == "http://127.0.0.1:19466"


def test_nested_reference_detection_does_not_fetch():
    assert runner.has_ref({"items": [{"$ref": "https://invalid.example"}]})
    assert not runner.has_ref({"properties": {"fixture": {"type": "string"}}})


def test_receipt_keeps_execution_and_conformance_separate():
    root = Path(__file__).parents[1]
    receipt = json.loads(
        (
            root
            / "docs/workload-data/validation/jsonschemabench-original-schema-npu-20261023.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["unique_tasks"] == 3
    assert receipt["completed_samples"] == list(runner.SCHEMAS)
    assert all(row["finish_reason"] == "stop" for row in receipt["samples"])
    assert sum(row["original_validator_valid"] for row in receipt["samples"]) == 1
    assert receipt["source_checkbox_eligible"] is False
    assert receipt["full_suite_claim"] is False
    assert receipt["schema_adaptation"] is False
