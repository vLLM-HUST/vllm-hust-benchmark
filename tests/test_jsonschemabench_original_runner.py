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
