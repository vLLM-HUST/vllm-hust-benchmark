"""HTTP response fixtures only; no original tasks, SQL execution or model calls."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def prepared(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "spider_generation", ROOT / "scripts/run_spider_original_generation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "prepared"
    root.mkdir()
    rows = [
        {
            "original_dev_index": i,
            "db_id": "fixture",
            "question": f"question{i}",
            "schema": {"db_id": "fixture", "tables": ["fixture"]},
        }
        for i in range(3)
    ]
    (root / "inputs.private.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    # Deliberately invalid reference JSON: opening it would fail this test.
    (root / "references.private.jsonl").write_text("REFERENCE_MUST_NOT_BE_OPENED")
    return module, root, tmp_path / "generation"


def test_plan_opens_no_reference_and_makes_no_requests(prepared):
    module, root, _ = prepared

    def forbidden(*a, **k):
        pytest.fail("plan-only must not make requests")

    result = module.run(
        prepared=root, model="fixture", max_tokens=4096, transport=forbidden
    )
    assert result["status"] == "PLAN_ONLY_NO_MODEL_CALLS"
    assert result["original_dev_indices"] == [0, 1, 2]
    assert not result["reference_files_opened"]
    assert result["model_calls"] == 0


@pytest.mark.parametrize(
    "kind", ["valid", "multiline", "request_failure", "input_changed"]
)
def test_fixed_cases_retained_no_sql_repair_or_replacement(prepared, kind):
    module, root, output = prepared
    calls = []

    def transport(url, payload):
        index = len(calls)
        calls.append((url, payload))
        assert "REFERENCE_MUST_NOT_BE_OPENED" not in json.dumps(payload)
        assert f"question{index}" in payload["messages"][0]["content"]
        if index == 0 and kind == "request_failure":
            raise ConnectionError("inert failure")
        if index == 0 and kind == "input_changed":
            path = root / "inputs.private.jsonl"
            path.write_text(path.read_text() + "\n")
        content = (
            "SELECT unchanged"
            if kind != "multiline" or index != 0
            else "```sql\nSELECT unchanged\n```"
        )
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

    result = module.run(
        prepared=root,
        model="fixture",
        max_tokens=4096,
        output=output,
        execute=True,
        transport=transport,
    )
    assert len(calls) == 3
    assert result["model_calls"] == 3
    assert not result["official_scoring_completed"]
    assert not result["source_checkbox_eligible"]
    assert (output / "predictions.private.txt").exists() == (kind == "valid")
    if kind == "valid":
        assert (
            output / "predictions.private.txt"
        ).read_bytes() == b"SELECT unchanged\n" * 3
    else:
        assert (
            result["status"]
            == "ORIGINAL_GENERATION_BLOCKED_RETAINED_NO_TASK_REPLACEMENT"
        )


def test_reference_field_in_inputs_is_rejected(prepared):
    module, root, _ = prepared
    path = root / "inputs.private.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["query"] = "forbidden reference"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="contract"):
        module.requests(root, model="fixture", max_tokens=4096)
