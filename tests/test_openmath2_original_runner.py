"""Small artificial corpus records/API responses; not original execution evidence."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "openmath_runner", ROOT / "scripts/run_openmath2_original.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "openmath_saved", ROOT / "scripts/verify_openmath2_original_saved.py"
)
saved_verifier = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(saved_verifier)


def fixture_rows(count=3):
    return [
        SimpleNamespace(
            source={
                "source_id": "OPENMATH2",
                "file": "original.parquet",
                "file_sha256": "fixture-hash",
            },
            task={
                "task_id": f"fixture-hash:{index}",
                "input": {"problem": f"fixture problem{index}"},
            },
            outcome={
                "generated_solution": "fixture teacher only",
                "expected_answer": "",
            },
            replay={
                "source_row_index": index,
                "annotations": {"problem_source": "math"},
            },
        )
        for index in range(count)
    ]


def good_transport(url, payload):
    if url.endswith("/tokenize"):
        return {"count": 7}
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": "fixture answer"}}
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2},
    }


def execute(tmp_path, rows=None, **overrides):
    kwargs = {
        "output": tmp_path / "output",
        "endpoint": "http://127.0.0.1:19466/v1",
        "model": "fixture-model",
        "max_tokens": 8,
        "context": 20,
        "expected_count": 3,
        "transport": good_transport,
    }
    kwargs.update(overrides)
    return runner.execute(fixture_rows() if rows is None else rows, **kwargs)


def test_original_problem_only_no_teacher_leak(tmp_path):
    calls = []

    def transport(url, payload):
        calls.append((url, payload))
        return good_transport(url, payload)

    receipt = execute(tmp_path, transport=transport)
    assert receipt["completed_count"] == 3 and receipt["generation_attempts"] == 3
    assert (
        receipt["source_checkbox_eligible"] is False
        and receipt["quality_scored"] is False
    )
    assert all("fixture teacher only" not in json.dumps(body) for _, body in calls)
    assert calls[1][1]["messages"] == [{"role": "user", "content": "fixture problem0"}]
    saved = json.loads((tmp_path / "output/request-000000.private.json").read_text())
    assert saved["original_reference"]["generated_solution"] == "fixture teacher only"


def test_opencode_native_profile_keeps_original_ids_and_never_executes_tests(tmp_path):
    rows = fixture_rows()
    for index, row in enumerate(rows):
        row.source["source_id"] = "OPENCODE"
        row.task = {
            "task_id": f"original-id-{index}",
            "input": {"input": f"original coding question{index}"},
        }
        row.outcome = {"output": "teacher code", "unit_tests": "never eval this text"}
        row.replay["annotations"] = {"id": row.task["task_id"]}
    calls = []

    def transport(url, payload):
        calls.append(payload)
        return good_transport(url, payload)

    receipt = execute(tmp_path, rows, source_id="OPENCODE", transport=transport)
    assert receipt["source_id"] == "OPENCODE"
    assert receipt["completed_count"] == 3
    assert all("teacher code" not in json.dumps(body) for body in calls)
    assert all("never eval" not in json.dumps(body) for body in calls)
    audit = saved_verifier.verify(
        tmp_path / "output", rows, expected_count=3, source_id="OPENCODE"
    )
    assert audit["source_id"] == "OPENCODE"
    assert audit["source_checkbox_eligible"] is False


def test_unreviewed_corpus_profile_rejected_before_model_calls(tmp_path):
    with pytest.raises(ValueError, match="reviewed original corpus"):
        execute(tmp_path, source_id="UNKNOWN")
    assert not (tmp_path / "output").exists()


def test_full_mode_streams_without_preloading_and_keeps_original_order(tmp_path):
    emitted = []
    calls = []

    def rows():
        for index, row in enumerate(fixture_rows(4)):
            if index:
                assert len(calls) == 2 * index
            emitted.append(index)
            yield row

    def transport(url, payload):
        calls.append(url)
        return good_transport(url, payload)

    result = execute(tmp_path, rows(), expected_count=None, transport=transport)
    assert emitted == [0, 1, 2, 3] and result["completed_count"] == 4


def test_failed_request_not_retried_or_replaced(tmp_path):
    attempts = []

    def transport(url, payload):
        if url.endswith("/chat/completions"):
            attempts.append(payload)
            raise RuntimeError("fixture failure")
        return good_transport(url, payload)

    with pytest.raises(RuntimeError):
        execute(tmp_path, transport=transport)
    assert len(attempts) == 1
    receipt = json.loads((tmp_path / "output/receipt-private.json").read_text())
    assert (
        receipt["status"] == "FAILED_NOT_COMPLETE" and receipt["completed_count"] == 0
    )


def test_length_result_retained_not_quality_pass(tmp_path):
    def transport(url, payload):
        value = good_transport(url, payload)
        if "choices" in value:
            value["choices"][0]["finish_reason"] = "length"
        return value

    receipt = execute(tmp_path, transport=transport)
    assert receipt["length_count"] == 3 and not receipt["quality_scored"]


def test_context_overflow_stops_before_generation(tmp_path):
    with pytest.raises(ValueError, match="context"):
        execute(tmp_path, context=9)
    receipt = json.loads((tmp_path / "output/receipt-private.json").read_text())
    assert receipt["generation_attempts"] == 0


def test_preselected_count_mismatch_has_no_outputs_or_requests(tmp_path):
    with pytest.raises(ValueError, match="count"):
        execute(tmp_path, fixture_rows(2))
    assert not (tmp_path / "output").exists()


def test_duplicate_original_identity_not_regenerated(tmp_path):
    rows = fixture_rows()
    rows[1] = rows[0]
    with pytest.raises(ValueError):
        execute(tmp_path, rows)
    assert not (tmp_path / "output").exists()


def test_saved_source_reference_and_response_audit_needs_no_models(tmp_path):
    execute(tmp_path)
    result = saved_verifier.verify(
        tmp_path / "output", fixture_rows(), expected_count=3
    )
    assert result["original_cases"] == 3 and result["model_calls"] == 0
    assert len(result["saved_case_digests"]) == 3
    assert (
        result["source_checkbox_eligible"] is False
        and result["quality_scored"] is False
    )


@pytest.mark.parametrize("field", ["problem", "teacher", "usage", "terminus"])
def test_saved_mutation_cannot_become_a_pass(tmp_path, field):
    execute(tmp_path)
    path = tmp_path / "output/request-000000.private.json"
    case = json.loads(path.read_text())
    if field == "problem":
        case["request"]["messages"][0]["content"] += " changed"
    elif field == "teacher":
        case["original_reference"]["generated_solution"] = "changed"
    elif field == "usage":
        case["response"]["usage"]["prompt_tokens"] += 1
    else:
        case["response"]["choices"][0]["finish_reason"] = "length"
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError):
        saved_verifier.verify(tmp_path / "output", fixture_rows(), expected_count=3)


def test_started_receipt_is_not_completed(tmp_path):
    execute(tmp_path)
    path = tmp_path / "output/receipt-private.json"
    receipt = json.loads(path.read_text())
    receipt["status"] = "STARTED"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="complete"):
        saved_verifier.verify(tmp_path / "output", fixture_rows(), expected_count=3)
