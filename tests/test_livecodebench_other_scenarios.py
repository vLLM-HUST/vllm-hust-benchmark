"""Tiny identity/byte/wire fixtures, not publisher data or native scoring."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "lcb_other", SCRIPTS / "livecodebench_other_scenarios.py"
)
other = importlib.util.module_from_spec(spec)
spec.loader.exec_module(other)
gen_spec = importlib.util.spec_from_file_location(
    "lcb_other_generation", SCRIPTS / "run_livecodebench_original_generation.py"
)
generation = importlib.util.module_from_spec(gen_spec)
gen_spec.loader.exec_module(generation)


def test_fixed_other_assets_match_shared_original_manifest():
    manifest = json.loads(
        (
            ROOT / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text("utf-8")
    )
    files = {row["path"]: row for row in manifest["sources"]["LIVECODEBENCH"]["files"]}
    for asset in other.ASSETS.values():
        assert files[asset["path"]]["sha256"] == asset["sha256"]
        assert files[asset["path"]]["bytes"] == asset["bytes"]
    assert other.ASSETS["codeexecution"]["rows"] == 479
    assert other.ASSETS["testoutputprediction"]["rows"] == 442


def test_native_case_identity_preserves_repeated_question_and_test_dimensions():
    a = SimpleNamespace(question_id=7, id="inert_sample_0")
    b = SimpleNamespace(question_id=7, id="inert_sample_1")
    assert other.case_identity(a, "codeexecution") != other.case_identity(
        b, "codeexecution"
    )
    a, b = (
        SimpleNamespace(question_id="007", test_id=0),
        SimpleNamespace(question_id="007", test_id=1),
    )
    assert json.loads(other.case_identity(a, "testoutputprediction")) == ["007", 0]
    assert other.case_identity(a, "testoutputprediction") != other.case_identity(
        b, "testoutputprediction"
    )


@pytest.mark.parametrize("full,expected", [(False, 3), (True, 4)])
def test_asset_preparation_keeps_all_original_binary_bytes_without_host_decode(
    tmp_path, monkeypatch, full, expected
):
    raw = b"inert byte identity fixture; not an actual Parquet"
    original = tmp_path / "original"
    path = original / "fixture.parquet"
    original.mkdir()
    path.write_bytes(raw)
    monkeypatch.setitem(
        other.ASSETS,
        "codeexecution",
        {
            "path": path.name,
            "bytes": len(raw),
            "rows": 4,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    output = tmp_path / "prepared"
    receipt = other.prepare_asset(original, output, scenario="codeexecution", full=full)
    assert (output / receipt["task_file_name"]).read_bytes() == raw
    assert receipt["selected_case_count"] == expected
    assert receipt["selected_question_ids"] is None
    assert not receipt["source_checkbox_eligible"] and receipt["model_calls"] == 0


@pytest.mark.parametrize("scenario", ["codeexecution", "testoutputprediction"])
def test_native_repeated_question_requests_are_not_deduplicated(tmp_path, scenario):
    rows = []
    for index in range(2):
        task = SimpleNamespace(
            question_id=7 if scenario == "codeexecution" else "007",
            id=f"inert_sample_{index}",
            test_id=index,
        )
        record = {
            "question_id": task.question_id,
            "request_id": other.case_identity(task, scenario),
            "request": {
                "model": "inert-local-model",
                "n": 10,
                "max_tokens": 2000,
                "temperature": 0.2,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "messages": [
                    {"role": "user", "content": "inert official-format fixture"}
                ],
            },
        }
        record.update(
            {"id": task.id} if scenario == "codeexecution" else {"test_id": index}
        )
        rows.append(record)
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(rows))
    assert generation.requests_from(path, scenario=scenario) == rows
    calls = []

    def transport(*args):
        calls.append(args)
        return {
            "choices": [
                {
                    "index": i,
                    "message": {"content": "inert literal"},
                    "finish_reason": "stop",
                }
                for i in range(10)
            ]
        }

    output = tmp_path / "responses"
    receipt = generation.run(
        path,
        output,
        endpoint="http://localhost:19466/v1",
        scenario=scenario,
        transport=transport,
    )
    assert len(calls) == 2 and receipt["completed_request_ids"] == [
        r["request_id"] for r in rows
    ]
    assert receipt["completed_question_ids"] == [rows[0]["question_id"]] * 2
    saved = json.loads((output / "1.request-response.private.json").read_text())
    assert saved["question_id"] == rows[1]["question_id"]
