"""Software fixture strings only; no code_context or generated code executes."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "ds_prepare", ROOT / "scripts/prepare_ds1000_official_samples.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TASK_COUNT", 3)
    generation = tmp_path / "historical"
    generation.mkdir()
    profile = {
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 1024,
            "num_beams": 1,
        },
        "stop_words": ["</code>", "# SOLUTION END"],
    }
    rows, samples = [], []
    for i in range(3):
        row = {
            "prompt": f"fixture question {i}",
            "metadata": {"problem_id": i, "library": "Pandas"},
            "code_context": "raise RuntimeError('reference must not execute')",
        }
        rows.append(row)
        response = {
            "choices": [
                {
                    "text": "raise RuntimeError('generated must not execute')\n</code>trailing text",
                    "finish_reason": "length" if i == 1 else "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        summary, code = runner.response_summary(response, 10, profile["stop_words"])
        samples.append({"row_index": i, "original_problem_id": i, **summary})
        (generation / f"{i}.private.json").write_text(
            json.dumps(
                {
                    "request": runner.request_body(row["prompt"], "fixture-model"),
                    "response": response,
                }
            ),
            encoding="utf-8",
        )
        (generation / f"{i}.postprocessed.private.json").write_text(
            json.dumps({"id": i, "code": code, "metadata": row["metadata"]}),
            encoding="utf-8",
        )
    receipt = {
        "source_id": "DS1000",
        "code_revision": runner.CODE_REVISION,
        "selected_original_row_indices": [0, 1, 2],
        "publisher_literal_config": profile,
        "status": "ORIGINAL_COMPLETION_PROFILE_NOT_FUNCTIONALLY_SCORED",
        "model": "fixture-model",
        "samples": samples,
    }
    (generation / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return generation, rows, profile


def test_author_generation_postprocessing_reused_and_length_preserved(
    tmp_path, monkeypatch
):
    generation, rows, profile = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    receipt = runner.prepare(generation, rows=rows, profile=profile, output=out)
    answers = [
        json.loads(line)
        for line in (out / "answers-private.jsonl").read_text().splitlines()
    ]
    assert answers[0]["code"] == "raise RuntimeError('generated must not execute')\n"
    assert receipt["historical_files"][1]["finish_reason"] == "length"
    assert [
        json.loads(line)
        for line in (out / "original-dataset-private.jsonl").read_text().splitlines()
    ] == rows
    assert not receipt["code_context_executed"]
    assert not receipt["official_functional_score_complete"]
    assert not receipt["prior_per_file_hash_binding_available"]
    assert not receipt["source_checkbox_eligible"]


def test_silent_candidate_repair_is_rejected(tmp_path, monkeypatch):
    generation, rows, profile = fixture(tmp_path, monkeypatch)
    path = generation / "0.postprocessed.private.json"
    saved = json.loads(path.read_text())
    saved["code"] = "pass"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="no silent repair"):
        runner.prepare(generation, rows=rows, profile=profile, output=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_changed_original_request_is_rejected(tmp_path, monkeypatch):
    generation, rows, profile = fixture(tmp_path, monkeypatch)
    path = generation / "0.private.json"
    saved = json.loads(path.read_text())
    saved["request"]["stop"] = ["</code>"]
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration mismatch"):
        runner.prepare(generation, rows=rows, profile=profile, output=tmp_path / "out")
