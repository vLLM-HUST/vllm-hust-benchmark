"""Tiny software fixtures; never execute generated code or author sanitizer."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "bcb_prepare", ROOT / "scripts/prepare_bigcodebench_official_samples.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TASK_COUNT", 3)
    generation = tmp_path / "historical"
    generation.mkdir()
    formatter = NS(
        make_raw_chat_prompt=lambda **kw: "fixture prefix " + kw["task_prompt"]
    )
    rows, samples = [], []
    for i in range(3):
        row = {
            "task_id": f"BigCodeBench/{i}",
            "instruct_prompt": f"fixture question {i}",
            "complete_prompt": "fixture source",
            "canonical_solution": "private fixture reference",
            "test": "private fixture test",
            "entry_point": "fixture",
        }
        rows.append(row)
        response = {
            "choices": [
                {
                    "message": {
                        "content": "```python\nraise RuntimeError('must not execute')\n```"
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        sample = runner.response_summary(response, 10)
        samples.append({"row_index": i, "task_id": row["task_id"], **sample})
        (generation / f"{i}.private.json").write_text(
            json.dumps(
                {
                    "task_id": row["task_id"],
                    "messages": [
                        {
                            "role": "user",
                            "content": runner.format_prompt(
                                formatter, row["instruct_prompt"]
                            ),
                        }
                    ],
                    "response": response,
                }
            ),
            encoding="utf-8",
        )
    receipt = {
        "source_id": "BIGCODEBENCH",
        "code_revision": runner.CODE_REVISION,
        "split": "instruct",
        "selected_original_row_indices": [0, 1, 2],
        "status": "ORIGINAL_GENERATION_PROFILE_ATTEMPTED_NOT_FUNCTIONALLY_SCORED",
        "max_completion_tokens": 1280,
        "temperature": 0,
        "top_p": 0.95,
        "samples": samples,
    }
    (generation / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return generation, rows, formatter


def test_raw_full_solutions_not_mislabeled_as_completions(tmp_path, monkeypatch):
    generation, rows, formatter = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    receipt = runner.prepare(generation, rows=rows, formatter=formatter, output=out)
    solutions = [
        json.loads(line)
        for line in (out / "raw-solutions-private.jsonl").read_text().splitlines()
    ]
    assert all(
        "solution" in sample and "completion" not in sample for sample in solutions
    )
    assert solutions[0]["solution"].startswith("```python\n")
    assert [
        json.loads(line)
        for line in (out / "original-dataset-private.jsonl").read_text().splitlines()
    ] == rows
    assert not receipt["prior_per_file_cryptographic_binding_available"]
    assert not receipt["official_sanitizer_executed"]
    assert not receipt["functional_scoring_executed"]
    assert not receipt["source_checkbox_eligible"]


def test_sparse_original_ids_and_filenames_preserved(tmp_path, monkeypatch):
    generation, rows, formatter = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "TASK_COUNT", 12)
    rows = [dict(rows[0], task_id=f"BigCodeBench/{i}") for i in range(12)]
    receipt = json.loads((generation / "receipt.json").read_text())
    receipt["selected_original_row_indices"] = [5, 8, 11]
    for previous, selected, sample in zip(range(3), [5, 8, 11], receipt["samples"]):
        saved = json.loads((generation / f"{previous}.private.json").read_text())
        saved["task_id"] = f"BigCodeBench/{selected}"
        saved["messages"] = [
            {
                "role": "user",
                "content": runner.format_prompt(
                    formatter, rows[selected]["instruct_prompt"]
                ),
            }
        ]
        (generation / f"{selected}.private.json").write_text(json.dumps(saved))
        sample.update(row_index=selected, task_id=saved["task_id"])
    (generation / "receipt.json").write_text(json.dumps(receipt))
    out = tmp_path / "out"
    result = runner.prepare(generation, rows=rows, formatter=formatter, output=out)
    assert result["selected_task_ids"] == [
        "BigCodeBench/5",
        "BigCodeBench/8",
        "BigCodeBench/11",
    ]
    assert result["selected_original_row_indices"] == [5, 8, 11]
    assert result["original_dataset_rows_preserved"] == 12
    assert result["source_checkbox_eligible"] is False


def test_changed_original_prompt_cannot_be_reused(tmp_path, monkeypatch):
    generation, rows, formatter = fixture(tmp_path, monkeypatch)
    rows[0]["instruct_prompt"] += "changed"
    with pytest.raises(ValueError, match="prompt mismatch"):
        runner.prepare(
            generation, rows=rows, formatter=formatter, output=tmp_path / "out"
        )


def test_usage_or_task_mismatch_rejected_before_writing(tmp_path, monkeypatch):
    generation, rows, formatter = fixture(tmp_path, monkeypatch)
    path = generation / "0.private.json"
    saved = json.loads(path.read_text())
    saved["response"]["usage"]["completion_tokens"] = 6
    saved["response"]["usage"]["total_tokens"] = 16
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata mismatch"):
        runner.prepare(
            generation, rows=rows, formatter=formatter, output=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()
