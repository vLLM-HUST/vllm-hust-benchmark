"""Software fixture text only; no generated code is executed."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "he_prepare", ROOT / "scripts/prepare_humaneval_official_samples.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(tmp_path):
    generation = tmp_path / "historical"
    generation.mkdir()
    records, samples = [], []
    for index in range(3):
        record = NS(
            runtime={"kind": "task_dataset"},
            task={
                "task_id": f"HumanEval/{index}",
                "input": {"prompt": f"def fixture_{index}():\n"},
            },
        )
        records.append(record)
        request = runner.completion_payload(record, "fixture-model")
        saved = {
            "request": request,
            "response": {
                "choices": [
                    {
                        "text": "```python\nraise RuntimeError('must not execute')\n```\n",
                        "finish_reason": "length",
                    }
                ]
            },
        }
        raw = json.dumps(saved).encode()
        (generation / f"HumanEval-{index}.private.json").write_bytes(raw)
        samples.append(
            {
                "task_id": record.task["task_id"],
                "status": "GENERATION_RECEIVED_NOT_SCORED",
                "private_result_sha256": hashlib.sha256(raw).hexdigest(),
                "prompt_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
            }
        )
    receipt = {
        "schema_version": "humaneval-generation/v1",
        "source_id": "HUMANEVAL",
        "revision": runner.REVISION,
        "data_sha256": runner.DATA_SHA,
        "status": "THREE_ORIGINAL_PROMPTS_GENERATED_NOT_SCORED",
        "model": "fixture-model",
        "generation_config": {"temperature": 0, "max_tokens": 2048, "n": 1},
        "samples": samples,
    }
    (generation / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return generation, records


def test_exact_completion_reused_without_cleanup_or_execution(tmp_path):
    generation, records = fixture(tmp_path)
    output = tmp_path / "official-input"
    result = runner.prepare(generation, records=records, output=output)
    samples = [
        json.loads(line)
        for line in (output / "samples-private.jsonl").read_text().splitlines()
    ]
    assert (
        samples[0]["completion"]
        == "```python\nraise RuntimeError('must not execute')\n```\n"
    )
    assert [sample["task_id"] for sample in samples] == [
        f"HumanEval/{i}" for i in range(3)
    ]
    assert result["new_model_calls"] == 0
    assert not result["functional_tests_executed"]
    assert not result["source_checkbox_eligible"]


def test_changed_completion_identity_rejected(tmp_path):
    generation, records = fixture(tmp_path)
    (generation / "HumanEval-0.private.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        runner.prepare(generation, records=records, output=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_changed_prompt_rejected_even_with_unchanged_generation(tmp_path):
    generation, records = fixture(tmp_path)
    records[0].task["input"]["prompt"] += "changed"
    with pytest.raises(ValueError, match="prompt"):
        runner.prepare(generation, records=records, output=tmp_path / "output")


def test_output_and_task_selection_fail_closed(tmp_path):
    generation, records = fixture(tmp_path)
    with pytest.raises(ValueError, match="fresh output"):
        runner.prepare(generation, records=records, output=generation / "new")
    with pytest.raises(ValueError, match="preselected"):
        runner.prepare(generation, records=records[::-1], output=tmp_path / "output")
