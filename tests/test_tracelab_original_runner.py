"""Native replay wrapper contracts with fake metadata, no model calls."""

import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
runner = importlib.import_module("run_tracelab_original")


def test_selection_keeps_all_rounds_of_lexical_first_three(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text(
        "id,round_idx,prefix_len,input_len,output_len,tool_wait_after_ms\nz,0,0,1,2,0\na,0,0,2,3,4\nb,0,0,3,4,0\na,1,8,2,3,4\nc,0,0,4,5,0\n",
        encoding="utf-8",
    )
    ids, steps, demand = runner.selection(path)
    assert ids == ["a", "b", "c"] and steps == 4 and demand == 13
    assert runner.selection(path, full=True)[1] == 5


def test_summary_requires_every_native_round_no_skips():
    summary = {
        "workload": {"steps": 20},
        "replay": {
            "attempted_steps": 20,
            "success_steps": 20,
            "failed_steps": 0,
            "context_overflow_steps": 0,
        },
    }
    runner.validate_summary(summary, 20)
    summary["replay"]["context_overflow_steps"] = 1
    with pytest.raises(ValueError):
        runner.validate_summary(summary, 20)


def test_round_fidelity_and_failure_retention(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "id,round_idx,prefix_len,input_len,output_len,tool_wait_after_ms\na,0,3,4,5,600\n",
        encoding="utf-8",
    )
    log = tmp_path / "log.jsonl"
    record = {
        "session_id": "a",
        "round_idx": 0,
        "prefix_len": 3,
        "input_len": 4,
        "output_len_target": 5,
        "tool_wait_after_ms": 600,
        "status": "SUCCESS",
    }
    log.write_text(json.dumps(record) + "\n")
    runner.verify_rounds(trace, log, ["a"])
    record["tool_wait_after_ms"] = 0
    log.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        runner.verify_rounds(trace, log, ["a"])


def test_default_plan_never_calls_endpoint(monkeypatch, tmp_path, capsys):
    output = tmp_path / "absent"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--official-source",
            "absent",
            "--build",
            "absent",
            "--trace",
            "absent",
            "--text-file",
            "absent",
            "--tokenizer",
            "absent",
            "--model",
            "local",
            "--max-model-len",
            "262144",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        runner, "urlopen", lambda *a, **kw: pytest.fail("No endpoint call in plan")
    )
    runner.main()
    assert not output.exists()
    result = json.loads(capsys.readouterr().out)
    assert result["model_calls"] == 0
    assert result["mandatory_original_cache_probe"]
    assert not result["original_private_prompts_claim"]
