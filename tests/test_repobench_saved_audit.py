"""Saved-output audit fixtures only; no native publisher/model execution."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "repobench_saved", ROOT / "scripts/verify_repobench_original_saved.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.mark.parametrize(
    "tamper", [None, "raw", "pred", "gt", "unknown", "missing", "duplicate"]
)
def test_saved_original_outputs_and_gt_cannot_be_changed(tmp_path, tamper):
    task = {
        "language": "python",
        "setting": "in_file",
        "filtered_index": 4,
        "original": {"level": "2k", "next_line": "ground truth"},
    }
    attempt = {
        "task_index": 0,
        "language": "python",
        "setting": "in_file",
        "filtered_index": 4,
        "status": "GENERATED_ORIGINAL_NEXT_LINE",
        "token_ids": [3],
        "raw_output": "generated line",
        "pred": "generated line",
    }
    row = {"idx": 4, "level": "2k", "pred": "generated line", "gt": "ground truth"}
    if tamper == "raw":
        attempt["raw_output"] = "changed raw answer"
    elif tamper == "pred":
        attempt["pred"] = "cleaned to pass"
    elif tamper == "gt":
        row["gt"] = "changed reference"
    elif tamper == "unknown":
        attempt["status"] = "STARTED_UNKNOWN_OUTCOME"
    if tamper != "missing":
        (tmp_path / "attempt-00000.private.json").write_text(json.dumps(attempt))
    output = tmp_path / "python"
    output.mkdir()
    (output / "in_file.jsonl").write_text(
        json.dumps(row)
        + "\n"
        + (json.dumps(row) + "\n" if tamper == "duplicate" else "")
    )
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: "generated line")
    author = SimpleNamespace(get_first_line_not_comment=lambda raw, language: raw)

    def run():
        completed, unknown = AUDIT.audit_saved_tasks(
            [task], tmp_path, author, tokenizer
        )
        AUDIT.audit_result_files(completed, tmp_path)
        return completed, unknown

    if tamper:
        with pytest.raises(ValueError):
            run()
    else:
        completed, unknown = run()
        assert len(completed) == 1 and not unknown


def test_partial_unknown_attempt_is_retained_not_completed(tmp_path):
    task = {"language": "python", "setting": "in_file", "filtered_index": 0}
    attempt = {**task, "task_index": 0, "status": "STARTED_UNKNOWN_OUTCOME"}
    (tmp_path / "attempt-00000.private.json").write_text(json.dumps(attempt))
    completed, unknown = AUDIT.audit_saved_tasks(
        [task], tmp_path, None, None, partial=True
    )
    assert completed == [] and unknown == [0]
