"""CPU consistency checks, not new tasks, scores or workload acceptance."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bbh_receipt_preserves_stopped_before_answer_boundary():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/bbh-boolean-original-npu-20261023.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["source_checkbox_eligible"] is False
    assert receipt["unique_model_tasks"] == len(receipt["samples"]) == 3
    assert receipt["total_generation_calls_including_repeat"] == 6
    for sample in receipt["samples"]:
        assert sample["prompt_tokens"] == sample["preflight_tokens"]
        assert sample["finish_reason"] == "stop"
        assert sample["stop_reason"] == "\n\n"
        assert sample["answer_extracted"] is False
        assert sample["harness_exact_match"] == 0
    assert receipt["installed_reader"]["original_rows_checked"] == 250
    assert receipt["installed_reader"]["model_calls"] == 0
    assert receipt["complete_task_answers_validated"] is False
    assert receipt["full_suite_validated"] is False
    script = ROOT / receipt["public_runner"]
    assert (
        hashlib.sha256(script.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == receipt["public_runner_sha256"]
    )
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text(
            encoding="utf-8"
        )
    )
    row = next(row for row in ledger["sources"] if row["source_id"] == "BBH")
    assert row["checked"] is False
    assert row["selected_samples"] == receipt["selected_original_indices"]
    assert row["completed_samples"] == []
