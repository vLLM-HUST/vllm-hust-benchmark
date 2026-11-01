"""Actual partial-run evidence stays distinct from source delivery."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/workload-data"


def load():
    return json.loads(
        (
            DOCS
            / "validation/scienceagentbench-original-fixed3-native-progress-20261101.json"
        ).read_text(encoding="utf-8")
    )


def test_native_partial_progress_binds_executed_entries_and_preserves_gaps():
    report = load()
    assert report["source_checkbox_eligible"] is False
    generation = report["generation"]
    assert generation["completed_generation_responses"] == 3
    assert generation["output_budget"] == 2000
    assert generation["api_dollar_cost"] is None
    assert [v["completion_tokens"] for v in generation["usage"]] == [701, 359, 329]
    for part in [generation, report["scientific_execution"]]:
        assert (
            hashlib.sha256(
                (ROOT / part["entry"]).read_text(encoding="utf-8").encode()
            ).hexdigest()
            == part["entry_sha256"]
        )
    aux = report["actual_program_auxiliary_scoring"]
    assert aux["programs_scored"] == 3 and aux["original_npu_codebert_forwards"] == 9
    scientific = report["scientific_execution"]
    assert scientific["original_case_ids_with_terminal_results"] == [2, 3]
    assert scientific["pending_case_ids"] == [1]
    assert scientific["case2"]["success"] == scientific["case3"]["success"] == 0
    assert scientific["case2"]["codebert_score"] == aux["original_codebert_scores"][1]
    assert scientific["case3"]["codebert_score"] == aux["original_codebert_scores"][2]
    assert scientific["independent_audit_program_executions"] == 0


def test_partial_scientific_run_never_promotes_whole_source():
    ledger = json.loads(
        (DOCS / "dataset-validation-status.json").read_text(encoding="utf-8")
    )
    entry = ledger["delivery_inventory"]["SCIENCEAGENTBENCH"]
    assert (
        entry["current_native_progress"]
        == "docs/workload-data/validation/scienceagentbench-original-fixed3-native-progress-20261101.json"
    )
    assert entry["checked"] is False
    assert entry["delivery_receipt"] is None
