"""Keep corpus-generation delivery distinct from model quality and tool execution."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_xlam_receipt_preserves_fixed_original_scope_and_negative_format_results():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["XLAM_TOOL"]
    receipt = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    assert row["checked"] and row["status"] == "AVAILABLE"
    assert receipt["source_checkbox_eligible"]
    native = receipt["native_profile"]
    assert native["original_positions"] == [0, 1, 2]
    assert native["original_tool_counts"] == [1, 2, 2]
    assert native["original_query_and_all_tools_preserved"]
    assert native["reference_answers_excluded_from_input"]
    assert native["finish_reasons"] == {"stop": 3}
    assert native["original_author_output_budget"] == 512
    assert native["raw_json_reply_count"] == 0
    assert native["markdown_fenced_reply_count"] == 3
    for key in [
        "raw_outputs_rewritten",
        "quality_scored",
        "real_tool_api_execution",
        "author_model_reproduced",
    ]:
        assert native[key] is False
    handoff = receipt["provider_and_research"]
    assert handoff["installed_payloads_verified"] == 94
    assert handoff["executed_generation_files_equal_published_commit"] == 5
    assert handoff["fixed_public_research_readback_positions"] == [
        0,
        1,
        2,
        59997,
        59998,
        59999,
    ]
    assert handoff["actual_research_external_lock_cli_passed"]
    assert handoff["native_generations_repeated"] == 0
    assert handoff["original_index_rebuilt"] is False
    assert handoff["research_default_lock_changed"] is False
