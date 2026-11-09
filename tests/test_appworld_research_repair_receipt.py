"""Receipt consistency only; original AppWorld tasks are not executed here."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_appworld_research_repair_preserves_native_and_lock_boundaries():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    repair = ledger["current_research_consumer_integrity_followup"]["latest_repair"]
    receipt = json.loads((ROOT / repair["receipt"]).read_text("utf-8"))
    assert receipt["source_id"] == "APPWORLD"
    assert receipt["local_synthetic_asset_controls_on_previous_consumer"] == {
        "failed": 6,
        "passed": 1,
    }
    assert receipt["local_repaired_appworld_controls_passed"] == 10
    assert receipt["local_appworld_gaia_provider_regressions_passed"] == 21
    assert receipt["fixed_public_and_repaired_research_readbacks"] == 3
    assert receipt["server_installed_provider_payloads_verified"] == 71
    assert receipt["all_four_exact_head_ci_checks_passed"]
    assert receipt["published_consumer_equals_server_qualified_candidate"]
    assert receipt["research_pr_draft"] and not receipt["research_pr_merged"]
    assert receipt["model_calls"] == 0
    for key in (
        "research_default_lock_changed",
        "research_main_deployment_claimed",
        "agent_or_official_scorer_rerun",
        "native_execution_or_scoring_receipts_rewritten",
        "source_delivery_count_changed",
        "raw_data_published",
    ):
        assert receipt[key] is False
