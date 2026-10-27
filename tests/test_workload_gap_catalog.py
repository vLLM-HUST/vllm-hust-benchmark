"""Validate proposal honesty and source identity, not dataset or model execution."""

import json
from pathlib import Path


CATALOG = Path(__file__).parents[1] / "docs/workload-data/gap-sources-20261021.json"


def test_gap_sources_are_proposals_not_claimed_integrations():
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert data["baseline"] == {
        "research_classes": 36,
        "public_candidate_sources": 89,
        "enterprise_assets": 8,
    }
    state = data["status_for_all_new_sources"]
    assert state["official_reference_checked"]
    for key in (
        "payload_acquired_in_this_change",
        "adapter_implemented_in_this_change",
        "real_execution_validated_in_this_change",
        "redistribution_review_completed",
    ):
        assert state[key] is False
    ids = [entry["id"] for entry in data["sources"]]
    assert len(ids) == len(set(ids)) == 17
    for source in data["sources"]:
        assert source["required_assets"] and source["limit"]
        assert source["official_urls"]
        assert all(url.startswith("https://") for url in source["official_urls"])
    for proposal in data["classification_proposals"]:
        assert set(proposal["sources"]) <= set(ids)


def test_unsupported_event_evidence_is_not_filled_by_adjacent_data():
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    gaps = {entry["id"]: entry for entry in data["unresolved_evidence_gaps"]}
    assert (
        gaps["joint-update-delete-query-events"]["status"]
        == "NO_COMPLETE_DIRECT_SOURCE_CONFIRMED"
    )
    assert gaps["runtime-kv-preemption-recovery-events"]["related_sources"] == []
    sources = {entry["id"]: entry for entry in data["sources"]}
    assert sources["COPILOT_AGENT_2026"]["fit"] == "DIRECT_METADATA_ONLY"
    assert "non-commercial" in sources["MSMARCO_PASSAGE"]["license_hint"]
    assert sources["DRAGON_PUBLIC"]["fit"].startswith("PARTIAL")
