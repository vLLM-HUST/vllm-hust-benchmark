"""Metadata classification only; no datasets, native execution or readiness claim."""

import json
from pathlib import Path

import pytest

from vllm_hust_workload_data import provider, taxonomy


def test_all_118_ledger_sources_have_one_source_plan():
    root = Path(__file__).resolve().parents[2]
    ledger = json.loads(
        (root / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]
    assets = taxonomy.delivery_assets()
    assert len(assets) == len({r["source_id"] for r in assets}) == 118
    assert {r["source_id"] for r in assets} == set(ledger)
    assert len(taxonomy.delivery_classes()) == 40
    assert len(taxonomy.research_classes()) == 38
    assert all(
        r["native_validation"] == "NOT_ESTABLISHED_BY_CLASSIFICATION" for r in assets
    )
    assert all(r["local_availability"] == "NOT_CHECKED_ON_THIS_MACHINE" for r in assets)


def test_enterprise_and_traces_not_forced_into_task_classes():
    assets = taxonomy.delivery_assets()
    private = [r for r in assets if r["asset_group"] == "enterprise-private-requests"]
    assert len(private) == 8
    assert provider.delivery_source_plan("BURSTGPT")["asset_group"] == "serving-traces"
    assert provider.delivery_source_plan("MOONCAKE")["related_workload_ids"] == []
    assert (
        provider.delivery_source_plan("RULER")["asset_group"]
        == "controlled-long-context-probes"
    )
    assert provider.delivery_source_plan("PAIO-CODE-EVAL-1000")[
        "related_workload_ids"
    ] == ["W12"]
    assert (
        provider.delivery_source_plan("PAIO-SEMANTIC-SIMILAR-5000")[
            "related_workload_ids"
        ]
        == []
    )
    assert not provider.delivery_source_plan("COPILOT_AGENT_2026")[
        "data_provider_source_registered"
    ]


def test_class_lookup_preserves_family_and_plan_identity():
    for row in taxonomy.delivery_classes():
        assert provider.delivery_class_plan(
            row["workload_id"]
        ) == provider.delivery_class_plan(row["family_id"])
    with pytest.raises(ValueError):
        provider.delivery_class_plan("W41")
    with pytest.raises(ValueError):
        provider.delivery_source_plan("invented")


def test_five_classification_only_sources_are_present_not_claimed_ready():
    root = Path(__file__).resolve().parents[2]
    ledger = json.loads(
        (root / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]
    special = [
        sid
        for sid, row in ledger.items()
        if row.get("classification_only") or row["status"] == "CLASSIFICATION_ONLY"
    ]
    assert len(special) == 5
    for sid in special:
        row = provider.delivery_source_plan(sid)
        assert row["related_workload_ids"]
        assert row["native_validation"] == "NOT_ESTABLISHED_BY_CLASSIFICATION"
