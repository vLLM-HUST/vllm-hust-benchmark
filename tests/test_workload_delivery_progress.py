"""Progress is derived from the single current ledger, not historical flags."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "delivery_progress", ROOT / "scripts/workload_delivery_progress.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def ledger():
    return json.loads(runner.LEDGER.read_text("utf-8"))


def test_groups_partition_current_scope_without_double_counting():
    result = runner.summarize(ledger())
    all_sources = result["all_sources"]
    assert all_sources["required"] == 113
    assert all_sources["classification_only"] == 5
    for key in all_sources:
        assert (
            result["existing_37"][key] + result["outside_existing_37"][key]
            == all_sources[key]
        )
    assert all_sources["delivered"] + all_sources["pending"] == 113
    assert result["existing_37"]["registered"] == 37
    assert len(result["existing_37_pending"]) == result["existing_37"]["pending"]


def test_frozen_source_flags_never_drive_current_progress():
    original = ledger()
    changed = deepcopy(original)
    for item in changed["sources"]:
        item["checked"] = not item["checked"]
    assert runner.summarize(original) == runner.summarize(changed)


def test_historical_46_is_evidence_coverage_not_a_second_completion_counter():
    current = ledger()
    result = runner.summarize(current)
    historical = result["historical_evidence_union_46"]
    outside = result["outside_historical_evidence_union_46"]
    assert historical["registered"] == 46
    for key in result["all_sources"]:
        assert historical[key] + outside[key] == result["all_sources"][key]
    assert historical["pending"] == len(result["historical_evidence_union_pending_ids"])
    changed = deepcopy(current)
    changed["historical_evidence_union_review"]["source_ids"].pop()
    with pytest.raises(ValueError, match="46-source"):
        runner.summarize(changed)


def test_missing_delivery_receipt_prevents_completion_count():
    changed = ledger()
    row = next(
        item for item in changed["delivery_inventory"].values() if item["checked"]
    )
    row["delivery_receipt"] = None
    with pytest.raises(ValueError, match="delivery receipt"):
        runner.summarize(changed)


def test_scope_or_checkbox_drift_is_rejected():
    changed = ledger()
    row = next(
        item for item in changed["delivery_inventory"].values() if item["checked"]
    )
    row["checked"] = False
    with pytest.raises(ValueError, match="checkbox/status"):
        runner.summarize(changed)
    changed["delivery_inventory"].pop(next(iter(changed["delivery_inventory"])))
    with pytest.raises(ValueError, match="118-source"):
        runner.summarize(changed)
