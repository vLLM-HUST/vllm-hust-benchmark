"""Progress is derived from the single current ledger, not historical flags."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "delivery_progress", ROOT / "scripts/workload_delivery_progress.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def ledger():
    return json.loads(runner.LEDGER.read_text("utf-8"))


def test_pending_sources_use_current_inventory_only():
    data = ledger()
    report = runner.summarize(data)
    pending = report["pending_sources"]
    assert len(pending) == report["all_sources"]["pending"]
    assert {row["source_id"] for row in pending} == {
        source_id
        for source_id, item in data["delivery_inventory"].items()
        if not item["checked"] and item["status"] != "CLASSIFICATION_ONLY"
    }
    for row in pending:
        item = data["delivery_inventory"][row["source_id"]]
        assert row["status"] == item["status"]
        assert row["blocker"] == item.get("blocker")


def test_current_markdown_total_matches_ledger_not_historical_snapshots():
    text = (ROOT / "docs/workload-data/DATASET_CHECKLIST.zh-CN.md").read_text("utf-8")
    result = runner.summarize(ledger())["all_sources"]
    totals = re.findall(
        r"当前交付\s+\*\*(\d+)/113项\*\*，待完成(\d+)项，另(\d+)项仅分类",
        text,
    )
    assert totals == [
        tuple(
            str(result[key]) for key in ("delivered", "pending", "classification_only")
        )
    ]
    assert "当前机器清单计算：**57/" not in text
    assert "当前交付可用48/" not in text


def test_current_118_row_markdown_checkboxes_match_single_ledger():
    lines = (
        (ROOT / "docs/workload-data/DATASET_CHECKLIST.zh-CN.md")
        .read_text("utf-8")
        .splitlines()
    )
    starts = [
        i
        for i, line in enumerate(lines)
        if line.startswith("| 来源 ID") and "本轮勾选" in line
    ]
    assert len(starts) == 1
    rows = {}
    for line in lines[starts[0] + 2 :]:
        if not line.startswith("|"):
            break
        fields = [field.strip() for field in line.split("|")[1:-1]]
        source_id, state = fields[:2]
        assert source_id not in rows
        rows[source_id] = "[x]" in state
    assert rows == {
        source_id: item["checked"]
        for source_id, item in ledger()["delivery_inventory"].items()
    }


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


def test_current_checked_receipts_are_real_contained_json_and_source_bound():
    # Custody/consistency only. This cannot replace source-specific native audits.
    for source_id, row in ledger()["delivery_inventory"].items():
        if not row["checked"]:
            continue
        receipt_path = (ROOT / row["delivery_receipt"]).resolve(strict=True)
        assert receipt_path.is_relative_to(ROOT)
        assert receipt_path.is_file() and receipt_path.suffix == ".json"
        receipt = json.loads(receipt_path.read_text("utf-8"))
        assert isinstance(receipt, dict)
        # Shared multi-source receipts are allowed, but must name this source.
        assert json.dumps(source_id) in json.dumps(receipt)
        assert receipt.get("source_checkbox_eligible") is not False


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
