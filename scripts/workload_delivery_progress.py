"""Derive progress from delivery_inventory only, never frozen historical stages."""

import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/workload-data/dataset-validation-status.json"


def summarize(ledger):
    inventory = ledger["delivery_inventory"]
    if len(inventory) != 118:
        raise ValueError(
            "preserve agreed 118-source scope including five classification-only"
        )
    rows = []
    for sid, item in inventory.items():
        excluded = item["status"] == "CLASSIFICATION_ONLY"
        if type(item["checked"]) is not bool or item["checked"] != (
            item["status"] == "AVAILABLE"
        ):
            raise ValueError("checkbox/status mismatch")
        if item["checked"] and not item.get("delivery_receipt"):
            raise ValueError("checked source requires explicit delivery receipt")
        rows.append(
            {
                "source_id": sid,
                "status": item["status"],
                "checked": item["checked"],
                "classification_only": excluded,
                "delivery_receipt": item.get("delivery_receipt"),
            }
        )
    historical = ledger["existing_37_review"]["sources"]
    if len(historical) != 37 or set(historical) - set(inventory):
        raise ValueError("historical review identities must match current inventory")
    union = ledger["historical_evidence_union_review"]
    union_ids = union["source_ids"]
    if (
        len(union_ids) != 46
        or len(set(union_ids)) != 46
        or union["historical_source_count"] != 46
        or set(historical) - set(union_ids)
        or set(union_ids) - set(inventory)
    ):
        raise ValueError("historical 46-source evidence union identity drift")

    def group(selected):
        return {
            "registered": len(selected),
            "classification_only": sum(row["classification_only"] for row in selected),
            "required": sum(not row["classification_only"] for row in selected),
            "delivered": sum(row["checked"] for row in selected),
            "pending": sum(
                not row["classification_only"] and not row["checked"]
                for row in selected
            ),
        }

    totals = group(rows)
    if totals["classification_only"] != 5 or totals["required"] != 113:
        raise ValueError("preserve agreed 113 delivery plus five classification scope")
    return {
        "status": "LEDGER_DERIVED_PROGRESS_NOT_NEW_RUNTIME_EVIDENCE",
        "ledger_updated_at": ledger["updated_at"],
        "all_sources": totals,
        "existing_37": group([row for row in rows if row["source_id"] in historical]),
        "outside_existing_37": group(
            [row for row in rows if row["source_id"] not in historical]
        ),
        "historical_evidence_union_46": group(
            [row for row in rows if row["source_id"] in union_ids]
        ),
        "outside_historical_evidence_union_46": group(
            [row for row in rows if row["source_id"] not in union_ids]
        ),
        "historical_evidence_union_pending_ids": [
            row["source_id"]
            for row in rows
            if row["source_id"] in union_ids
            and not row["checked"]
            and not row["classification_only"]
        ],
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "existing_37_pending": [
            row
            for row in rows
            if row["source_id"] in historical
            and not row["checked"]
            and not row["classification_only"]
        ],
        "scope_note": "Historical 37 means reviewed read/prepare evidence, not 37 native deliveries. Do not use frozen sources array checked flags for current progress.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    args = parser.parse_args()
    print(json.dumps(summarize(json.loads(args.ledger.read_text("utf-8"))), indent=2))
