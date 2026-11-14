"""Validate complete native/reused campaigns without rewriting stage identities."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

STAGES = (
    "continuous-B0", "continuous-B1", "mmlu-pair",
    "frontier-solver-B0", "frontier-solver-B1", "frontier-judge-B0", "frontier-judge-B1",
    "hle-solver-B0", "hle-solver-B1", "hle-judge-B0", "hle-judge-B1",
    "swe-trace-B0", "swe-trace-B1", "terminal-trace-B0", "terminal-trace-B1",
)


def validate(data):
    receipt = data["receipt"]
    rows = receipt["stages"]
    plan = data.get("plan", {})
    if (set(plan.get("datasets", [])) != {"mmlu", "hle", "frontier", "swe", "terminal"}
            or plan.get("smoke_evaluators") is not False
            or plan.get("hle_profile") != "fixed200-v1"
            or plan.get("mmlu_inputs", {}).get("requests") != 200):
        raise ValueError("Export requires the frozen five-profile standard, not diagnostic or alternate scope")
    if receipt["status"] != "completed" or not receipt.get("selected_execution_complete"):
        raise ValueError("Export requires a completed frozen five-dataset campaign")
    if len(rows) != len(STAGES) or {s["name"] for s in rows} != set(STAGES):
        raise ValueError("Export requires all15 original stages exactly once")
    if any(s["status"] not in ("completed", "reused_verified") for s in rows):
        raise ValueError("Failed or unexecuted stages cannot supply a complete publication")
    reused = sum(s["status"] == "reused_verified" for s in rows)
    if reused != receipt.get("reused_inference_stages", 0) or len(rows) - reused != receipt.get("new_inference_stages", len(rows)):
        raise ValueError("Campaign new/reused counts differ from its actual stages")
    paths = data.get("reused_stage_sources", {})
    if set(paths) != {s["name"] for s in rows if s["status"] == "reused_verified"}:
        raise ValueError("Reused stage origins do not match the frozen receipt")
    for row in rows:
        if row["status"] == "reused_verified" and not row.get("source"):
            raise ValueError("Reused stage is missing its original provenance")
        if row["status"] == "completed" and row.get("returncode") != 0:
            raise ValueError("New stage did not finish successfully")
    return reused, len(rows) - reused


def roots(run, data):
    validate(data)
    run = Path(run)
    reused = data.get("reused_stage_sources", {})
    return {name: Path(reused[name]) if name in reused else run / name for name in STAGES}


def provenance(data):
    validate(data)
    result = []
    for row in data["receipt"]["stages"]:
        original = row["source"]["source_stage_receipt"] if row["status"] == "reused_verified" else row
        result.append(dict(
            name=row["name"], status=row["status"],
            execution="reused" if row["status"] == "reused_verified" else "new",
            source_index=row["source"]["source_index"] if row["status"] == "reused_verified" else None,
            original_started=original["started"], original_ended=original["ended"],
        ))
    return result


def report_date(data):
    return datetime.fromtimestamp(data["receipt"]["ended"], timezone(timedelta(hours=8))).date().isoformat()


def device_cards(root):
    """Use each stage's recorded physical devices, including reused origins."""
    cards = json.loads((Path(root) / "device-mapping.json").read_bytes())["physical_cards"]
    if (not isinstance(cards, list) or len(cards) != 2
            or any(type(card) is not int or card < 0 for card in cards)
            or len(set(cards)) != 2):
        raise ValueError("Utilization requires two distinct recorded physical cards")
    return tuple(str(card) for card in cards)
