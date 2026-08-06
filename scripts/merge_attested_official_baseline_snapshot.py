#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_aggregator(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("vllm_hust_website_aggregate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load aggregator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError(f"snapshot must be a JSON object array: {path}")
    return payload


def _target_id(entry: dict[str, Any]) -> str:
    return str((entry.get("same_spec") or {}).get("spec_id") or "")


def _mark_covered(entry: dict[str, Any], target_id: str) -> None:
    if _target_id(entry) != target_id:
        return
    accountable = (entry.get("constraints") or {}).get("accountable_scope") or {}
    accountable["baseline_engine"] = "vllm"
    accountable["declared_baseline_engine"] = "vllm"
    accountable["baseline_status"] = "official-covered"


def _mark_legacy_unverified(entry: dict[str, Any]) -> None:
    metadata = entry.setdefault("metadata", {})
    if metadata.get("target_id") or metadata.get("verified") is True:
        return
    metadata["official_admission_status"] = "historical-unverified"
    metadata["official_admission_reason"] = (
        "Pre-attestation leaderboard record retained for historical visibility; "
        "not admitted as official fixed-target evidence."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge one verified official baseline into curated snapshots."
    )
    parser.add_argument("--aggregator-script", type=Path, required=True)
    parser.add_argument("--current-snapshot-dir", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--comparison-side",
        choices=("baseline", "current"),
        default="baseline",
        help="Merge the official upstream baseline or its verified current pair.",
    )
    args = parser.parse_args()

    aggregate = _load_aggregator(args.aggregator_script.resolve())
    schema = (
        args.aggregator_script.resolve().parents[1]
        / "data"
        / "schemas"
        / "leaderboard_v1.schema.json"
    )
    validator = aggregate.Draft7Validator(aggregate.load_schema(schema))
    incoming = aggregate.load_manifest_entries(args.submission_dir.resolve(), validator)
    incoming, rejected = aggregate.filter_public_snapshot_entries(incoming)
    if rejected or len(incoming) != 1:
        raise ValueError(
            f"expected one valid incoming baseline, got {len(incoming)}; rejected={rejected}"
        )
    incoming_entry = incoming[0]
    metadata = incoming_entry.get("metadata") or {}
    target_id = _target_id(incoming_entry)
    expected_engine = "vllm" if args.comparison_side == "baseline" else "vllm-hust"
    attestation = metadata.get("verification_attestation") or {}
    if (
        incoming_entry.get("engine") != expected_engine
        or metadata.get("verified") is not True
        or attestation.get("comparison_side", "baseline") != args.comparison_side
    ):
        raise ValueError(
            f"incoming entry is not a verified {args.comparison_side} "
            f"comparison ({expected_engine})"
        )
    if not target_id or metadata.get("target_id") != target_id:
        raise ValueError("incoming baseline target binding is missing")

    current_single = _load_entries(
        args.current_snapshot_dir / "leaderboard_single.json"
    )
    current_multi = _load_entries(args.current_snapshot_dir / "leaderboard_multi.json")
    current = current_single + current_multi
    old_ids = {str(entry.get("entry_id")) for entry in current}
    merged = [
        entry
        for entry in current
        if not (
            entry.get("engine") == expected_engine
            and _target_id(entry) == target_id
        )
    ]
    merged.append(incoming_entry)
    for entry in merged:
        _mark_legacy_unverified(entry)
        _mark_covered(entry, target_id)

    single, multi = aggregate.split_entries(merged)
    new_ids = {str(entry.get("entry_id")) for entry in single + multi}
    removed_ids = old_ids - new_ids
    if removed_ids:
        raise ValueError(
            f"curated merge removed existing entries: {sorted(removed_ids)}"
        )
    compare_entries = aggregate.filter_compare_publish_entries(single + multi)
    aggregate.validate_same_spec_goal_pairs(compare_entries)
    aggregate.validate_same_spec_compare_pairs(compare_entries)
    compare = aggregate.build_compare_snapshot(compare_entries)
    aggregate.write_outputs(args.output_dir.resolve(), single, multi, compare)
    rejected_report = args.current_snapshot_dir / "rejected_superseded_report.json"
    rejected_output = args.output_dir.resolve() / "rejected_superseded_report.json"
    if rejected_report.is_file() and rejected_report.resolve() != rejected_output:
        shutil.copy2(
            rejected_report,
            rejected_output,
        )
    print(f"single={len(single)} multi={len(multi)} target={target_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
