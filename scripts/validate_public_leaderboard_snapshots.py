#!/usr/bin/env python3
"""Validate that public leaderboard snapshots do not include retired baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PUBLIC_BASELINE_ENGINE = "vllm"
PUBLIC_BASELINE_VERSION = "0.18.0"
RETIRED_BASELINE_TOKENS = ("v0.11.0", "v0110", "0.11.0")
OFFICIAL_PUBLIC_WORKLOADS = {
    "instructcoder-online",
    "prefix-repetition-online",
    "random-latency",
    "random-online",
    "sharegpt-online",
    "sharegpt-throughput",
    "sonnet-throughput",
    "visionarena-online",
}
OFFICIAL_V0180_SPEC_PREFIX = "official-ascend-jan-2026-v0.18.0-"
SNAPSHOT_FILES = (
    "leaderboard_single.json",
    "leaderboard_multi.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate curated public leaderboard snapshot files."
    )
    parser.add_argument(
        "--snapshot-dir",
        default="leaderboard-data/snapshots",
        help="Directory containing public leaderboard snapshot JSON files.",
    )
    return parser.parse_args()


def load_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"{path} must contain a JSON array")
    return [entry for entry in payload if isinstance(entry, dict)]


def contains_retired_baseline_token(value: Any) -> bool:
    normalized = str(value or "")
    return any(token in normalized for token in RETIRED_BASELINE_TOKENS)


def validate_entry(entry: dict[str, Any], *, source: Path) -> list[str]:
    errors: list[str] = []
    entry_id = str(entry.get("entry_id") or "<missing-entry-id>")
    engine = str(entry.get("engine") or "").strip().lower()
    engine_version = str(entry.get("engine_version") or "").strip()
    workload = str((entry.get("workload") or {}).get("name") or "").strip()
    model = entry.get("model") if isinstance(entry.get("model"), dict) else {}
    hardware = entry.get("hardware") if isinstance(entry.get("hardware"), dict) else {}
    same_spec = entry.get("same_spec") if isinstance(entry.get("same_spec"), dict) else {}
    spec_id = str(same_spec.get("spec_id") or "")

    if engine == PUBLIC_BASELINE_ENGINE and engine_version != PUBLIC_BASELINE_VERSION:
        errors.append(
            f"{source.name}:{entry_id}: public vllm baseline must be "
            f"{PUBLIC_BASELINE_VERSION}, got {engine_version!r}"
        )

    if contains_retired_baseline_token(engine_version):
        errors.append(
            f"{source.name}:{entry_id}: retired baseline version in engine_version "
            f"{engine_version!r}"
        )

    if contains_retired_baseline_token(spec_id):
        errors.append(
            f"{source.name}:{entry_id}: retired baseline spec_id {spec_id!r}"
        )

    if engine == "vllm-hust" and workload in OFFICIAL_PUBLIC_WORKLOADS and not spec_id:
        errors.append(
            f"{source.name}:{entry_id}: public vllm-hust official workload "
            f"{workload!r} must include same_spec"
        )

    if (
        engine == "vllm-hust"
        and workload in OFFICIAL_PUBLIC_WORKLOADS
        and spec_id
        and not spec_id.startswith(OFFICIAL_V0180_SPEC_PREFIX)
    ):
        errors.append(
            f"{source.name}:{entry_id}: public vllm-hust official workload "
            f"{workload!r} must use official v0.18.0 same_spec, got {spec_id!r}"
        )

    if spec_id.startswith(OFFICIAL_V0180_SPEC_PREFIX):
        expected_chip = "910B2" if spec_id.endswith("-910b2") else None
        entry_precision = str(model.get("precision") or "")
        spec_precision = str(same_spec.get("model_precision") or "")
        entry_chip = str(hardware.get("chip_model") or "")
        spec_chip = str(same_spec.get("hardware_chip_model") or "")

        if not spec_precision or entry_precision != spec_precision:
            errors.append(
                f"{source.name}:{entry_id}: official v0.18.0 public spec "
                f"precision must match same_spec; entry={entry_precision!r} "
                f"same_spec={spec_precision!r}"
            )
        if expected_chip and (entry_chip != expected_chip or spec_chip != expected_chip):
            errors.append(
                f"{source.name}:{entry_id}: official v0.18.0 public spec "
                f"{spec_id!r} must be {expected_chip}; entry={entry_chip!r} "
                f"same_spec={spec_chip!r}"
            )

    return errors


def main() -> int:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir)
    errors: list[str] = []

    for file_name in SNAPSHOT_FILES:
        path = snapshot_dir / file_name
        if not path.is_file():
            errors.append(f"missing snapshot file: {path}")
            continue
        for entry in load_entries(path):
            errors.extend(validate_entry(entry, source=path))

    if errors:
        print("public leaderboard snapshot validation failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    print("public leaderboard snapshots passed retired-baseline checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
