#!/usr/bin/env python3
"""Atomically repair effective server parameters and their same-spec hash."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from vllm_hust_benchmark.same_spec import compute_resolved_spec_hash  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair all same-spec payloads with one recorded hash."
    )
    parser.add_argument("--old-hash", required=True)
    parser.add_argument("--server-parameters-json", required=True)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--expected-payloads", required=True, type=int)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def matching_payloads(value: Any, *, old_hash: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if (
            value.get("schema_version") == "benchmark-same-spec/v1"
            and value.get("resolved_spec_hash") == old_hash
        ):
            matches.append(value)
        for child in value.values():
            matches.extend(matching_payloads(child, old_hash=old_hash))
    elif isinstance(value, list):
        for child in value:
            matches.extend(matching_payloads(child, old_hash=old_hash))
    return matches


def main() -> int:
    args = parse_args()
    overrides = json.loads(args.server_parameters_json)
    if not isinstance(overrides, dict):
        raise SystemExit("--server-parameters-json must contain a JSON object")

    changed: list[tuple[Path, Any]] = []
    payload_count = 0
    new_hashes: set[str] = set()
    paths = sorted(
        path for root in args.root for path in root.rglob("*.json") if path.is_file()
    )
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        matches = matching_payloads(value, old_hash=args.old_hash)
        if not matches:
            continue
        for payload in matches:
            server = payload.get("resolved_server_parameters")
            if not isinstance(server, dict):
                raise SystemExit(
                    f"{path}: resolved_server_parameters must be an object"
                )
            server.update(overrides)
            payload["resolved_spec_hash"] = compute_resolved_spec_hash(payload)
            new_hashes.add(payload["resolved_spec_hash"])
            payload_count += 1
        changed.append((path, value))

    if payload_count != args.expected_payloads:
        raise SystemExit(
            f"expected {args.expected_payloads} matching payloads, found {payload_count}"
        )
    if len(new_hashes) != 1:
        raise SystemExit(
            "repair would still produce multiple hashes: "
            + ", ".join(sorted(new_hashes))
        )

    new_hash = next(iter(new_hashes))
    if args.execute:
        for path, value in changed:
            path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    action = "updated" if args.execute else "would update"
    print(
        f"{action} {payload_count} payloads in {len(changed)} files; new hash={new_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
