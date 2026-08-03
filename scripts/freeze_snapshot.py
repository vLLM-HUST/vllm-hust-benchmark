#!/usr/bin/env python3
"""Freeze a leaderboard snapshot by recording checksums and entry ids."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO_ROOT / "leaderboard-data" / "snapshots"
LEADERBOARD_SINGLE_FILE = "leaderboard_single.json"
LEADERBOARD_MULTI_FILE = "leaderboard_multi.json"
SCHEMA_VERSION = "freeze-snapshot/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a leaderboard snapshot into a reproducible manifest."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path for the freeze manifest.",
    )
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def load_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"{path} must contain a JSON array")
    return [entry for entry in payload if isinstance(entry, dict)]


def collect_entry_ids(path: Path) -> list[str]:
    entry_ids: list[str] = []
    for entry in load_entries(path):
        entry_id = entry.get("entry_id")
        if entry_id is not None:
            entry_ids.append(str(entry_id))
    return entry_ids


def resolve_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        print(
            "warning: unable to determine git commit; source_commit set to null",
            file=sys.stderr,
        )
        return None
    if result.returncode != 0:
        print(
            "warning: unable to determine git commit; source_commit set to null",
            file=sys.stderr,
        )
        return None
    return result.stdout.strip() or None


def main() -> int:
    args = parse_args()

    single_path = SNAPSHOT_DIR / LEADERBOARD_SINGLE_FILE
    multi_path = SNAPSHOT_DIR / LEADERBOARD_MULTI_FILE

    for path in (single_path, multi_path):
        if not path.is_file():
            print(f"error: missing snapshot file: {path}", file=sys.stderr)
            return 1

    leaderboard_single_checksum = compute_sha256(single_path)
    leaderboard_multi_checksum = compute_sha256(multi_path)
    entry_ids = collect_entry_ids(single_path)
    source_commit = resolve_git_commit()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "leaderboard_single_checksum": leaderboard_single_checksum,
        "leaderboard_multi_checksum": leaderboard_multi_checksum,
        "entry_ids": entry_ids,
        "source_commit": source_commit,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"froze snapshot manifest to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
