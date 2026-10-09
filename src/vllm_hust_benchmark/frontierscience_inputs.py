"""Frozen FrontierScience sources shared by auditing and execution."""

import hashlib
import json
from pathlib import Path

REVISION = "25ed67db7da8f4591484e764008ff585544f5a30"  # pragma: allowlist secret
SOURCES = {
    "olympiad": (
        100,
        "450e7fceb53c8140283a752f0d62a3abc69a85e252d4dbbd5d9dfbfb762db4e4",  # pragma: allowlist secret -- public dataset SHA-256
    ),
    "research": (
        60,
        "96c0434abfcbadd6ef6f59a03cc374be4caf9c1f2d5e62d8fe921e768f66aa46",  # pragma: allowlist secret -- public dataset SHA-256
    ),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_rows(source: Path, track: str) -> list[dict]:
    expected_count, expected_hash = SOURCES[track]
    data = source.read_bytes()
    if sha(data) != expected_hash:
        raise ValueError(f"Source SHA-256 mismatch: {track}")
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != expected_count:
        raise ValueError(f"Unexpected task count: {track}")
    return rows
