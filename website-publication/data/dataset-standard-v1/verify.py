"""Validate the frozen standard without downloading data or running inference."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def validate(root=ROOT):
    expected_files = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if Path(name).name != name or name in expected_files:
            raise ValueError("Unsafe or duplicate checksum entry")
        expected_files[name] = digest
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Bundle checksum mismatch: " + name)
    if {p.name for p in root.iterdir() if p.is_file()} != set(expected_files) | {
        "SHA256SUMS"
    }:
        raise ValueError("Unexpected bundle membership")
    standard = json.loads((root / "standard.json").read_bytes())
    if standard["standard_id"] != "five-datasets-20261113-v1":
        raise ValueError("Unexpected standard identity")
    for name, expected in standard["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe artifact path")
        path = root / name
        raw = path.read_bytes()
        if (
            path.is_symlink()
            or len(raw) != expected["bytes"]
            or hashlib.sha256(raw).hexdigest() != expected["sha256"]
        ):
            raise ValueError("Frozen artifact mismatch: " + name)
    mmlu = json.loads((root / "mmlu-fixed200.json").read_bytes())
    hle = json.loads((root / "hle-fixed200.json").read_bytes())
    frontier = json.loads((root / "frontier-full160.json").read_bytes())
    trace = json.loads((root / "public-agent-trace-fixed32-v1.json").read_bytes())
    for data, count in [(mmlu, 200), (hle, 200), (frontier, 160)]:
        ids = [c["case_id"] for c in data["cases"]]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError("Invalid frozen task membership")
    if Counter((c["track"], c["modality"]) for c in hle["cases"]) != {
        ("gold", "text"): 64,
        ("gold", "image"): 10,
        ("revision", "text"): 113,
        ("revision", "image"): 13,
    }:
        raise ValueError("HLE allocation changed")
    if [c["case_id"] for c in frontier["cases"]] != [
        f"{part}-{i:04d}"
        for part, n in [("olympiad", 100), ("research", 60)]
        for i in range(n)
    ]:
        raise ValueError("Frontier ordering changed")
    for name, count in [("swe", 64), ("terminal", 48)]:
        dataset = trace["datasets"][name]
        requests = dataset["requests"]
        if len(requests) != count or len({r["task_id"] for r in requests}) != 32:
            raise ValueError("Trace coverage changed: " + name)
    return standard


if __name__ == "__main__":
    validate()
    print(
        "PASS: five frozen datasets; 200/200/160 tasks and 64/48 trace requests per arm"
    )
