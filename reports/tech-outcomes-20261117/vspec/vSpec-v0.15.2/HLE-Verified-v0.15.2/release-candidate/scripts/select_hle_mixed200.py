"""Freeze the requested HLE-Verified 200/1811 systematic sample.

Requires a verified hust-eval-hle-verified plan containing Gold and Revision.
The output is a private selection manifest consumable by paired.py; it never
changes the plan or fetches data. Each stratum retains its source-plan order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

QUOTAS = {
    "gold:text": 64,
    "gold:image": 10,
    "revision:text": 113,
    "revision:image": 13,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stratum(task: dict) -> str:
    return f"{task['track']}:{'image' if task.get('image') else 'text'}"


def systematic_ids(tasks: list[dict], seed: int = 17) -> tuple[list[str], dict[str, int]]:
    groups: dict[str, list[str]] = {key: [] for key in QUOTAS}
    for task in tasks:
        key = stratum(task)
        if key not in groups:
            raise ValueError(f"Unexpected task stratum: {key}")
        groups[key].append(task["case_id"])
    selected: set[str] = set()
    sizes = {key: len(group) for key, group in groups.items()}
    for key, quota in QUOTAS.items():
        group = groups[key]
        if len(group) < quota:
            raise ValueError(f"Insufficient {key} cases: {len(group)} < {quota}")
        step = len(group) / quota
        seed_bytes = hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8]
        offset = int.from_bytes(seed_bytes, "big") / (1 << 64) * step
        indices = [math.floor(offset + i * step) for i in range(quota)]
        if len(set(indices)) != quota or indices[-1] >= len(group):
            raise AssertionError(f"Systematic index failure in {key}")
        selected.update(group[index] for index in indices)
    result = [task["case_id"] for task in tasks if task["case_id"] in selected]
    if len(result) != 200 or len(set(result)) != 200:
        raise AssertionError("Expected 200 unique sampled cases")
    actual = Counter(stratum(task) for task in tasks if task["case_id"] in selected)
    if dict(actual) != QUOTAS:
        raise AssertionError(f"Quota mismatch: {actual}")
    return result, sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    plan = args.plan.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    contract_path = plan / "contract.json"
    contract = json.loads(contract_path.read_text())
    if contract.get("selection", {}).get("subsets") != ["gold", "revision"]:
        raise ValueError("Plan must include Gold and Revision in source order")
    if contract.get("selection", {}).get("modality") != "all":
        raise ValueError("Plan must include both text and image cases")
    tasks_path = plan / "tasks.json"
    if sha256_file(tasks_path) != contract.get("tasks_sha256"):
        raise ValueError("Plan tasks do not match the frozen contract")
    tasks = json.loads(tasks_path.read_text())
    if len(tasks) != 1811:
        raise ValueError(f"Expected 1811 source cases, got {len(tasks)}")
    ids, sizes = systematic_ids(tasks, args.seed)
    encoded_ids = json.dumps(ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "schema_version": 1,
        "dataset": "hle-verified",
        "source_plan_contract_sha256": sha256_file(contract_path),
        "selection_seed": f"stratified-systematic-sha256-{args.seed}",
        "selection_method": "Within each track/modality stratum, source order; SHA-256 seeded start offset; fixed interval N/k",
        "source_strata": sizes,
        "quotas": QUOTAS,
        "count": 200,
        "case_ids": ids,
        "case_ids_sha256": hashlib.sha256(encoded_ids).hexdigest(),
        "strata": {"gold": 74, "revision": 126},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    output.chmod(0o600)
    print(json.dumps({"count": 200, "source_strata": sizes,
                      "quotas": QUOTAS, "case_ids_sha256": manifest["case_ids_sha256"]}))


if __name__ == "__main__":
    main()
