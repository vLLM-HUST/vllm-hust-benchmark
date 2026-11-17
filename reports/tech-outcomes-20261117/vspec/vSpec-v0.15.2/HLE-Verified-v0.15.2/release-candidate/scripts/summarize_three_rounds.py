"""Audit and print the private three-round FS/HLE throughput matrix.

Only complete solver receipts are accepted. This script never substitutes a
historical run for a missing current-round arm and does not write public data.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path


ROOTS = {
    "frontierscience": {
        "MTP2": (Path("/root/f1311n2"), "native", "r4"),
        "MTP4": (Path("/root/f1311n4"), "native", "r3"),
        "MTP6": (Path("/root/f1311n6"), "native", "r1"),
        "MTP8": (Path("/root/f1311n8"), "native", "r1"),
        "vSpec": (Path("/root/f1311bridge"), "dynamic", "r1"),
    },
    "hle-verified": {
        "MTP2": (Path("/root/h1311n2"), "native", "r1"),
        "MTP4": (Path("/root/h1311n4"), "native", "r1"),
        "MTP6": (Path("/root/h1311n6"), "native", "r1"),
        "MTP8": (Path("/root/h1311n8"), "native", "r2"),
        "vSpec": (Path("/root/h1311hist1"), "dynamic", "r1"),
    },
}

RETEST_DYNAMIC_ROOTS = {
    "frontierscience": Path("/root/f1311bridge-retest-20261116"),
    "hle-verified": Path("/root/h1311hist1-retest-20261116"),
}


def receipt(dataset: str, root: Path, arm: str, tag: str) -> dict | None:
    path = root / "runs" / dataset / arm / tag / "solver-receipt.json"
    if not path.is_file():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if row["dataset"] != dataset or row["arm"] != arm:
        raise ValueError(f"Receipt identity mismatch: {path}")
    if row["client_exit_code"] != 0 or row["transport_or_response_failures"] != 0:
        raise ValueError(f"Incomplete run: {path}")
    if row["case_count"] != row["record_count"]:
        raise ValueError(f"Incomplete case list: {path}")
    calculated = row["completion_tokens"] / row["client_wall_seconds"]
    if abs(calculated - row["output_tokens_per_second"]) > 1e-6:
        raise ValueError(f"Throughput arithmetic mismatch: {path}")
    return row


def finish_reasons(row: dict) -> dict[str, int]:
    request_dir = Path(row["solver_output"]) / "requests"
    request_files = sorted(request_dir.glob("*.json"))
    if len(request_files) != row["record_count"]:
        raise ValueError(f"Raw response count differs from receipt: {request_dir}")
    reasons: Counter[str] = Counter()
    for path in request_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "ok":
            raise ValueError(f"Non-OK raw response despite zero failures: {path}")
        reasons[str(record.get("finish_reason", "missing"))] += 1
    return dict(sorted(reasons.items()))


def main() -> None:
    for dataset, arms in ROOTS.items():
        print(f"\n## {dataset}")
        print("| Arm | Round 1 | Round 2 | Round 3 | Median | Range |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        rows: dict[str, list[dict | None]] = {}
        for name, (root, arm, first_tag) in arms.items():
            records = [
                receipt(
                    dataset,
                    RETEST_DYNAMIC_ROOTS[dataset] if name == "vSpec" and index else root,
                    arm,
                    tag,
                )
                for index, tag in enumerate((first_tag, "round2", "round3"))
            ]
            rows[name] = records
            available = [r["output_tokens_per_second"] for r in records if r]
            cells = [f"{r['output_tokens_per_second']:.2f}" if r else "pending" for r in records]
            median = f"{statistics.median(available):.2f}" if len(available) == 3 else "pending"
            spread = f"{min(available):.2f}–{max(available):.2f}" if len(available) == 3 else "pending"
            print(f"| {name} | {' | '.join(cells)} | {median} | {spread} |")
        for index in range(3):
            available_rows = [records[index] for records in rows.values() if records[index]]
            if not available_rows:
                continue
            keys = {"plan_contract_sha256", "selection_sha256", "common_config_sha256"}
            for key in keys:
                values = {row[key] for row in available_rows}
                if len(values) > 1:
                    raise ValueError(f"Round {index + 1} {dataset}: differing {key}: {values}")
            npus = {tuple(row["physical_npus"]) for row in available_rows}
            if len(npus) > 1:
                print(f"WARNING: round {index + 1} has different physical NPUs: {npus}")
            dynamic = rows["vSpec"][index]
            if dynamic:
                for name in ("MTP2", "MTP4", "MTP6", "MTP8"):
                    native = rows[name][index]
                    if native:
                        ratio = dynamic["output_tokens_per_second"] / native["output_tokens_per_second"]
                        print(f"Round {index + 1} vSpec/{name}: {ratio:.4f}x")
        if all(all(record is not None for record in records) for records in rows.values()):
            for name in ("MTP2", "MTP4", "MTP6", "MTP8"):
                paired = [
                    rows["vSpec"][index]["output_tokens_per_second"]
                    / rows[name][index]["output_tokens_per_second"]
                    for index in range(3)
                ]
                print(f"Median paired vSpec/{name}: {statistics.median(paired):.4f}x")
        print("\nRaw completion and timing audit:")
        for name, records in rows.items():
            for index, row in enumerate(records, 1):
                if row is None:
                    continue
                print(
                    f"{name} round {index}: {row['completion_tokens']} output tokens / "
                    f"{row['client_wall_seconds']:.3f}s; finish reasons "
                    f"{finish_reasons(row)}; transport/response failures "
                    f"{row['transport_or_response_failures']}"
                )


if __name__ == "__main__":
    main()
