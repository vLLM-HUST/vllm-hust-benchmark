"""Read-only post-run compiler constants and P-group callback metadata audit."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source(path):
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def group_checks(events):
    cohorts = defaultdict(list)
    failures = []
    for e in events:
        cohorts[(e["sequence"], e["prefill_cohort_id"])].append(e["group_index"])
        index = e["group_index"]
        names = e["all_moe_layers"] or []
        layer_ids = [
            int(m.group(1))
            if (m := re.search(r"(?:^|\.)layers\.(\d+)\.mlp$", name))
            else None
            for name in names
        ]
        checks = {
            "moe_index": e["moe_layer_index"] == 4 * (index + 1),
            "completed_layers_before_checkpoint": e[
                "task_completed_layers_before_checkpoint"
            ]
            == 4 * index,
            "prefill_lane_and_task": e["active_lane"] == e["task_name"] == "prefill",
            "final_flag": e["final"] is (index == 9),
            "all_40_absolute_layer_names_in_order": layer_ids == list(range(40)),
            "all_40_names_untruncated": e["all_moe_layers_count"] == 40
            and e["all_moe_layers_truncated"] is False,
        }
        failures.extend(
            {
                "sequence": e["sequence"],
                "cohort": e["prefill_cohort_id"],
                "group": index,
                "check": key,
            }
            for key, passed in checks.items()
            if not passed
        )
    return {
        "events": len(events),
        "cohorts": [
            {
                "sequence": seq,
                "cohort": cohort,
                "group_indices": indices,
                "complete_0_through_9": indices == list(range(10)),
            }
            for (seq, cohort), indices in cohorts.items()
        ],
        "failed_event_checks": failures,
        "scope": "Expected host state for qualified 40-layer model only; not KV contents or numerical equivalence.",
    }


def audit(directory):
    report = json.loads((directory / "run.json").read_text())
    if not report.get("finished_at") or not report.get("shutdown_returned"):
        raise ValueError(f"Run has no terminal shutdown receipt: {directory}")
    result = {
        "run": source(directory / "run.json"),
        "collection_passed": report["passed"],
        "compiler_ranks": [],
        "group_ranks": [],
        "sources": [],
    }
    observed = directory / "worker-observations"
    for path in sorted(observed.glob("rank-*-compiler-constants.json")):
        value = json.loads(path.read_text())
        result["sources"].append(source(path))
        table = value["constant_args"]
        payloads = Counter(digest(payload) for payload in table.values())
        result["compiler_ranks"].append(
            {
                "rank": value["rank"],
                "constant_args_count": len(table),
                "constant_ids": sorted(map(int, table)),
                "table_sha256": digest(table),
                "payload_sha256_multiplicities": dict(payloads),
                "payloads": {digest(p): p for p in table.values()},
                "environment": value["environment"],
                "deterministic_algorithms_enabled": value[
                    "deterministic_algorithms_enabled"
                ],
            }
        )
    for path in sorted(observed.glob("rank-*-prefill.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        events = [
            row for row in rows if row["event"] == "prefill-group-return-observed"
        ]
        result["sources"].append(source(path))
        result["group_ranks"].append({"file": path.name, **group_checks(events)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [audit(directory.resolve()) for directory in args.run]
    value = {
        "schema": "muxpd-numerical-host-audit-v1",
        "producer": source(Path(__file__)),
        "scope": "Host metadata only. Payload identity does not identify kernels, tensor values, graph references, or collective determinism.",
        "runs": rows,
    }
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for row in rows:
        print(
            Path(row["run"]["path"]).parent.parent.name,
            [
                (r["rank"], r["constant_args_count"], r["table_sha256"])
                for r in row["compiler_ranks"]
            ],
        )


if __name__ == "__main__":
    main()
