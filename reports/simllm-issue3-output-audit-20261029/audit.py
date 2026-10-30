"""Audit retained 30-request SimLLM outputs against baseline and reuse logs.

This is a diagnostic comparison, not a task-quality score. Run it against the
committed issue #2 inference/rollback evidence; no server or model is needed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REUSE = re.compile(
    r"SimLLM: reusing (?P<skipped>\d+) semantic tokens for request "
    r"(?P<target>\S+) \(source=(?P<source>\S+), cosine=(?P<cosine>[0-9.]+)\)"
)
SOURCE = (
    "vLLM-HUST/vllm-ascend-simllm-hust@28f2297c58cec2249576dff2ccdd251c662ad8d6/"
    "benchmarks/results/issue2_inference_rollback_20261019"
)


def rows(path: Path) -> list[dict]:
    return [
        row
        for line in path.read_text().splitlines()
        if line
        if (row := json.loads(line)).get("id") is not None
    ]


def resolve(server_id: str, requests: dict[str, dict]) -> dict:
    matches = [row for rid, row in requests.items() if server_id.startswith(rid + "-")]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely resolve server request {server_id}")
    return matches[0]


def audit(root: Path) -> dict:
    run = root / "run5"
    baseline = {row["id"]: row for row in rows(run / "baseline/outputs.jsonl")}
    final = rows(run / "final/outputs.jsonl")
    rollback = {row["id"]: row for row in rows(run / "rollback/outputs.jsonl")}
    if len(baseline) != 30 or len(final) != 30 or len(rollback) != 30:
        raise ValueError("Expected three complete 30-request arms")
    if set(baseline) != {row["id"] for row in final} or set(baseline) != set(rollback):
        raise ValueError("Arms have different request IDs")
    by_request = {row["request_id"]: row for row in final}
    events = [
        match.groupdict()
        for line in (run / "final/server.log").read_text().splitlines()
        if (match := REUSE.search(line))
    ]
    if len(events) != 24:
        raise ValueError(f"Expected 24 recorded reuse events, got {len(events)}")
    by_target: dict[str, dict] = {}
    for event in events:
        target = resolve(event["target"], by_request)
        source = resolve(event["source"], by_request)
        if target["id"] in by_target:
            raise ValueError(f"Duplicate reuse event for {target['id']}")
        by_target[target["id"]] = {
            "source_id": source["id"],
            "source_group_id": source["group_id"],
            "source_output_text": source["output_text"],
            "skipped_tokens": int(event["skipped"]),
            "cosine": float(event["cosine"]),
        }

    comparisons = []
    for target in final:
        original = baseline[target["id"]]
        restored = rollback[target["id"]]
        if original["output_text"] != restored["output_text"]:
            raise ValueError(f"Rollback differs from baseline for {target['id']}")
        reuse = by_target.get(target["id"])
        comparisons.append(
            {
                "id": target["id"],
                "group_id": target["group_id"],
                "reused": reuse is not None,
                "reuse": reuse,
                "baseline_output_text": original["output_text"],
                "plugin_output_text": target["output_text"],
                "output_text_equal": original["output_text"] == target["output_text"],
                "baseline_completion_tokens": original["completion_tokens"],
                "plugin_completion_tokens": target["completion_tokens"],
            }
        )
    hits = [row for row in comparisons if row["reused"]]
    misses = [row for row in comparisons if not row["reused"]]
    return {
        "source": SOURCE,
        "scope": "retained 30-request lifecycle diagnostic, 16 output tokens each",
        "requests": len(comparisons),
        "reuse_hits": len(hits),
        "cross_group_sources": sum(
            row["group_id"] != row["reuse"]["source_group_id"] for row in hits
        ),
        "skipped_tokens": sum(row["reuse"]["skipped_tokens"] for row in hits),
        "hit_text_equal_baseline": sum(row["output_text_equal"] for row in hits),
        "miss_text_equal_baseline": sum(row["output_text_equal"] for row in misses),
        "rollback_text_equal_baseline": len(comparisons),
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(audit(args.evidence), indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
