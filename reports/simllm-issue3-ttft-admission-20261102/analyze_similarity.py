"""Summarize the separately replayed final-workload similarity subset."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from pathlib import Path


def load(path: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    summary = next(row for row in rows if row["phase"] == "summary")
    return [row for row in rows if row["phase"] != "summary"], summary


def describe(rows: list[dict], summary: dict, log: Path) -> dict:
    warm = [row for row in rows if row["phase"] == "warm"]
    token_counts = [row["prompt_tokens"] for row in rows]
    event_pattern = (
        r"SimLLM: reusing (\d+) semantic tokens for request (\S+) "
        r"\(source=(\S+), cosine=([^)]*)\)"
    )
    with gzip.open(log, "rt", errors="replace") as source:
        events = re.findall(event_pattern, source.read())
    by_id = {row["id"]: row for row in rows}
    request_ids = {row["request_id"]: row["id"] for row in rows}

    def client_id(worker_id: str) -> str:
        matches = [
            item_id
            for request_id, item_id in request_ids.items()
            if worker_id == request_id or worker_id.startswith(request_id + "-")
        ]
        if len(matches) != 1:
            raise ValueError(f"Unmapped cache event: {worker_id}")
        return matches[0]

    mapped = [
        (client_id(request_id), client_id(source_id), int(skipped))
        for skipped, request_id, source_id, _cosine in events
    ]
    hit_targets = {target for target, _source, _skipped in mapped}
    return {
        "requests": len(rows),
        "warm_requests": len(warm),
        "first_request_latency_s": rows[0]["latency_s"],
        "warm_ttft_median_s": statistics.median(row["ttft_s"] for row in warm),
        "warm_ttft_mean_s": statistics.mean(row["ttft_s"] for row in warm),
        "requests_per_s": summary["requests_per_s"],
        "requests_per_s_after_first": (len(rows) - 1)
        / (summary["wall_s"] - rows[0]["latency_s"]),
        "wall_s": summary["wall_s"],
        "prompt_tokens_min": min(token_counts),
        "prompt_tokens_max": max(token_counts),
        "semantic_reuse_events": len(mapped),
        "semantic_skipped_tokens": sum(skipped for _, _, skipped in mapped),
        "cold_hit_events": sum(
            by_id[target]["phase"] == "cold" for target, _, _ in mapped
        ),
        "cross_group_hit_events": sum(
            by_id[target]["group_id"] != by_id[source]["group_id"]
            for target, source, _ in mapped
        ),
        "source_rollover_hit_events": sum(
            source in hit_targets for _target, source, _skipped in mapped
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    names = ("baseline", "plugin_default")
    if (args.run / "plugin_floor8192.outputs.jsonl").exists():
        names += ("plugin_floor8192",)
    loaded = {name: load(args.run / f"{name}.outputs.jsonl") for name in names}
    baseline = loaded["baseline"][0]
    baseline_by_id = {row["id"]: row for row in baseline}
    result = {}
    for name in names:
        rows, summary = loaded[name]
        if {row["id"] for row in rows} != set(baseline_by_id):
            raise ValueError(f"Request IDs mismatch in {name}")
        arm = describe(rows, summary, args.run / f"{name}.server.log.gz")
        arm["output_equal_to_baseline"] = sum(
            row["output_text"] == baseline_by_id[row["id"]]["output_text"]
            for row in rows
        )
        arm["warm_output_equal_to_baseline"] = sum(
            row["output_text"] == baseline_by_id[row["id"]]["output_text"]
            for row in rows
            if row["phase"] == "warm"
        )
        result[name] = arm
    if "plugin_floor8192" not in loaded:
        result["plugin_floor8192"] = {
            "status": "startup_failed",
            "log": "plugin_floor8192.server.log.gz",
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
