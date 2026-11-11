"""Summarize the frozen paired queue / driver submission diagnostic capsule."""

import argparse
import bisect
import collections
import json
import statistics
from pathlib import Path


def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(root):
    result = {"native": {}, "paired_resources": {}, "driver": {}, "calibration": {}}
    for path in sorted(root.glob("hust-zsx*/native-*/operations.jsonl")):
        key = path.parent.relative_to(root).as_posix()
        operations = [row for row in jsonl(path) if row["event"] == "restore"]
        result["native"][key] = {
            "completed": (path.parent / "done.json").exists(),
            "restores": [
                {
                    field: row[field]
                    for field in [
                        "phase",
                        "policy",
                        "complete_s",
                        "descriptors",
                        "bytes",
                        "all_tags",
                    ]
                    if field in row
                }
                for row in operations
            ],
        }
        api = jsonl(path.parent / "raw/api.jsonl")
        paired = [
            row
            for row in api
            if row["event"] == "sq_progress_v3"
            and row.get("resource_time_ns")
            and not any(row["errors"])
        ]
        if paired:
            lag = [(row["start_ns"] - row["resource_time_ns"]) / 1e3 for row in paired]
            result["paired_resources"][key] = {
                "samples": len(paired),
                "median_observer_lag_us": statistics.median(lag),
                "max_observer_lag_us": max(lag),
                "full_resource_samples": sum(
                    (r["resource_tail"] + 1) % r["resource_cap"] == r["resource_head"]
                    for r in paired
                ),
                "hardware_head_tail_equal_samples": sum(
                    r["head"] == r["tail"] == r["head_again"] for r in paired
                ),
            }
        mappings = [row for row in api if row["event"] == "sq_stream_map_v3"]
        hal = [row for row in api if row["event"] == "hal_send"]
        if not hal or not mappings:
            continue
        sq = mappings[0]["sq"]
        calls = sorted(
            [row for row in hal if row["sq"] == sq and row["kind"] == "task"],
            key=lambda row: row["end_ns"],
        )
        accepted = [row for row in calls if row["error"] == 0]
        expected = sum(row["descriptors"] for row in operations)
        # Native input geometry uses one descriptor per successful driver SQ entry.
        depth = 2048
        assert len(accepted) == expected, (key, len(accepted), expected)
        assert all(
            r["count"] == 1 and r["position"] == i % depth
            for i, r in enumerate(accepted)
        ), key
        ends = [row["end_ns"] for row in accepted]
        posts = sorted(
            [
                row
                for row in api
                if row["event"] == "sq_progress_v3" and row["during_copy"] == 0
            ],
            key=lambda row: row["start_ns"],
        )
        queries = sorted(
            [
                row
                for row in api
                if row["event"] == "sq_progress_v3" and row["during_copy"] == 1
            ],
            key=lambda row: row["start_ns"],
        )
        assert len(posts) == len(operations), key
        phases = []
        base = 0
        for index, operation in enumerate(operations):
            stop = posts[index]["start_ns"]
            start = posts[index - 1]["end_ns"] if index else 0
            absolute, previous = base, base % depth
            joined = []
            wrap_corrections = 0
            for query in (q for q in queries if start < q["start_ns"] < stop):
                absolute += (query["head"] - previous) % depth
                previous = query["head"]
                sent = bisect.bisect_right(ends, query["start_ns"])
                # A coarse sampler can miss complete laps on the healthy host.
                # Bound inferred outstanding entries by the verified 2048-slot ring.
                while sent - absolute >= depth:
                    absolute += depth
                    wrap_corrections += 1
                if not 0 <= absolute < len(accepted) or sent <= absolute:
                    continue
                task = accepted[absolute]
                if task["end_ns"] >= query["start_ns"] or any(query["errors"]):
                    continue
                joined.append(
                    {
                        "head": query["head"],
                        "tail": query["tail"],
                        "head_stable": query["head"] == query["head_again"],
                        "driver_accepted_pending": sent - absolute,
                        "next_head_task_driver_lead_ms": (
                            query["start_ns"] - task["end_ns"]
                        )
                        / 1e6,
                    }
                )
            if joined:
                leads = [q["next_head_task_driver_lead_ms"] for q in joined]
                phases.append(
                    {
                        "phase": operation["phase"],
                        "complete_s": operation["complete_s"],
                        "queries_joined": len(joined),
                        "coarse_sampler_wrap_corrections": wrap_corrections,
                        "median_driver_lead_ms": statistics.median(leads),
                        "largest_driver_lead": max(
                            joined, key=lambda q: q["next_head_task_driver_lead_ms"]
                        ),
                        "accepted_pending_histogram": dict(
                            collections.Counter(
                                q["driver_accepted_pending"] for q in joined
                            )
                        ),
                    }
                )
            base += operation["descriptors"]
        result["driver"][key] = {
            "accepted_descriptors": len(accepted),
            "errors": dict(collections.Counter(r["error"] for r in calls)),
            "max_submission_call_ms": max(
                (r["end_ns"] - r["start_ns"]) / 1e6 for r in calls
            ),
            "phases": phases,
        }
    for path in sorted(root.glob("hust-zsx*/sq-calibration*")):
        key = path.relative_to(root).as_posix()
        rows = jsonl(path / "stdout.jsonl") or jsonl(path / "results.jsonl")
        totals = [row for row in rows if "wall_ns" in row]
        result["calibration"][key] = {
            "completed_result": totals[-1] if totals else None
        }
        api = jsonl(path / "api.jsonl")
        sends = [
            row
            for row in api
            if row["event"] == "hal_send" and row["sq"] == 2 and row["error"] == 0
        ]
        if sends:
            result["calibration"][key]["driver_accepted_entries"] = len(sends)
            result["calibration"][key]["driver_submit_span_ms"] = (
                sends[-1]["end_ns"] - sends[0]["start_ns"]
            ) / 1e6
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.capsule), indent=2) + "\n")


if __name__ == "__main__":
    main()
