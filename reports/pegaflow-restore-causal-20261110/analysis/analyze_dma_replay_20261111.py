"""Analyze frozen model replay and mutable-source acceptance evidence."""

import argparse
import json
import math
import statistics
import tarfile
from collections import Counter
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def stats(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(
        n=len(values),
        mean=statistics.mean(values),
        p50=statistics.median(values),
        p95=ordered[math.ceil(0.95 * len(values)) - 1],
        p99=ordered[math.ceil(0.99 * len(values)) - 1],
        max=ordered[-1],
    )


STAGES = {
    "P_queue_s": ("P_slot_s", "ready_s"),
    "source_budget_wait_s": ("P_submit_s", "P_slot_s"),
    "P_service_s": ("P_response_s", "P_submit_s"),
    "publication_wait_s": ("visible_s", "P_response_s"),
    "D_queue_s": ("D_submit_s", "visible_s"),
    "D_to_first_s": ("first_s", "D_submit_s"),
}


def request_identity(record):
    return tuple(
        record[key]
        for key in (
            "session",
            "turn",
            "prompt_tokens",
            "output_tokens_target",
            "P",
            "D",
        )
    )


def staging_stats(calls):
    return dict(
        batches=len(calls),
        bytes=sum(r["bytes"] for r in calls),
        descriptors=sum(r["copies"] for r in calls),
        errors=dict(Counter(r["error"] for r in calls)),
        cpu_staging_sum_s=sum(r["cpu_copy_ns"] for r in calls) / 1e9,
        slot_wait_events_sum_s=sum(r["slot_wait_and_event_ns"] for r in calls) / 1e9,
        batch_submit_sum_s=sum(r["batch_submit_ns"] for r in calls) / 1e9,
        source_addresses=len({r["src_first"] for r in calls}),
        staging_addresses=len({r["staging_first"] for r in calls}),
        destination_addresses=len({r["dst_first"] for r in calls}),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    root = args.root
    raw = root / "raw"
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    result = {"runs": [], "acceptance": []}
    native_staging = {}
    for path in (raw / "services").glob("D*-dma-stage.jsonl"):
        native_staging[path.name.split("-", 1)[0]] = rows(path)
    resource_file = raw / "resources/D.jsonl"
    clock_epoch_offset = None
    if resource_file.exists():
        offsets = []
        with resource_file.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                offsets.append(record["epoch_ns"] - record["monotonic_ns"])
        if offsets:
            clock_epoch_offset = statistics.median(offsets)
    acceptance = root / "acceptance-20261111.tar.gz"
    with tarfile.open(acceptance) as archive:
        for host in ("hust-zsx-81", "hust-zsx-91"):
            name = f"mutable-source-acceptance/{host}/operations.jsonl"
            records = [
                json.loads(x) for x in archive.extractfile(name).read().splitlines()
            ]
            restored = [r for r in records if r["event"] == "restore"]
            assert len(restored) == 7
            assert all(
                r["all_tags"] and r["verified_tag_count"] == 1536 for r in restored
            )
            versions = [r["verified_tags"][0][0] for r in restored]
            assert len(set(versions)) == 7
            result["acceptance"].append(
                dict(
                    host=host,
                    restores=7,
                    tag_checks=7 * 1536,
                    source_versions=versions,
                    complete_s=stats([r["complete_s"] for r in restored]),
                )
            )
    old = rows(args.baseline) if args.baseline else []
    for path in sorted((raw / "runs").glob("*/requests.jsonl")):
        records = rows(path)
        good = [r for r in records if r["status"] == "ok"]
        manifest = json.loads(path.with_name("manifest.json").read_text())
        summary_path = path.with_name("summary.json")
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        stages = {}
        for key, (end, start) in STAGES.items():
            stages[key] = stats([r[end] - r[start] for r in good])
        for record in good:
            total = sum(record[end] - record[start] for end, start in STAGES.values())
            assert abs(total - (record["first_s"] - record["ready_s"])) < 1e-6
        run = dict(
            label=path.parent.name,
            planned=manifest["requests"],
            counts=dict(Counter(r["status"] for r in records)),
            complete=len(good) == manifest["requests"],
            span_s=summary.get("span_s"),
            completed_rps=summary.get("completed_rps"),
            output_tokens_s=summary.get("output_tokens_s"),
            TTFT_s=stats([r["first_s"] - r["ready_s"] for r in good]),
            stages=stages,
            errors=[r for r in records if r["status"] == "error"],
        )
        if clock_epoch_offset is not None and summary.get("span_s") is not None:
            clock_file = path.with_name("clock-calibration.json")
            samples = json.loads(clock_file.read_text()).get("D", [])
            valid = [r for r in samples if "offset_ns" in r]
            offset = min(valid, key=lambda r: r["rtt_ns"])["offset_ns"] if valid else 0
            start = manifest["origin_epoch_ns"] + offset
            end = start + summary["span_s"] * 1e9
            run["native_staging"] = {
                node: staging_stats(
                    [
                        r
                        for r in values
                        if r["event"] == "dma_stage_h2d"
                        and start <= r["start_ns"] + clock_epoch_offset <= end
                    ]
                )
                for node, values in native_staging.items()
            }
        if (
            old
            and path.parent.name.startswith("random-")
            and not manifest.get("output_cap", 0)
        ):
            previous = {request_identity(r): r for r in old if r["status"] == "ok"}
            pairs = [
                (previous[request_identity(r)], r)
                for r in good
                if request_identity(r) in previous
            ]
            run["historical_common_successful_cohort"] = dict(
                n=len(pairs),
                historical_counts=dict(Counter(r["status"] for r in old)),
                historical_TTFT_s=stats(
                    [a["first_s"] - a["ready_s"] for a, _ in pairs]
                ),
                current_TTFT_s=stats([b["first_s"] - b["ready_s"] for _, b in pairs]),
                interpretation="Conditional historical cohort; different dates and runtime storage; not a same-day causal speedup estimate",
            )
        result["runs"].append(run)
    stage_rows = []
    for path in (raw / "services").glob("D*-dma-stage.jsonl"):
        records = rows(path)
        calls = [r for r in records if r["event"] == "dma_stage_h2d"]
        stage_rows.append(
            dict(
                node=path.name.split("-", 1)[0],
                **staging_stats(calls),
                initialize=[r for r in records if r["event"] == "dma_stage_initialize"],
                scope="All route probes and formal runs in this process, summed host-side calls",
            )
        )
    result["dma_staging"] = stage_rows
    (out / "model-replay-summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
