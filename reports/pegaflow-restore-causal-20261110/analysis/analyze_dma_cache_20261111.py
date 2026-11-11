"""Join request, connector, scheduler and native timings from real 14B replay."""

import collections
import csv
import datetime
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiment-results/long-pd-20261109"


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def stable(key):
    match = re.match(r"^cmpl-(.+)-\d+-[0-9a-f]+$", key)
    return match[1] if match else key


def stats(values):
    if not values:
        return None
    values = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "p99": values[math.ceil(0.99 * len(values)) - 1],
        "max": values[-1],
    }


def main():
    raw = ROOT / "raw"
    requests = {}
    for path in (raw / "runs").glob("*/requests.jsonl"):
        for row in records(path):
            if row["status"] == "ok":
                requests[row["id"]] = row | {"condition": path.parent.name}
    telemetry = collections.defaultdict(list)
    for path in (raw / "telemetry").rglob("*.jsonl"):
        for row in records(path):
            keys = row.get("request_ids", [row.get("request_id", "")])
            for key in keys:
                telemetry[stable(key)].append(row)
    first_computed = {}
    prefill = collections.Counter()
    for path in (raw / "traces").rglob("*.jsonl"):
        node = path.parent.name
        for row in records(path):
            if row["event"] != "schedule":
                continue
            for key, count in row["scheduled"].items():
                key_id = stable(key)
                if key_id not in requests or count <= 0:
                    continue
                state = row["requests"][key]
                # schedule() has already advanced num_computed_tokens.
                before = state["computed"] - count
                first_computed.setdefault((node, key_id), before)
                prefill[node, key_id] += min(count, max(0, state["prompt"] - before))
    rdma = collections.defaultdict(list)
    loads = []
    pending_stage = {}
    rdma_pattern = re.compile(
        r"RDMA fetch summary: req_id=(\S+) remote=(\S+) blocks=(\d+)/(\d+).*bytes_mib=([\d.]+) total_ms=([\d.]+)"
    )
    load_pattern = re.compile(
        r"Load task completed: layers=(\d+) blocks=(\d+) copies=(\d+) bytes=(\d+) elapsed_ms=([\d.]+)"
    )
    stage_pattern = re.compile(
        r"Load stage timing: .*build_us=(\d+) api_us=(\d+) sync_us=(\d+)"
    )
    for path in raw.rglob("D*-engine.log"):
        node = path.stem.split("-")[0]
        for line in path.read_text(encoding="utf-8").splitlines():
            match = rdma_pattern.search(line)
            if match:
                rdma[stable(match[1])].append(
                    {
                        "remote": match[2],
                        "blocks": int(match[3]),
                        "total_blocks": int(match[4]),
                        "MiB": float(match[5]),
                        "seconds": float(match[6]) / 1000,
                    }
                )
            stage = stage_pattern.search(line)
            if stage:
                pending_stage[node] = {
                    "build_s": int(stage[1]) / 1e6,
                    "api_s": int(stage[2]) / 1e6,
                    "sync_s": int(stage[3]) / 1e6,
                }
            match = load_pattern.search(line)
            if match:
                stamp = datetime.datetime.fromisoformat(line.split()[0]).timestamp()
                loads.append(
                    {
                        "node": node,
                        "epoch_s": stamp,
                        "blocks": int(match[2]),
                        "bytes": int(match[4]),
                        "elapsed_s": float(match[5]) / 1000,
                        **pending_stage.pop(node, {}),
                    }
                )
    joined = []
    for key, row in requests.items():
        events = telemetry[key]
        queries = [x for x in events if x["event"] == "cache_query_ready"]
        restores = [x for x in events if x["event"] == "load_completed"]
        joined.append(
            {
                "id": key,
                "condition": row["condition"],
                "P": row["P"],
                "D": row["D"],
                "prompt_tokens": row["prompt_tokens"],
                "output_tokens": row.get("D_usage", {}).get("completion_tokens"),
                "TTFT_s": row["first_s"] - row["ready_s"],
                "D_to_first_s": row["first_s"] - row["D_submit_s"],
                "P_initial_cached_tokens": first_computed.get((row["P"], key)),
                "D_initial_cached_tokens": first_computed.get((row["D"], key)),
                "P_prefill_tokens": prefill[row["P"], key]
                if (row["P"], key) in first_computed
                else None,
                "D_prefill_tokens": prefill[row["D"], key]
                if (row["D"], key) in first_computed
                else None,
                "discovery_prefetch_wait_s": sum(x["wait_ms"] / 1000 for x in queries),
                "restore_completion_wait_s": sum(
                    x["duration_ms"] / 1000 for x in restores
                ),
                "remote_payload_MiB": sum(x["MiB"] for x in rdma[key]),
                "cross_node_payload_MiB": sum(
                    x["MiB"]
                    for x in rdma[key]
                    if x["remote"].startswith("192.168.0.4:")
                ),
                "same_host_payload_MiB": sum(
                    x["MiB"]
                    for x in rdma[key]
                    if x["remote"].startswith("192.168.0.2:")
                ),
                "remote_fetch_sum_s": sum(x["seconds"] for x in rdma[key]),
                "remote_fetches": len(rdma[key]),
                "load_completions": len(restores),
            }
        )
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    if joined:
        with (out / "cache-and-transfer.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(joined[0]))
            writer.writeheader()
            writer.writerows(joined)
    summaries = []
    for label in sorted({r["condition"] for r in joined}):
        selected = [r for r in joined if r["condition"] == label]
        summaries.append(
            {
                "label": label,
                "requests": len(selected),
                "P_prefill_computed_tokens_observed": sum(
                    x["P_prefill_tokens"] or 0 for x in selected
                ),
                "D_prefill_computed_tokens_observed": sum(
                    x["D_prefill_tokens"] or 0 for x in selected
                ),
                "P_trace_coverage": sum(
                    x["P_prefill_tokens"] is not None for x in selected
                ),
                "D_trace_coverage": sum(
                    x["D_prefill_tokens"] is not None for x in selected
                ),
                "total_prompt_tokens": sum(x["prompt_tokens"] for x in selected),
                "remote_payload_GiB": sum(x["remote_payload_MiB"] for x in selected)
                / 1024,
                "cross_node_payload_GiB": sum(
                    x["cross_node_payload_MiB"] for x in selected
                )
                / 1024,
                "same_host_payload_GiB": sum(
                    x["same_host_payload_MiB"] for x in selected
                )
                / 1024,
                **{
                    name: stats([x[name] for x in selected])
                    for name in [
                        "discovery_prefetch_wait_s",
                        "restore_completion_wait_s",
                        "remote_fetch_sum_s",
                    ]
                },
            }
        )
    (out / "transfer-summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    (out / "native-loads.json").write_text(
        json.dumps(loads, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    ROOT = parser.parse_args().root
    main()
