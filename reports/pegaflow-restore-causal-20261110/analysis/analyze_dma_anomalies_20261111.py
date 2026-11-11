"""Separate request, connector, native-copy and staging waits in frozen replay."""

import argparse
import datetime
import json
import statistics
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def cpu_stat(text):
    return {key: int(value) for key, value in (x.split() for x in text.splitlines())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    raw = root / "raw"
    resources = []
    offsets = []
    for row in rows(raw / "resources/D.jsonl"):
        offsets.append(row["epoch_ns"] - row["monotonic_ns"])
        resources.append({key: row[key] for key in ("epoch_ns", "cgroup")})
    mono_offset = statistics.median(offsets) / 1e9
    native = json.loads((root / "analysis/native-loads.json").read_text())
    staging = {
        node: [
            r
            for r in rows(raw / f"services/{node}-dma-stage.jsonl")
            if r["event"] == "dma_stage_h2d"
        ]
        for node in ("D0", "D1")
    }
    telemetry = {}
    for node in ("D0", "D1"):
        telemetry[node] = [
            r for path in (raw / "telemetry" / node).glob("*.jsonl") for r in rows(path)
        ]
    output = {"native_by_condition": [], "focus": []}
    for directory in sorted((raw / "runs").iterdir()):
        if directory.name.startswith("smoke"):
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        clocks = json.loads((directory / "clock-calibration.json").read_text())["D"]
        clock = min([r for r in clocks if "offset_ns" in r], key=lambda r: r["rtt_ns"])
        start = (manifest["origin_epoch_ns"] + clock["offset_ns"]) / 1e9
        end = start + summary["span_s"]
        selected = [r for r in native if start <= r["epoch_s"] <= end]
        output["native_by_condition"].append(
            dict(
                label=directory.name,
                jobs=len(selected),
                bytes=sum(r["bytes"] for r in selected),
                max_s=max(r["elapsed_s"] for r in selected),
                over_120s=sum(r["elapsed_s"] > 120 for r in selected),
            )
        )
        warnings = []
        for path in (raw / "services").glob("D*-engine.log"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if " WARN " not in line and " ERROR " not in line:
                    continue
                epoch = datetime.datetime.fromisoformat(line.split()[0]).timestamp()
                if start <= epoch <= end:
                    warnings.append(
                        dict(node=path.name.split("-", 1)[0], epoch_s=epoch, line=line)
                    )
        output["native_by_condition"][-1]["native_warnings"] = warnings
        output["native_by_condition"][-1]["pool_exhausted_records"] = sum(
            "Pinned memory pool exhausted" in r["line"] for r in warnings
        )
        for request in rows(directory / "requests.jsonl"):
            suffix = request["id"].rsplit("-", 1)[-1]
            focus = ("repeat" in directory.name and suffix in ("0110", "0185")) or (
                "6sessions" in directory.name and suffix in ("0214", "0238")
            )
            if not focus:
                continue
            node = request["D"]
            events = [
                r
                for r in telemetry[node]
                if any(
                    request["id"] in key
                    for key in r.get("request_ids", [r.get("request_id", "")])
                )
            ]
            begins = [r for r in events if r["event"] == "load_start"]
            completes = [r for r in events if r["event"] == "load_completed"]
            if not begins or not completes:
                continue
            begin, complete = begins[0], completes[0]
            begin_s, complete_s = begin["unix_ns"] / 1e9, complete["unix_ns"] / 1e9
            jobs = [
                r
                for r in selected
                if r["node"] == node
                and r["blocks"] == complete["blocks"]
                and begin_s <= r["epoch_s"] <= complete_s
            ]
            job = min(jobs, key=lambda r: complete_s - r["epoch_s"]) if jobs else None
            overlapping_syncs = []
            for row in rows(raw / f"services/{node}-copy-api.jsonl"):
                if row["event"] != "sync":
                    continue
                api_start = row["start_ns"] / 1e9 + mono_offset
                api_end = row["end_ns"] / 1e9 + mono_offset
                if (
                    api_end - api_start >= 1
                    and api_start <= complete_s
                    and api_end >= begin_s
                ):
                    overlapping_syncs.append(row)
            schedules = []
            for trace in (raw / "traces" / node).glob("*.jsonl"):
                schedules.extend(
                    r
                    for r in rows(trace)
                    if r["event"] == "schedule"
                    and begin_s <= r["epoch_ns"] / 1e9 <= complete_s + 1
                )
            schedules.sort(key=lambda r: r["epoch_ns"])
            gaps = [
                dict(
                    start_ns=a["epoch_ns"],
                    end_ns=b["epoch_ns"],
                    duration_s=(b["epoch_ns"] - a["epoch_ns"]) / 1e9,
                    preceding_running=a["running"],
                )
                for a, b in zip(schedules, schedules[1:])
                if b["epoch_ns"] - a["epoch_ns"] > 1e9
            ]
            calls = (
                []
                if job is None
                else [
                    r
                    for r in staging[node]
                    if job["epoch_s"] - job["elapsed_s"] - 0.001
                    <= r["start_ns"] / 1e9 + mono_offset
                    <= job["epoch_s"]
                ]
            )
            window = [
                r for r in resources if begin_s <= r["epoch_ns"] / 1e9 <= complete_s
            ]
            delta = {}
            if len(window) >= 2:
                first, last = (
                    cpu_stat(r["cgroup"]["cpu.stat"]) for r in (window[0], window[-1])
                )
                delta = {key: last[key] - first[key] for key in first}
            output["focus"].append(
                dict(
                    request=request,
                    connector_events=[
                        r for r in events if r["event"].startswith("load")
                    ],
                    D_submit_to_load_start_s=begin_s - (start + request["D_submit_s"]),
                    callback_s=complete["duration_ms"] / 1000,
                    native=job,
                    staging=dict(
                        batches=len(calls),
                        bytes=sum(r["bytes"] for r in calls),
                        matches_native_bytes=bool(
                            job and sum(r["bytes"] for r in calls) == job["bytes"]
                        ),
                        cpu_copy_s=sum(r["cpu_copy_ns"] for r in calls) / 1e9,
                        slot_wait_and_events_s=sum(
                            r["slot_wait_and_event_ns"] for r in calls
                        )
                        / 1e9,
                        batch_submit_s=sum(r["batch_submit_ns"] for r in calls) / 1e9,
                    ),
                    cpu_cgroup_delta=delta,
                    cgroup_sample_count=len(window),
                    overlapping_native_syncs=overlapping_syncs,
                    scheduler_gaps_over_1s=gaps,
                    interpretation="Event section includes synchronize, destroy and create; callback includes queue and polling.",
                )
            )
    path = root / "analysis/model-replay-anomalies.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
