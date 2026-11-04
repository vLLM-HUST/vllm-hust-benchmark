"""Independent dense-TP1 replay membership from native controls and capture metadata.

Uses raw TASK/COMPUTE_TASK_INFO/CaptureStreamInfo and the raw-provider context
baseline. No TraceLoom relation is an input. Ambiguous completion controls are
retained as unresolved, not assigned by nearest timestamp alone.
"""

import argparse
import bisect
import collections
import json
from pathlib import Path
import sqlite3
import time
from common import sha256, write_json


def reconstruct(tasks, semantic_ids, captured_streams, assignments, tolerance_ns=10000):
    launches = sorted(
        [t for t in tasks if t["label"] == "MODEL_EXECUTE"], key=lambda t: t["startNs"]
    )
    waits = collections.defaultdict(list)
    records = collections.defaultdict(list)
    body = collections.defaultdict(list)
    for task in tasks:
        if task["label"] == "NOTIFY_WAIT":
            waits[(task["deviceId"], task["streamId"], task["connectionId"])].append(
                task
            )
        if (
            task["label"] == "NOTIFY_RECORD"
            and (task["deviceId"], task["modelId"]) in captured_streams
        ):
            records[task["deviceId"]].append(task)
        if task["globalTaskId"] in semantic_ids:
            body[(task["deviceId"], task["modelId"], task["streamId"])].append(task)
    starts = {}
    for key, rows in body.items():
        rows.sort(key=lambda r: (r["startNs"], r["endNs"], r["source_rowid"]))
        starts[key] = [r["startNs"] for r in rows]
    contexts = {
        r["source_rowid"]: r for r in assignments if r["task_type"] == "MODEL_EXECUTE"
    }
    claimed_waits = set()
    claimed_records = set()
    accepted = []
    unresolved = []
    for launch in launches:
        key = (launch["deviceId"], launch["streamId"], launch["connectionId"])
        matches = [
            r
            for r in waits[key]
            if r["startNs"] >= launch["endNs"]
            and r["source_rowid"] not in claimed_waits
        ]
        if len(matches) != 1:
            unresolved.append(
                dict(
                    launch_rowid=launch["source_rowid"],
                    reason="missing_or_ambiguous_wait",
                )
            )
            continue
        wait = matches[0]
        matches = [
            r
            for r in records[launch["deviceId"]]
            if abs(r["endNs"] - wait["endNs"]) <= tolerance_ns
            and r["source_rowid"] not in claimed_records
        ]
        if len(matches) != 1:
            unresolved.append(
                dict(
                    launch_rowid=launch["source_rowid"],
                    reason="missing_or_ambiguous_completion",
                )
            )
            continue
        record = matches[0]
        model = record["modelId"]
        streams = captured_streams[(launch["deviceId"], model)]
        if record["streamId"] not in streams:
            unresolved.append(
                dict(
                    launch_rowid=launch["source_rowid"],
                    reason="completion_not_on_captured_stream",
                )
            )
            continue
        claimed_waits.add(wait["source_rowid"])
        claimed_records.add(record["source_rowid"])
        end = max(launch["endNs"], wait["endNs"], record["endNs"])
        members = []
        for stream in sorted(streams):
            key = (launch["deviceId"], model, stream)
            rows = body[key]
            times = starts.get(key, [])
            lo = bisect.bisect_left(times, launch["startNs"])
            hi = bisect.bisect_right(times, end)
            members.extend(r for r in rows[lo:hi] if r["endNs"] <= end)
        context = contexts.get(launch["source_rowid"])
        accepted.append(
            dict(
                launch_rowid=launch["source_rowid"],
                wait_rowid=wait["source_rowid"],
                completion_rowid=record["source_rowid"],
                model_id=model,
                device_id=launch["deviceId"],
                captured_streams=sorted(streams),
                start_ns=launch["startNs"],
                end_ns=end,
                context={k: context[k] for k in ("run_id", "step_id", "execution_id")}
                if context
                else None,
                members=[
                    dict(
                        task_rowid=r["source_rowid"],
                        start_ns=r["startNs"],
                        end_ns=r["endNs"],
                        stream_id=r["streamId"],
                    )
                    for r in members
                ],
            )
        )
    return dict(native_launches=len(launches), accepted=accepted, unresolved=unresolved)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("database", type=Path)
    p.add_argument("--stream-database", type=Path, required=True)
    p.add_argument("--context-baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    started = time.perf_counter()
    context = json.loads(a.context_baseline.read_text())
    if context["raw_database_sha256"] != sha256(a.database):
        raise ValueError("raw source differs")
    with sqlite3.connect(a.database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        strings = dict(db.execute("SELECT id,value FROM STRING_IDS"))
        tasks = [
            dict(r) for r in db.execute("SELECT rowid AS source_rowid,* FROM TASK")
        ]
        semantic = {
            r[0]
            for r in db.execute(
                "SELECT globalTaskId FROM COMPUTE_TASK_INFO WHERE name IS NOT NULL OR opType IS NOT NULL"
            )
        }
        if db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='COMMUNICATION_TASK_INFO'"
        ).fetchone()[0]:
            if db.execute("SELECT COUNT(*) FROM COMMUNICATION_TASK_INFO").fetchone()[0]:
                raise ValueError(
                    "this baseline is scoped to dense TP1, not HCCL body reconstruction"
                )
    for task in tasks:
        task["label"] = strings.get(task["taskType"], str(task["taskType"]))
    streams = collections.defaultdict(set)
    with sqlite3.connect(
        a.stream_database.resolve().as_uri() + "?mode=ro", uri=True
    ) as db:
        columns = {r[1] for r in db.execute("PRAGMA table_info(CaptureStreamInfo)")}
        column = "stream_id" if "stream_id" in columns else "model_stream_id"
        for device, model, stream in db.execute(
            f"SELECT device_id,model_id,{column} FROM CaptureStreamInfo"
        ):
            if model is not None and 0 <= model < 4294967295:
                streams[(device, model)].add(stream)
    result = reconstruct(tasks, semantic, streams, context["assignments"])
    result.update(
        elapsed_seconds=time.perf_counter() - started,
        script_sha256=sha256(__file__),
        raw_database_sha256=sha256(a.database),
        stream_database_sha256=sha256(a.stream_database),
        context_baseline_sha256=sha256(a.context_baseline),
        completion_tolerance_ns=10000,
        scope="Dense TP1 compute-identified members, explicit captured model/stream set and unique completion controls. "
        "This is a separate implementation of the declared native-evidence contract, not an independent hardware oracle.",
    )
    write_json(a.output, result)
    print(
        json.dumps(
            dict(
                native_launches=result["native_launches"],
                accepted=len(result["accepted"]),
                unresolved=len(result["unresolved"]),
                members=sum(len(r["members"]) for r in result["accepted"]),
            )
        )
    )


if __name__ == "__main__":
    main()
