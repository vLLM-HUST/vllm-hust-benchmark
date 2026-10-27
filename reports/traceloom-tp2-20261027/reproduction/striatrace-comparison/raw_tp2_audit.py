"""Recover TP2 launch bodies from original provider controls, independently of AugDB.

Extends the existing raw completion-control implementation to provider-labelled
communication tasks. Native source rows, not TraceLoom names, are compared.
"""

import argparse
import collections
import json
from pathlib import Path
import sqlite3
import sys
import time
from tp2_pilot import digest

BASELINE = Path(__file__).resolve().parent.parent / "optimization-guidance-v2"
sys.path.insert(0, str(BASELINE))
from raw_replay_baseline import reconstruct  # noqa: E402 -- sibling reproduction directory


def exact_period(values):
    """Return the shortest full-sequence period repeated at least three times."""
    for width in range(1, len(values) // 3 + 1):
        if len(values) % width == 0 and all(
            v == values[i % width] for i, v in enumerate(values)
        ):
            return width
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--streams", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    start = time.perf_counter()
    with sqlite3.connect(a.raw.resolve().as_uri() + "?mode=ro", uri=True) as db:
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
        tables = {
            r[0]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        comm = set()
        if "COMMUNICATION_TASK_INFO" in tables:
            comm = {
                r[0]
                for r in db.execute(
                    "SELECT globalTaskId FROM COMMUNICATION_TASK_INFO WHERE name IS NOT NULL"
                )
            }
        semantic.update(comm)
    for row in tasks:
        row["label"] = strings.get(row["taskType"], str(row["taskType"]))
    streams = collections.defaultdict(set)
    with sqlite3.connect(a.streams.resolve().as_uri() + "?mode=ro", uri=True) as db:
        columns = {r[1] for r in db.execute("PRAGMA table_info(CaptureStreamInfo)")}
        column = "stream_id" if "stream_id" in columns else "model_stream_id"
        for device, model, stream in db.execute(
            f"SELECT device_id,model_id,{column} FROM CaptureStreamInfo"
        ):
            if model is not None and 0 <= model < 4294967295:
                streams[(device, model)].add(stream)
    raw = reconstruct(tasks, semantic, streams, [])
    raw_seconds = time.perf_counter() - start
    expected = {
        (r["device_id"], r["start_ns"], m["task_rowid"])
        for r in raw["accepted"]
        for m in r["members"]
    }
    with sqlite3.connect(a.analysis.resolve().as_uri() + "?mode=ro", uri=True) as db:
        observed = set(
            db.execute(
                "SELECT m.device_id,l.start_ns,CAST(s.source_key AS INTEGER) "
                "FROM traceloom_replay_cost_member m JOIN traceloom_graph_launch l "
                "ON l.launch_id=m.launch_id AND l.db_idx=m.db_idx AND l.device_id=m.device_id "
                "JOIN traceloom_event_source s ON s.event_id=m.event_id AND s.db_idx=m.db_idx AND s.device_id=m.device_id "
                "WHERE s.source_table='TASK' AND s.source_role='primary'"
            )
        )
        joint_launches = list(
            db.execute(
                "SELECT device_id,start_ns,replay_unit_id,slot_order FROM traceloom_graph_launch "
                "ORDER BY device_id,start_ns"
            )
        )
    period = (
        exact_period([r["model_id"] for r in raw["accepted"]])
        if not raw["unresolved"]
        else None
    )
    coordinate_audit = None
    if period:
        raw_coordinates = {
            (r["device_id"], r["start_ns"]): (
                raw["accepted"][i - i % period]["start_ns"],
                i % period,
            )
            for i, r in enumerate(raw["accepted"])
        }
        first = {}
        for device, launch_start, unit, slot in joint_launches:
            first.setdefault((device, unit), launch_start)
        joint_coordinates = {
            (device, launch_start): (first[(device, unit)], slot)
            for device, launch_start, unit, slot in joint_launches
        }
        keys = set(raw_coordinates) | set(joint_coordinates)
        coordinate_audit = dict(
            raw=len(raw_coordinates),
            joint=len(joint_coordinates),
            equal=sum(raw_coordinates.get(k) == joint_coordinates.get(k) for k in keys),
            differences=[
                dict(
                    device=k[0],
                    start_ns=k[1],
                    raw=raw_coordinates.get(k),
                    joint=joint_coordinates.get(k),
                )
                for k in sorted(keys)
                if raw_coordinates.get(k) != joint_coordinates.get(k)
            ],
            interpretation="Raw full-sequence period and ordinal against recovered native unit and slot. "
            "Agreement on this regular input does not test interrupted or ambiguous sequences.",
        )
    result = dict(
        raw_database=str(a.raw.resolve()),
        analysis=str(a.analysis.resolve()),
        input_sha256={str(x): digest(x) for x in (a.raw, a.streams, a.analysis)},
        script_sha256=digest(__file__),
        reconstruction_sha256=digest(BASELINE / "raw_replay_baseline.py"),
        native_launches=raw["native_launches"],
        accepted=len(raw["accepted"]),
        unresolved=raw["unresolved"],
        raw_members=len(expected),
        joint_members=len(observed),
        missing_from_joint=sorted(expected - observed),
        absent_from_raw=sorted(observed - expected),
        exact_model_sequence_period=period,
        repeated_units=len(raw["accepted"]) // period if period else None,
        coordinate_audit=coordinate_audit,
        raw_reconstruction_seconds=raw_seconds,
        provider_communication_task_ids=len(comm),
        completion_tolerance_ns=10000,
        scope="Same provider contract, independently implemented. Full-sequence period only; no approximate matching or timestamp-only body assignment.",
        accepted_bodies=raw["accepted"],
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "accepted_bodies",
                    "unresolved",
                    "missing_from_joint",
                    "absent_from_raw",
                )
            },
            indent=2,
        )
    )
    print(
        "unresolved",
        len(raw["unresolved"]),
        "missing",
        len(expected - observed),
        "extra",
        len(observed - expected),
    )


if __name__ == "__main__":
    main()
