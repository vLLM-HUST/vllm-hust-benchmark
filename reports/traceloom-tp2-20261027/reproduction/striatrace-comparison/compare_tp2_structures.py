"""Compare complete ordered lanes at independently checked replay coordinates."""

import argparse
import collections
import json
from pathlib import Path
import sqlite3
import statistics
from tp2_pilot import digest


def lane_signature(rows):
    return tuple((r["identity"], r["kind"]) for r in rows)


def shared_lanes(left, right):
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError(
            "Duplicate lane sequences do not establish unique correspondence"
        )
    shared = set(left) & set(right)
    if any(not isinstance(identity, str) for lane in shared for identity, kind in lane):
        raise ValueError("Ambiguous provider labels cannot establish a shared lane")
    return shared


def union_ns(rows):
    total = 0
    end = None
    for a, b in sorted((r["start_ns"], r["end_ns"]) for r in rows):
        if b < a:
            raise ValueError("Negative interval")
        total += max(0, b - max(a, end if end is not None else a))
        end = max(end if end is not None else b, b)
    return total


def read(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        launches = {
            r["launch_id"]: dict(r)
            for r in db.execute("SELECT * FROM traceloom_replay_cost_launch")
        }
        members = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM traceloom_replay_cost_member ORDER BY launch_id,lane_ordinal,task_ordinal"
            )
        ]
        envelopes = {
            r["replay_unit_id"]: r["end_ns"] - r["start_ns"]
            for r in db.execute(
                "SELECT replay_unit_id,MIN(start_ns) start_ns,MAX(end_ns) end_ns FROM traceloom_graph_launch GROUP BY replay_unit_id"
            )
        }
    by_launch = collections.defaultdict(lambda: collections.defaultdict(list))
    units = collections.defaultdict(dict)
    for member in members:
        by_launch[member["launch_id"]][member["lane_ordinal"]].append(member)
    if not launches or any(
        r["support_status"] != "supported" for r in launches.values()
    ):
        raise ValueError("Empty or unsupported launch population")
    for launch_id, launch in launches.items():
        coordinate = (launch["slot_role"], launch["slot_order"])
        lanes = list(by_launch[launch_id].values())
        signatures = [lane_signature(lane) for lane in lanes]
        shared_lanes(signatures, signatures)  # Reject ambiguous duplicated signatures.
        target = units[launch["replay_unit_id"]]
        if coordinate in target:
            raise ValueError("Duplicate slot in replay")
        target[coordinate] = dict(zip(signatures, lanes))
    first = next(iter(units.values()))
    shapes = {k: set(v) for k, v in first.items()}
    if any({k: set(v) for k, v in unit.items()} != shapes for unit in units.values()):
        raise ValueError("Within-profile structural population changed")
    return units, shapes, envelopes


def read_raw(path):
    audit = json.loads(path.read_text())
    if audit["unresolved"] or audit["missing_from_joint"] or audit["absent_from_raw"]:
        raise ValueError("Body membership audit differs")
    if not audit["coordinate_audit"] or audit["coordinate_audit"]["differences"]:
        raise ValueError("Independent raw coordinate audit differs")
    raw_path = Path(audit["raw_database"])
    if digest(raw_path) != audit["input_sha256"][str(raw_path)]:
        raise ValueError("Raw database changed after membership audit")
    with sqlite3.connect(raw_path.as_uri() + "?mode=ro", uri=True) as db:
        strings = dict(db.execute("SELECT id,value FROM STRING_IDS"))
        labels = collections.defaultdict(set)
        for task, name, op in db.execute(
            "SELECT globalTaskId,name,opType FROM COMPUTE_TASK_INFO"
        ):
            labels[task].add((strings.get(name, strings.get(op, str(name))), "compute"))
        compute_ids = set(labels)
        for task, name in db.execute(
            "SELECT globalTaskId,name FROM COMMUNICATION_TASK_INFO"
        ):
            if task not in compute_ids:
                labels[task].add((strings.get(name, str(name)), "communication"))
        identities = {
            rowid: labels[task]
            for rowid, task in db.execute("SELECT rowid,globalTaskId FROM TASK")
        }
    period = audit["exact_model_sequence_period"]
    units, envelopes, starts = collections.defaultdict(dict), {}, {}
    for i, body in enumerate(audit["accepted_bodies"]):
        unit_id, slot = divmod(i, period)
        # Neutral boundary/ordinal mapping. No model-layer semantic labels inferred.
        coordinate = (
            "head" if slot == 0 else "tail" if slot == period - 1 else "layer",
            slot,
        )
        lanes = collections.defaultdict(list)
        for member in body["members"]:
            names = identities[member["task_rowid"]]
            if not names or len({kind for name, kind in names}) != 1:
                raise ValueError("Unlabelled member or conflicting provider task kinds")
            if len(names) == 1:
                identity, kind = next(iter(names))
            else:
                # Replayed globalTaskId values may have multiple occurrence-labelled
                # HCCL names. Preserve the whole candidate set, never pick one name.
                # Such a lane can contribute specific costs, but cannot be shared.
                kind = next(iter(names))[1]
                identity = (
                    "ambiguous_provider_labels",
                    tuple(sorted(name for name, _ in names)),
                )
            lanes[member["stream_id"]].append(
                dict(
                    identity=identity,
                    kind=kind,
                    start_ns=member["start_ns"],
                    end_ns=member["end_ns"],
                    duration_ns=member["end_ns"] - member["start_ns"],
                )
            )
        ordered = [
            sorted(rows, key=lambda r: (r["start_ns"], r["end_ns"]))
            for rows in lanes.values()
        ]
        signatures = [lane_signature(rows) for rows in ordered]
        if len(set(signatures)) != len(signatures):
            raise ValueError("Duplicate raw lane signatures")
        units[unit_id][coordinate] = dict(zip(signatures, ordered))
        starts.setdefault(unit_id, body["start_ns"])
        envelopes[unit_id] = body["end_ns"] - starts[unit_id]
    shapes = {k: set(v) for k, v in next(iter(units.values())).items()}
    if any({k: set(v) for k, v in unit.items()} != shapes for unit in units.values()):
        raise ValueError("Raw lane signatures change within the selected population")
    return units, shapes, envelopes


def compare(left, right, reader=read):
    l_units, l_shapes, l_env = reader(left)
    r_units, r_shapes, r_env = reader(right)
    if set(l_shapes) != set(r_shapes):
        raise ValueError("Outer coordinates differ; no implicit edit alignment")
    shared = {k: shared_lanes(list(l_shapes[k]), list(r_shapes[k])) for k in l_shapes}
    if not all(shared.values()):
        raise ValueError("A coordinate lacks any uniquely corresponding complete lane")
    slots = [
        dict(
            role=k[0],
            order=k[1],
            shared_members=sum(map(len, shared[k])),
            left_only_members=sum(map(len, l_shapes[k] - shared[k])),
            right_only_members=sum(map(len, r_shapes[k] - shared[k])),
        )
        for k in sorted(shared, key=lambda k: k[1])
    ]

    def measure(units, envelopes):
        samples = []
        for unit_id, unit in units.items():
            common, specific = [], []
            for coordinate, lanes in unit.items():
                for signature, rows in lanes.items():
                    (common if signature in shared[coordinate] else specific).extend(
                        rows
                    )
            all_rows = common + specific
            samples.append(
                dict(
                    replay_unit_id=unit_id,
                    shared_task_sum_ns=sum(r["duration_ns"] for r in common),
                    specific_task_sum_ns=sum(r["duration_ns"] for r in specific),
                    total_task_sum_ns=sum(r["duration_ns"] for r in all_rows),
                    visible_union_ns=union_ns(all_rows),
                    envelope_ns=envelopes[unit_id],
                )
            )
        return dict(
            replay_units=len(samples),
            samples=samples,
            medians_ns={
                k: statistics.median(s[k] for s in samples)
                for k in samples[0]
                if k != "replay_unit_id"
            },
        )

    return dict(
        left=str(left.resolve()),
        right=str(right.resolve()),
        left_sha256=digest(left),
        right_sha256=digest(right),
        slots=slots,
        members_per_unit={
            k: sum(s[k] for s in slots)
            for k in ("shared_members", "left_only_members", "right_only_members")
        },
        left_costs=measure(l_units, l_env),
        right_costs=measure(r_units, r_env),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", type=Path, nargs=2, required=True)
    p.add_argument("--right", type=Path, nargs=2, required=True)
    p.add_argument("--raw-left", type=Path, nargs=2, required=True)
    p.add_argument("--raw-right", type=Path, nargs=2, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = dict(
        script_sha256=digest(__file__),
        scope="One engine process per backend, two paired ranks. Profile costs are descriptive, not a service speedup estimate.",
        matching="Outer slot correspondence must be audited independently. Match only unique, complete ordered identity/kind lane sequences; stream IDs are local.",
        ranks=[compare(left, right) for left, right in zip(a.left, a.right)],
        raw_ranks=[
            compare(left, right, read_raw)
            for left, right in zip(a.raw_left, a.raw_right)
        ],
    )
    result["answer_agreement"] = [
        dict(
            slots=native["slots"] == raw["slots"],
            left_samples=native["left_costs"]["samples"]
            == raw["left_costs"]["samples"],
            right_samples=native["right_costs"]["samples"]
            == raw["right_costs"]["samples"],
        )
        for native, raw in zip(result["ranks"], result["raw_ranks"])
    ]
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            [
                dict(
                    members=r["members_per_unit"],
                    left=r["left_costs"]["medians_ns"],
                    right=r["right_costs"]["medians_ns"],
                )
                for r in result["ranks"]
            ],
            indent=2,
        )
    )
    print("Raw answer agreement", result["answer_agreement"])


if __name__ == "__main__":
    main()
