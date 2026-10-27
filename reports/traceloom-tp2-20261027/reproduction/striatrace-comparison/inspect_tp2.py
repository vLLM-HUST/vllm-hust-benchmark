"""Summarize native structure without assuming historical model/role counts."""

import argparse
import collections
import json
from pathlib import Path
import sqlite3
from tp2_pilot import digest


def inspect(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        result = dict(path=str(path.resolve()), sha256=digest(path))
        result["partition"] = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM traceloom_v_exact_replay_partition_status"
            )
        ]
        result["regions"] = [
            dict(r)
            for r in db.execute(
                "SELECT status,observed_launch_count,expected_launch_count,COUNT(*) AS n "
                "FROM traceloom_aclgraph_reconstruction_region GROUP BY 1,2,3"
            )
        ]
        result["cost_issues"] = db.execute(
            "SELECT COUNT(*) FROM traceloom_replay_cost_issue"
        ).fetchone()[0]
        launches = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM traceloom_replay_cost_launch ORDER BY replay_unit_id,member_order"
            )
        ]
        members = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM traceloom_replay_cost_member ORDER BY launch_id,lane_ordinal,task_ordinal"
            )
        ]
        by_launch = collections.defaultdict(lambda: collections.defaultdict(list))
        for row in members:
            by_launch[row["launch_id"]][row["lane_ordinal"]].append(row)
        shapes = collections.defaultdict(list)
        units = collections.defaultdict(list)
        for row in launches:
            # Lane signature ignores unstable stream IDs, but retains the entire ordered identity/kind sequence.
            lanes = sorted(
                tuple((m["identity"], m["kind"]) for m in lane)
                for lane in by_launch[row["launch_id"]].values()
            )
            shapes[(row["slot_role"], row["slot_order"])].append(tuple(lanes))
            units[row["replay_unit_id"]].append((row["slot_role"], row["slot_order"]))
        result.update(
            replay_units=len(units),
            launches=len(launches),
            members=len(members),
            unit_shapes=[
                dict(coordinates=k, count=v)
                for k, v in collections.Counter(
                    tuple(u) for u in units.values()
                ).items()
            ],
            coordinates=[
                dict(
                    role=k[0],
                    order=k[1],
                    occurrences=len(v),
                    distinct_lane_sets=len(set(v)),
                    lanes=v[0],
                )
                for k, v in sorted(shapes.items())
            ],
            member_identities=dict(collections.Counter(m["identity"] for m in members)),
            supported_launches=sum(
                launch["support_status"] == "supported" for launch in launches
            ),
        )
        result["source_members"] = db.execute(
            "SELECT COUNT(DISTINCT m.member_id) FROM traceloom_replay_cost_member m "
            "JOIN traceloom_event_source s ON s.event_id=m.event_id AND s.db_idx=m.db_idx AND s.device_id=m.device_id "
            "WHERE s.source_table='TASK'"
        ).fetchone()[0]
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("databases", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = dict(
        script_sha256=digest(__file__), workers=[inspect(path) for path in a.databases]
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "path",
                        "regions",
                        "replay_units",
                        "launches",
                        "members",
                        "source_members",
                        "cost_issues",
                    )
                }
                for r in result["workers"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
