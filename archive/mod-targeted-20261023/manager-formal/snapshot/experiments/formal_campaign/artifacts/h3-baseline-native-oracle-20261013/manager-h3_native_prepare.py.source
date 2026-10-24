"""ECPA native gate prepare barrier; requires separately owned live authority.

This executable does not adapt either baseline and does not establish formal
admission, model/Host coverage, or a runnable three-arm H3 campaign.
"""

import argparse
import hashlib
import json
import os
import signal
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

M = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(M / "src"), str(M)]
from experiments.exposure_gate.loopback_traffic import LoopbackTrafficAdapter  # noqa: E402
from vllm_hust_ext.exposure_gate import LeaseGrant, ReferenceExposureGate  # noqa: E402


def dg(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def save(p, v):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(v, sort_keys=True) + "\n")
    tmp.replace(p)


def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] == "Z":
        raise ValueError("target process is a zombie")
    return {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "session": int(fields[3]),
    }


def store_rows(path):
    with sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        state = dict(
            db.execute("select * from gate_state where singleton=1").fetchone()
        )
        rows = [
            dict(x) for x in db.execute("select * from gate_transition order by seq")
        ]
    for item in [state, *rows]:
        for k, v in item.items():
            if isinstance(v, bytes):
                item[k] = v.decode()
    return state, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("operation", choices=["launch", "state", "cleanup"])
    ap.add_argument("--spec", type=Path, required=True)
    a = ap.parse_args()
    s = json.loads(a.spec.read_text())
    root = Path(s["root"])
    root.mkdir(exist_ok=True)
    store = root / "gate.sqlite"
    barrier = root / "prepare-barrier.json"
    pidfile = root / "manager.pid"
    grant = LeaseGrant(**s["grant"])
    traffic = LoopbackTrafficAdapter(Path(s["authority_socket"]), grant)
    if a.operation == "cleanup":
        # Authority/backend remain externally owned; verify without killing them.
        p = json.loads((root / "manager-identity.json").read_text())
        try:
            live = identity(p["pid"]) == p
        except (FileNotFoundError, ProcessLookupError):
            live = False
        print(
            json.dumps(
                {
                    "coordinator_reclaimed": not live,
                    "external_authority_cleanup_required": True,
                }
            )
        )
        return
    if a.operation == "state":
        state, rows = store_rows(store)
        route = asdict(traffic.query_route())
        result = {
            "predecessor_digest": state["predecessor_digest"],
            "evidence_identity_sha256": s["evidence_identity_sha256"],
            "route": route,
            "final_route": "predecessor"
            if route["generation"] == state["predecessor_generation"]
            and route["fence"] == f"fence-{state['predecessor_generation']}"
            else "closed",
            "rollback_classification": json.loads((root / "recovery.json").read_text())[
                "rollback_classification"
            ]
            if (root / "recovery.json").exists()
            else None,
            "functional_correctness": False,
            "prepare_barrier": json.loads(barrier.read_text())
            if barrier.exists()
            else None,
            "native_journal_rows": rows,
        }
        print(json.dumps(result))
        return
    p = identity(os.getpid())
    save(root / "manager-identity.json", p)
    pidfile.write_text(str(os.getpid()) + "\n")
    gate = ReferenceExposureGate(store, traffic, clock=lambda: int(time.time()))
    if barrier.exists():
        b = json.loads(barrier.read_text())
        state, rows = store_rows(store)
        if dg(json.dumps(rows, sort_keys=True).encode()) != b["native_rows_sha256"]:
            raise ValueError("prepared journal changed before recovery")
        recovered = gate.reconcile()
        classification = gate.rollback(state["candidate_generation"])
        save(
            root / "recovery.json",
            {
                "manager": p,
                "reconciled_state": recovered.value,
                "rollback_classification": classification.value,
                "actual_route": asdict(traffic.query_route()),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
    else:
        generation = gate.stage(s["candidate"], s["predecessor"])
        original = gate._hit

        def hit(point):
            if point == "close.before_effect":
                state, rows = store_rows(store)
                pending = [
                    r
                    for r in rows
                    if r["operation"] == "close" and r["kind"] == "intent"
                ]
                receipts = [
                    r
                    for r in rows
                    if r["operation"] == "close" and r["kind"] == "receipt"
                ]
                assert len(pending) == 1 and not receipts
                if traffic.query_route().generation != state["predecessor_generation"]:
                    raise ValueError("authority changed before prepare barrier")
                save(
                    barrier,
                    {
                        "schema": "ecpa-native-close-prepare-barrier/v1",
                        "phase": "durable-close-intent-before-effect",
                        "manager": p,
                        "launch_id": s["launch_id"],
                        "journal_path": str(store),
                        "native_rows_sha256": dg(
                            json.dumps(rows, sort_keys=True).encode()
                        ),
                        "intent_sequence": pending[0]["seq"],
                        "intent_operation_id": pending[0]["operation_id"],
                        "observed_monotonic_ns": time.monotonic_ns(),
                    },
                )
                while True:
                    signal.pause()
            original(point)

        gate._hit = hit
        gate.close(generation)
    while True:
        signal.pause()


if __name__ == "__main__":
    main()
