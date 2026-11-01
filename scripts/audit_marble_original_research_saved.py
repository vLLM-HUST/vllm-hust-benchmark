"""Read-only original Research run recovery audit; never serve queued requests."""

import argparse
import json
from pathlib import Path
import re


def process_live(identity, *, proc_root=Path("/proc")):
    if not isinstance(identity, dict):
        return False
    pid, birth = identity.get("pid"), identity.get("birth")
    if (
        type(pid) is not int
        or pid <= 0
        or not isinstance(birth, str)
        or not birth.isdigit()
    ):
        return False
    try:
        fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] not in {"Z", "X"} and fields[19] == birth


def queue_state(root, kind):
    directory = root / (kind + "-queue")
    requests = list(directory.glob("*.request.json"))
    responses = list(directory.glob("*.response.json"))
    for path in requests + responses:
        if path.is_symlink() or not re.fullmatch(
            r"[a-f0-9]{32}\.(request|response)\.json", path.name
        ):
            raise ValueError("original private queue file identity required")
    requested = {path.name.split(".", 1)[0] for path in requests}
    returned = {path.name.split(".", 1)[0] for path in responses}
    if returned - requested:
        raise ValueError("response without original request; inspect before recovery")
    return {
        "requests": len(requested),
        "responses": len(returned),
        "pending": len(requested - returned),
    }


def audit(root, launch_receipt, *, proc_root=Path("/proc")):
    root = Path(root).resolve(strict=True)
    state = json.loads((root / "state-private.json").read_text("utf-8"))
    launch = json.loads(Path(launch_receipt).read_text("utf-8"))
    if (
        state.get("source_id") != "MULTIAGENTBENCH"
        or Path(launch.get("stage", "")).resolve() != root
    ):
        raise ValueError("matching original Research state and launch binding required")
    parent = process_live(launch, proc_root=proc_root)
    child = process_live(state.get("active_child"), proc_root=proc_root)
    completed = state.get("completed", [])
    if not isinstance(completed, list) or state.get("fixed_original_rows") != [0, 1, 2]:
        raise ValueError("original fixed task completion metadata required")
    if any(
        not isinstance(entry, dict)
        or type(entry.get("original_row")) is not int
        or entry["original_row"] not in state["fixed_original_rows"]
        or type(entry.get("cli_exit_code")) is not int
        for entry in completed
    ):
        raise ValueError("original per-task terminal metadata required")
    terminal_rows = [entry["original_row"] for entry in completed]
    if len(terminal_rows) != len(set(terminal_rows)):
        raise ValueError("duplicate original task terminal row")
    all_returned = set(terminal_rows) == set(state["fixed_original_rows"])
    status = (
        "LIVE_ORIGINAL_PROCESS"
        if parent or child
        else (
            "STOPPED_ALL_ORIGINAL_CLIS_RETURNED_AUDIT_PENDING"
            if all_returned
            else "STOPPED_WITH_INCOMPLETE_ORIGINAL_PROTOCOL"
        )
    )
    return {
        "source_id": "MULTIAGENTBENCH",
        "status": status,
        "reported_historical_state": state.get("status"),
        "parent_live": parent,
        "active_child_live": child,
        "returned_original_rows": terminal_rows,
        "returned_cli_exit_codes": [entry["cli_exit_code"] for entry in completed],
        "model_queue": queue_state(root, "model"),
        "keyword_queue": queue_state(root, "keyword"),
        "live_consumer_exists": parent or child,
        "request_scoped_consumer_identity_verified": False,
        "pending_requests_may_be_served": False,
        "recovery_boundary": "No inference authorized by this audit. Pending requests of stopped consumers must not be served; uncertain prior inference requires independent saved-evidence review.",
        "new_model_calls": 0,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_root, args.launch_receipt), indent=2))


if __name__ == "__main__":
    main()
