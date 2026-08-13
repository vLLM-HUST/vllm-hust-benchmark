#!/usr/bin/env python3
"""poy-180 NPU runner watchdog — deployable reference daemon (issue #125).

Scans NPU 0-4 every ``--interval`` seconds, classifies every NPU-occupying
process, reclaims unauthorized ones (SIGTERM, then SIGKILL after 5s), appends
one JSONL event record per decision, and posts a deduplicated GitHub alert
rendered from the record.

This is the *deployable reference* for the watchdog host
(``host-192-168-0-6``). It never prints or uploads the raw command line; only
``cmdline_sha256`` is recorded. Use ``--dry-run`` to preview every step without
signalling any process or posting to GitHub.

Usage::

    sudo python scripts/run_npu_runner_watchdog.py --dry-run --once
    sudo python scripts/run_npu_runner_watchdog.py \\
        --interval 30 --log-dir /var/log/npu-runner-watchdog

The GitHub alert is only posted when the event is NOT suppressed by dedup and
the machine has ``gh`` authenticated against the target repository. In
``--dry-run`` mode the rendered alert is printed to stdout instead.

Acceptance verification (one clean runner + one violating process): feed a
mocked ``npu-smi`` text via ``--npu-smi-text`` and container facts via
``--facts-file`` so the classification and alert logic can be exercised
deterministically without NPU hardware::

    python scripts/run_npu_runner_watchdog.py --dry-run --once \\
        --npu-smi-text "$(cat /tmp/npu_smi.txt)" \\
        --facts-file /tmp/facts.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.watchdog_ops import (
    DEFAULT_ISSUE,
    DEFAULT_OWNER,
    DEFAULT_REPO,
    NPU_MAX,
    NPU_MIN,
    classify_determination,
    derive_cmdline_sha256,
    derive_dedup_key,
    npu_is_policy_violation,
    parse_npu_smi_processes,
    render_github_summary,
    should_alert,
    validate_event_record,
)

SIGTERM_ESCALATION_DELAY_S = 5
_CGROUP_CONTAINER_RE = re.compile(r"docker[-/]([0-9a-f]{12,64})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="poy-180 NPU runner watchdog (issue #125) reference daemon."
    )
    parser.add_argument("--interval", type=int, default=30, help="scan interval (s)")
    parser.add_argument(
        "--npu",
        type=int,
        action="append",
        choices=list(range(NPU_MIN, NPU_MAX + 1)),
        default=[],
        help="NPU to scan (repeatable; default: all 0-4)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/var/log/npu-runner-watchdog"),
        help="directory for events.jsonl and state.json",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=None,
        help="JSONL audit path (default: log-dir/events.jsonl)",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="dedup state path (default: log-dir/state.json)",
    )
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--issue", type=int, default=DEFAULT_ISSUE)
    parser.add_argument("--gh-bin", default="gh")
    parser.add_argument("--sigkill-delay", type=int, default=SIGTERM_ESCALATION_DELAY_S)
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single scan then exit (verification mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="preview decisions and alerts without signalling or posting",
    )
    parser.add_argument(
        "--npu-smi-text",
        default=None,
        help="use this text instead of running `npu-smi info -t process`",
    )
    parser.add_argument(
        "--facts-file",
        type=Path,
        default=None,
        help="JSON mapping pid -> container facts used to simulate ownership",
    )
    return parser


def load_state(path: Path) -> dict[str, dict[str, Any]]:
    """Load the previous-alerts state (dedup_key -> last alerted record)."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def save_state(path: Path, state: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def append_event(path: Path, record: Mapping[str, Any]) -> int:
    """Append one event record to the JSONL audit; return its 1-based line number."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line_number = (sum(1 for _ in path.open()) if path.exists() else 0) + 1
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return line_number


def run_npu_smi_process() -> str:
    """Return the full ``npu-smi info -t process`` text (may be empty)."""
    try:
        completed = subprocess.run(
            ["npu-smi", "info", "-t", "process"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout or ""


def container_facts(pid: int) -> dict[str, Any] | None:
    """Best-effort docker facts for a host PID.

    Reads ``/proc/<pid>/cgroup`` for the container id, then ``docker inspect``
    for the container name and ownership labels. Returns ``None`` when the
    process is not in a container or docker is unavailable.
    """
    cgroup_path = Path(f"/proc/{pid}/cgroup")
    container_id: str | None = None
    try:
        for line in cgroup_path.read_text(encoding="utf-8").splitlines():
            found = _CGROUP_CONTAINER_RE.search(line)
            if found is not None:
                container_id = found.group(1)
                break
    except (OSError, ValueError):
        pass
    if container_id is None:
        return None

    try:
        completed = subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout or "[]")
        inspect = payload[0] if payload else {}
    except (ValueError, IndexError):
        return None
    config = inspect.get("Config") or {}
    labels = config.get("Labels") or {}
    return {
        "container_name": str(inspect.get("Name") or "").lstrip("/") or None,
        "container_runner_label": labels.get("org.vllm-hust.runner"),
        "npu_physical_label": labels.get("org.vllm-hust.npu-physical"),
    }


def load_facts_file(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load pid -> container facts from a JSON file (dry-run simulation)."""
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def build_record(
    process: Mapping[str, Any],
    facts: Mapping[str, Any] | None,
    *,
    determination: str,
    action: str,
    result: str,
    owner: str,
    sequence: int,
    scan_epoch: int,
    scan_time: str,
    host: str,
) -> dict[str, Any]:
    """Build one conforming event record for a scanned process."""
    npu = int(process["npu"])
    pid = int(process["pid"])
    cmdline_sha256 = derive_cmdline_sha256(process.get("cmdline"))
    return {
        "schema_version": "npu-watchdog-event/v1",
        "schema_name": "npu-watchdog-event",
        "host": host,
        "scan_epoch": scan_epoch,
        "scan_time": scan_time,
        "npu": npu,
        "pid": pid,
        "user": process.get("user"),
        "process": process.get("process") or f"pid{pid}",
        "exe": process.get("exe"),
        "vram_mb": process.get("vram_mb"),
        "cmdline_sha256": cmdline_sha256,
        "cmdline_redacted": True,
        "determination": determination,
        "owner": owner,
        "action": action,
        "result": result,
        "recovery_status": "open",
        "dedup_key": derive_dedup_key(npu, pid, cmdline_sha256),
        "alert_suppressed": False,
        "npu4_unregistered_runner": npu_is_policy_violation(npu),
        "event_sequence": sequence,
    }


def decide_action(determination: str) -> str:
    """Return the signal to apply: ``none`` for owned processes, else ``sigterm``."""
    if determination in ("runner-job", "sibling-container"):
        return "none"
    return "sigterm"


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def reclaim(pid: int, *, dry_run: bool, sigkill_delay: int) -> str:
    """Reclaim a violating process and return the exit/cleanup result.

    Returns one of ``not-found`` / ``exited-before-action`` / ``terminated`` /
    ``killed``. In ``--dry-run`` the signals are simulated with a short delay so
    the result can be previewed without touching the process.
    """
    if dry_run:
        time.sleep(0.05)
        return "terminated"
    if not _process_alive(pid):
        return "not-found"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "not-found"
    time.sleep(sigkill_delay)
    if not _process_alive(pid):
        return "terminated"
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "exited-before-action"
    return "killed"


def post_alert(
    summary: str, *, repo: str, issue: int, gh_bin: str, dry_run: bool
) -> None:
    """Post (or preview) the rendered GitHub alert for one event."""
    if dry_run:
        print("--- would post to GitHub (dry-run) ---")
        print(summary)
        print("--- end dry-run alert ---")
        return
    subprocess.run(
        [gh_bin, "issue", "comment", str(issue), "--repo", repo, "--body", summary],
        check=False,
    )


def run_scan(
    *,
    npus: list[int],
    owner: str,
    events: Path,
    state_file: Path,
    repo: str,
    issue: int,
    gh_bin: str,
    dry_run: bool,
    sigkill_delay: int,
    host: str,
    sequence_start: int,
    npu_smi_text: str | None,
    facts_by_pid: Mapping[str, Mapping[str, Any]],
) -> int:
    """Run one scan cycle; return the number of events appended."""
    scan_epoch = int(time.time())
    # RFC3339 date-time (jsonschema "format": "date-time" requires a ":" in the
    # UTC offset, e.g. +08:00; ``strftime('%z')`` emits +0800 without it).
    scan_time = datetime.now().astimezone().isoformat(timespec="seconds")
    text = npu_smi_text if npu_smi_text is not None else run_npu_smi_process()
    processes = parse_npu_smi_processes(text)
    state = load_state(state_file)
    appended = 0

    for process in processes:
        if process["npu"] not in npus:
            continue
        pid = process["pid"]
        facts = (
            facts_by_pid.get(str(pid))
            if facts_by_pid
            else (None if dry_run else container_facts(pid))
        )
        determination = classify_determination(
            npu=process["npu"],
            container_name=facts.get("container_name") if facts else None,
            container_runner_label=(
                facts.get("container_runner_label") if facts else None
            ),
            npu_physical_label=facts.get("npu_physical_label") if facts else None,
        )
        action = decide_action(determination)
        result = (
            "no-op"
            if action == "none"
            else reclaim(pid, dry_run=dry_run, sigkill_delay=sigkill_delay)
        )
        record = build_record(
            process,
            facts,
            determination=determination,
            action=action,
            result=result,
            owner=owner,
            sequence=sequence_start + appended,
            scan_epoch=scan_epoch,
            scan_time=scan_time,
            host=host,
        )

        errors = validate_event_record(record)
        if errors:
            print(
                f"WARN: dropping event for pid {pid}: " + "; ".join(errors),
                file=sys.stderr,
            )
            continue

        alert, reason = should_alert(record, state)
        record["alert_suppressed"] = not alert
        line_number = append_event(events, record)
        appended += 1

        if alert:
            state[record["dedup_key"]] = {
                "result": record["result"],
                "recovery_status": record["recovery_status"],
            }
            summary = render_github_summary(record, event_line=line_number)
            post_alert(summary, repo=repo, issue=issue, gh_bin=gh_bin, dry_run=dry_run)
            if not dry_run:
                print(
                    f"INFO: alerted pid {pid} ({record['dedup_key']}) reason={reason}"
                )
        elif not dry_run:
            print(f"INFO: suppressed pid {pid} ({record['dedup_key']}) reason={reason}")

    save_state(state_file, state)
    return appended


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    npus = args.npu or list(range(NPU_MIN, NPU_MAX + 1))
    events = args.events or args.log_dir / "events.jsonl"
    state_file = args.state_file or args.log_dir / "state.json"
    host = os.environ.get("HOSTNAME") or os.uname().nodename or "unknown-host"
    facts_by_pid = load_facts_file(args.facts_file)

    sequence = 1
    while True:
        appended = run_scan(
            npus=npus,
            owner=args.owner,
            events=events,
            state_file=state_file,
            repo=args.repo,
            issue=args.issue,
            gh_bin=args.gh_bin,
            dry_run=args.dry_run,
            sigkill_delay=args.sigkill_delay,
            host=host,
            sequence_start=sequence,
            npu_smi_text=args.npu_smi_text,
            facts_by_pid=facts_by_pid,
        )
        sequence += appended
        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
