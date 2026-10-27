"""Run unchanged official Mooncake storage replay with explicit private I/O budget.

Linux server only; no model calls, invented prompts, retries or cleanup.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from prepare_mooncake_storage_plan import AUTHOR_FILES, ROOT, build_plan, audit_author
from prepare_xsum_original_assets import sha


def checked_official_output(text, job):
    """Reject zero-exit partial or skipped-I/O runs; no performance interpretation."""

    def number(pattern):
        values = re.findall(pattern, text, re.MULTILINE)
        if len(values) != 1:
            raise ValueError("one unambiguous original final result required")
        return int(values[0].replace(",", ""))

    requests = number(r"^\s*Requests:\s*([0-9,]+)\s*$")
    tokens = number(r"^\s*Tokens:\s*([0-9,]+)\s*$")
    reads = number(r"^\[Read Operations\]\s*\n\s*Count:\s*([0-9,]+)\s*$")
    writes = number(r"^\[Write Operations\]\s*\n\s*Count:\s*([0-9,]+)\s*$")
    if (
        requests != job["selected_original_rows"]
        or tokens != job["expected_selected_total_tokens"]
        or reads + writes != job["expected_selected_page_operations"]
    ):
        raise ValueError("original request/token/page-operation coverage incomplete")
    return {
        "requests": requests,
        "total_tokens": tokens,
        "read_page_operations": reads,
        "write_page_operations": writes,
        "complete_expected_operation_coverage": True,
    }


def run(
    author,
    traces,
    storage,
    output,
    *,
    source,
    count,
    storage_budget,
    io_window_confirmed,
    deadline=1800,
    process=subprocess.run,
    scenarios=None,
):
    if sys.platform != "linux":
        raise RuntimeError("authorized Linux server required; no local KV writes")
    if not io_window_confirmed:
        raise ValueError("operator-confirmed independent disk I/O window required")
    if (
        type(storage_budget) is not int
        or storage_budget <= 0
        or type(deadline) is not int
        or deadline < 1
    ):
        raise ValueError("explicit positive storage budget and deadline required")
    author, traces, storage, output = [
        Path(path).resolve() for path in (author, traces, storage, output)
    ]
    if output.exists() or any(
        output.is_relative_to(path) or path.is_relative_to(output)
        for path in (author, traces, storage)
    ):
        raise ValueError(
            "fresh private output separate from inputs and storage required"
        )
    plan = build_plan(author, traces, storage, source=source, count=count)
    selected = (
        [job["scenario"] for job in plan["jobs"]] if scenarios is None else scenarios
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= {"conversation", "toolagent"}
    ):
        raise ValueError(
            "distinct original scenarios required; no unknown replay fallback"
        )
    plan["jobs"] = [job for job in plan["jobs"] if job["scenario"] in selected]
    plan["retained_storage_capacity_bytes"] = (
        len(plan["jobs"]) * plan["max_pages_per_scenario"] * plan["page_bytes"]
    )
    capacity = plan["retained_storage_capacity_bytes"]
    free = shutil.disk_usage(storage.parent).free
    if capacity > storage_budget or free < capacity + 2 * 1024**3:
        raise ValueError(
            "original default profile exceeds explicit budget or free-space reserve; no automatic page reduction"
        )
    output.mkdir(mode=0o700, parents=True)
    staged = output / "official-source"
    receipt = {
        "source_id": "MOONCAKE",
        "status": "STARTED",
        "plan": plan,
        "storage_budget_bytes": storage_budget,
        "observed_free_bytes_before_run": free,
        "operator_io_window_confirmed": True,
        "jobs": [],
        "model_calls": 0,
        "performance_claim": False,
        "source_checkbox_eligible": False,
        "python_version": sys.version,
    }

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    save()
    try:
        for relative, expected in AUTHOR_FILES.items():
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(author / relative, target)
            if sha(target) != expected:
                raise ValueError("staged original source identity mismatch")
        audit_author(staged)
        storage.mkdir(mode=0o700)
        for job in plan["jobs"]:
            if sha(traces / job["path"]) != job["sha256"]:
                raise ValueError("original trace identity changed before scenario")
            if (
                shutil.disk_usage(storage).free
                < plan["max_pages_per_scenario"] * plan["page_bytes"] + 2 * 1024**3
            ):
                raise ValueError(
                    "insufficient current space for unchanged next scenario"
                )
            command = list(job["command"])
            command[1] = str(staged / "benchmark.py")
            command.insert(1, "-B")
            state = {
                "scenario": job["scenario"],
                "status": "STARTED",
                "command": command,
                "source_trace_sha256": job["sha256"],
            }
            receipt["jobs"].append(state)
            save()
            with (output / (job["scenario"] + "-official-private.log")).open(
                "x", encoding="utf-8"
            ) as log:
                result = process(
                    command,
                    cwd=staged,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=deadline,
                    env={
                        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                        "LANG": "C.UTF-8",
                        "PYTHONUTF8": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )
            state["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(
                    "original storage tool failed; preserve output, never retry or clear storage"
                )
            text = (output / (job["scenario"] + "-official-private.log")).read_text(
                "utf-8"
            )
            state.update(
                status="ORIGINAL_STORAGE_PROFILE_EXECUTED",
                coverage=checked_official_output(text, job),
            )
            save()
        # Re-use immutable source/trace preflight, without treating existing storage as fresh.
        audit_author(author)
        audit_author(staged)
        for item in source["files"]:
            path = traces / item["path"]
            if sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError("original trace changed during replay")
        receipt["status"] = (
            "SELECTED_OFFICIAL_STORAGE_PROFILES_EXECUTED_NOT_ENGINE_INFERENCE"
        )
    except Exception as error:
        receipt.update(status="FAILED_NOT_COMPLETE", error_type=type(error).__name__)
        raise
    finally:
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("author-tool", "trace-root", "storage-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--storage-budget-bytes", type=int, required=True)
    parser.add_argument("--io-window-confirmed", action="store_true")
    parser.add_argument("--deadline", type=int, default=1800)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["conversation", "toolagent"],
        help="Only missing scenarios, preserving valid prior runs; defaults to both",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--count", type=int, default=3)
    selection.add_argument("--full", action="store_true")
    args = parser.parse_args()
    source = json.loads(
        (
            ROOT / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text("utf-8")
    )["sources"]["MOONCAKE"]
    receipt = run(
        args.author_tool,
        args.trace_root,
        args.storage_root,
        args.output,
        source=source,
        count=None if args.full else args.count,
        storage_budget=args.storage_budget_bytes,
        io_window_confirmed=args.io_window_confirmed,
        deadline=args.deadline,
        scenarios=args.scenarios,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "profiles": len(receipt["jobs"]),
                "model_calls": 0,
                "source_checkbox_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
