#!/usr/bin/env python3
"""Run locally installed, approved native clients and a receipt exporter."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def process_state(pid):
    # comm may contain spaces, so parse after the final closing parenthesis.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "state": fields[0],
        "pgrp": int(fields[2]),
        "session": int(fields[3]),
        "start_ticks": fields[19],
    }


def cleanup_group(identity, output, grace_s=15):
    """Stop a recorded session, including children whose group leader has exited."""
    group = identity["pid"]
    marker = ("VLLM_HUST_TEST_OUTPUT=" + str(output)).encode()

    def members():
        found = []
        anchored = False
        unmarked = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                state = process_state(int(entry.name))
                if (
                    state["pgrp"] != group
                    or state["session"] != group
                    or state["state"] == "Z"
                ):
                    continue
                if (
                    state["pid"] == group
                    and state["start_ticks"] != identity["start_ticks"]
                ):
                    raise RuntimeError("Recorded stage PID was reused")
                marked = marker in (entry / "environ").read_bytes().split(b"\0")
                # Exiting tasks can lose environ before their state becomes Z.
                # The recorded live session leader remains an ownership anchor.
                current = process_state(state["pid"])
                if (
                    current["state"] == "Z"
                    or current["start_ticks"] != state["start_ticks"]
                ):
                    continue
                if int(state["start_ticks"]) < int(identity["start_ticks"]):
                    raise RuntimeError("Stage session contains an older process")
                anchored = anchored or marked or state["pid"] == group
                if not marked:
                    unmarked.append(state["pid"])
                found.append(state)
            except (FileNotFoundError, ProcessLookupError):
                continue
        if found and unmarked and not anchored:
            raise RuntimeError("Refusing an unanchored reused process group")
        return found

    stopped = []
    for sig, timeout in [(signal.SIGTERM, grace_s), (signal.SIGKILL, 5)]:
        active = members()
        for state in active:
            try:
                if process_state(state["pid"])["start_ticks"] == state["start_ticks"]:
                    os.kill(state["pid"], sig)
                    stopped.append({"pid": state["pid"], "signal": int(sig)})
            except (FileNotFoundError, ProcessLookupError):
                pass
        deadline = time.monotonic() + timeout
        while members() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not members():
            return stopped
    raise RuntimeError("Owned stage processes remain alive; output must not be sealed")


def run_stage(step, index, output, values):
    name = step["name"]
    command = [word.format_map(values) for word in step["argv"]]
    timeout = step["timeout_s"]
    if not command or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"{name}: invalid argv or timeout")
    # Each step uses its own process group on the Linux test host.
    with (output / f"stage-{index:02d}.log").open("wb") as log:
        with subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        ) as process:
            identity = process_state(process.pid) if os.name == "posix" else None
            if identity:
                (output / f"stage-{index:02d}-process.json").write_text(
                    json.dumps(identity) + "\n"
                )
            try:
                code = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
                raise
            finally:
                if identity:
                    cleanup_group(identity, output)
    if code:
        raise RuntimeError(f"{name}: exit {code}; see stage-{index:02d}.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    args = parser.parse_args()
    workflow = json.loads(args.workflow.read_text(encoding="utf-8-sig"))
    if workflow["schema"] != "vllm-hust-native-workflow/v1" or not workflow["steps"]:
        raise ValueError("Supply an approved native workflow with at least one step")
    values = {
        name.lower(): os.environ[f"VLLM_HUST_TEST_{name}"]
        for name in ("OUTPUT", "CASE", "ARM", "REPEAT", "CONTRACT", "CONFIG")
    }
    output = Path(values["output"])
    try:
        for index, step in enumerate(workflow["steps"], 1):
            run_stage(step, index, output, values)
    finally:
        for index, step in enumerate(
            workflow.get("cleanup", []), len(workflow["steps"]) + 1
        ):
            run_stage(step, index, output, values)
    if not (output / "receipt.json").is_file():
        raise RuntimeError(
            "The final stage must export receipt.json and referenced evidence"
        )


if __name__ == "__main__":
    main()
