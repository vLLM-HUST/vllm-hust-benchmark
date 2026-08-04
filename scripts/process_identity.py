"""Process identity verification for safe cleanup.

Per reviewer round 5: '请补齐 launcher 和后代的 PID/start time/cmdline 身份
记录与复核，并增加快速 detach、launcher PID reuse、descendant cmdline
mismatch 的反向测试'.

This module provides triple-identity (PID + start_time + cmdline) verification
to prevent PID-reuse false kills during cleanup.  The bash retest script
calls it via CLI; tests import the functions directly.

Identity model:
- ``starttime``: On Linux, field 22 of /proc/<pid>/stat (clock ticks since
  boot).  On macOS, ``ps -o lstart=`` (less precise but unique per boot).
- ``cmdline``: On Linux, /proc/<pid>/cmdline with NULs replaced by spaces.
  On macOS, ``ps -o command=``.

A PID is considered safe to kill only if ALL THREE of the following match the
recorded snapshot:
  1. The PID still exists.
  2. Its current starttime equals the recorded starttime.
  3. Its current cmdline equals the recorded cmdline.

If any check fails, the PID may have been recycled to a different process and
must NOT be killed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def get_starttime(pid: int) -> str | None:
    """Return the start time of ``pid`` as a string identity token.

    On Linux, reads /proc/<pid>/stat field 22 (starttime in clock ticks
    since boot).  On other platforms, falls back to ``ps -o lstart=``.

    Returns None if the PID doesn't exist or identity can't be read.
    """
    stat_file = Path(f"/proc/{pid}/stat")
    if stat_file.is_file():
        try:
            content = stat_file.read_text()
            # comm is in parens and may contain spaces/parens; parse from
            # the LAST ')' to get fields after comm.
            after_comm = content[content.rfind(")") + 1 :]
            fields = after_comm.split()
            # Field 22 in the full stat line; after comm (field 2) it's
            # field 20 in the after_comm slice (0-indexed: 19).
            if len(fields) >= 20:
                return fields[19]
        except (OSError, IndexError):
            pass
    # Fallback for macOS / non-Linux
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return None


def get_cmdline(pid: int) -> str | None:
    """Return the command line of ``pid`` as a string identity token.

    On Linux, reads /proc/<pid>/cmdline (NUL-separated args joined by
    spaces).  Kernel threads (empty cmdline) use comm as identity.
    On other platforms, falls back to ``ps -o command=``.

    Returns None if the PID doesn't exist or identity can't be read.
    """
    cmdline_file = Path(f"/proc/{pid}/cmdline")
    if cmdline_file.is_file():
        try:
            raw = cmdline_file.read_bytes()
            if not raw:
                # Kernel thread — no cmdline.  Use comm as identity so
                # kernel threads can still be verified.
                comm_file = Path(f"/proc/{pid}/comm")
                if comm_file.is_file():
                    return f"[kernel]{comm_file.read_text().strip()}"
                return None
            return raw.decode("utf-8", errors="replace").replace("\x00", " ").rstrip()
        except OSError:
            pass
    # Fallback for macOS / non-Linux
    try:
        result = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return None


def verify_identity(
    pid: int,
    recorded_starttime: str | None,
    recorded_cmdline: str | None,
) -> bool:
    """Verify that ``pid`` still has the same starttime AND cmdline.

    Returns True only if ALL of the following hold:
    - ``recorded_starttime`` and ``recorded_cmdline`` are non-empty.
    - The PID exists.
    - Its current starttime equals ``recorded_starttime``.
    - Its current cmdline equals ``recorded_cmdline``.

    Returns False if any check fails — the PID may have been recycled to a
    different process and must NOT be killed.

    Per reviewer round 5: 'cleanup 时只要当前 PID 存在就会发送 TERM，仍有
    PID reuse 误杀风险' — must verify ALL three identity dimensions.
    """
    if not recorded_starttime or not recorded_cmdline:
        return False
    current_st = get_starttime(pid)
    if current_st is None or current_st != recorded_starttime:
        return False
    current_cmd = get_cmdline(pid)
    if current_cmd is None or current_cmd != recorded_cmdline:
        return False
    return True


def snapshot_descendants(parent_pid: int) -> list[dict[str, str | int]]:
    """Recursively snapshot all descendants of ``parent_pid``.

    Returns a list of dicts: ``[{"pid": int, "starttime": str, "cmdline": str}, ...]``.
    Captures the full process tree at this moment.  Called repeatedly while
    the launcher is alive to catch processes that spawn and detach quickly.

    Per reviewer round 5: '每 2 秒轮询一次会漏掉在首次/两次 snapshot 之间快速
    spawn、setsid、reparent 的 EngineCore' — the caller should poll at 0.5s
    intervals and call this function immediately after launch (before the
    first sleep) to minimize the window where a fast-detach process is missed.
    """
    snapshots: list[dict[str, str | int]] = []
    seen: set[int] = set()

    def _walk(p: int) -> None:
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(p)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return
            for line in result.stdout.split():
                try:
                    child = int(line.strip())
                except ValueError:
                    continue
                if child in seen:
                    continue
                seen.add(child)
                # Recurse first (depth-first) to capture grandchildren.
                _walk(child)
                st = get_starttime(child)
                cmd = get_cmdline(child)
                if st is not None and cmd is not None:
                    snapshots.append({"pid": child, "starttime": st, "cmdline": cmd})
        except (subprocess.SubprocessError, OSError):
            pass

    _walk(parent_pid)
    return snapshots


def merge_snapshots(
    snapshot_lists: list[list[dict[str, str | int]]],
) -> list[dict[str, str | int]]:
    """Merge multiple snapshot lists, keeping unique PIDs.

    Later snapshots override earlier ones for the same PID — this handles the
    case where a process restarts with a different cmdline (we keep the
    latest identity seen while the launcher was alive).

    Per reviewer round 5: '在 launcher 存活时持久化已验证后代的 PID/start time/
    cmdline' — merging multiple snapshots ensures we capture processes that
    appeared at any point during the benchmark run, even if they later
    detached or reparented.
    """
    merged: dict[int, dict[str, str | int]] = {}
    for snap in snapshot_lists:
        for entry in snap:
            pid = entry["pid"]
            if isinstance(pid, int):
                merged[pid] = entry
    return list(merged.values())


def cleanup_descendants(
    snapshot_file: Path,
    signal: int,
) -> dict[str, list[int] | int]:
    """Kill descendants whose identity still matches the recorded snapshot.

    Reads all snapshot JSON arrays from ``snapshot_file`` (one per line),
    merges them (keeping the latest identity per PID), verifies each
    descendant's identity, and sends ``signal`` to those that still match.

    Returns a summary dict: ``{"killed": [pid, ...], "skipped": count, "total": count}``.
    """
    import os

    snapshot_lists: list[list[dict[str, str | int]]] = []
    with snapshot_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    snapshot_lists.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    merged = merge_snapshots(snapshot_lists)
    killed: list[int] = []
    skipped = 0
    for entry in merged:
        pid = entry["pid"]
        if not isinstance(pid, int):
            continue
        st = entry["starttime"]
        cmd = entry["cmdline"]
        if not isinstance(st, str) or not isinstance(cmd, str):
            continue
        if verify_identity(pid, st, cmd):
            try:
                os.kill(pid, signal)
                killed.append(pid)
            except (ProcessLookupError, PermissionError):
                skipped += 1
        else:
            skipped += 1
    return {"killed": killed, "skipped": skipped, "total": len(merged)}


def main() -> int:
    """CLI for use from bash scripts.

    Usage:
        python process_identity.py get_starttime <pid>
        python process_identity.py get_cmdline <pid>
        python process_identity.py verify <pid> <starttime> <cmdline>
        python process_identity.py snapshot <parent_pid>
        python process_identity.py cleanup <snapshot_file> <signal>

    Output:
    - ``get_starttime`` / ``get_cmdline``: raw value on stdout (or empty).
    - ``verify``: exit 0 if identity matches, 1 if not (no stdout).
    - ``snapshot``: JSON array on stdout.
    - ``cleanup``: JSON summary on stdout.

    The raw-value output for ``get_*`` commands makes bash integration simple:
        st=$(python process_identity.py get_starttime $pid)
    """
    if len(sys.argv) < 2:
        print("Usage: process_identity.py <command> <args>", file=sys.stderr)
        return 2

    cmd = sys.argv[1]

    if cmd == "get_starttime" and len(sys.argv) == 3:
        pid = int(sys.argv[2])
        result = get_starttime(pid)
        if result is not None:
            print(result)
        return 0

    elif cmd == "get_cmdline" and len(sys.argv) == 3:
        pid = int(sys.argv[2])
        result = get_cmdline(pid)
        if result is not None:
            print(result)
        return 0

    elif cmd == "verify" and len(sys.argv) == 5:
        pid = int(sys.argv[2])
        st = sys.argv[3]
        cmdline = sys.argv[4]
        matches = verify_identity(pid, st, cmdline)
        return 0 if matches else 1

    elif cmd == "snapshot" and len(sys.argv) == 3:
        pid = int(sys.argv[2])
        result = snapshot_descendants(pid)
        print(json.dumps(result))
        return 0

    elif cmd == "cleanup" and len(sys.argv) == 4:
        snapshot_file = Path(sys.argv[2])
        signal = int(sys.argv[3])
        summary = cleanup_descendants(snapshot_file, signal)
        print(json.dumps(summary))
        return 0

    else:
        print(f"Unknown command/args: {sys.argv[1:]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
