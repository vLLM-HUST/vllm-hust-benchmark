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
import os
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

    Note: this uses ``pgrep -P`` tree-walking, which can miss processes that
    spawn and setsid/reparent between snapshots.  For a polling-independent
    ownership mechanism, use ``snapshot_session_descendants`` instead.
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


def get_session_id(pid: int) -> int | None:
    """Return the session ID (SID) of ``pid``.

    On Linux, reads /proc/<pid>/stat field 6 (session ID).  On other platforms,
    falls back to ``ps -o sid=``.

    Returns None if the PID doesn't exist or SID can't be read.

    Per reviewer round 6: '改用不会依赖轮询命中的归属机制，例如启动即进入
    job-owned cgroup' — session-based ownership is a polling-independent
    mechanism: all processes in the session are found via a single ``ps`` scan,
    regardless of when they spawned or whether they reparented.
    """
    stat_file = Path(f"/proc/{pid}/stat")
    if stat_file.is_file():
        try:
            content = stat_file.read_text()
            after_comm = content[content.rfind(")") + 1 :]
            fields = after_comm.split()
            # Field 6 in the full stat line is session ID; after comm (field 2)
            # it's field 4 in the after_comm slice (0-indexed: 3).
            if len(fields) >= 4:
                return int(fields[3])
        except (OSError, IndexError, ValueError):
            pass
    # Fallback for macOS / non-Linux
    try:
        result = subprocess.run(
            ["ps", "-o", "sid=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except (subprocess.SubprocessError, OSError, ValueError):
        pass
    return None


def snapshot_session_descendants(sid: int) -> list[dict[str, str | int]]:
    """Snapshot ALL processes in session ``sid`` via ``ps`` scan.

    This is a FALLBACK ownership mechanism for environments without cgroup v2.
    It scans ALL processes in the session via a single ``ps -eo pid,sid`` query.
    Any process in the session is captured, regardless of when it spawned or
    whether it reparented to init.

    LIMITATION (reviewer round 7): 'session scan 只能找到仍属于原 SID 的进程。
    EngineCore 如果自己调用 setsid()，会立即进入新的 session' — this function
    CANNOT find processes that called setsid() and left the session.  For
    setsid-surviving attribution, use ``snapshot_cgroup_descendants`` instead.

    The launcher must be started with ``setsid`` so it becomes a session
    leader and all its descendants inherit the same SID.

    Returns: list of ``{"pid": int, "starttime": str, "cmdline": str}``
    for all processes in the session (excluding the launcher itself if its
    PID equals the session ID).
    """
    snapshots: list[dict[str, str | int]] = []
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,sid="],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return snapshots
        session_pids: list[int] = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                p_pid = int(parts[0])
                p_sid = int(parts[1])
            except ValueError:
                continue
            if p_sid == sid:
                session_pids.append(p_pid)
        for pid in session_pids:
            st = get_starttime(pid)
            cmd = get_cmdline(pid)
            if st is not None and cmd is not None:
                snapshots.append({"pid": pid, "starttime": st, "cmdline": cmd})
    except (subprocess.SubprocessError, OSError):
        pass
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


# ---------------------------------------------------------------------------
# Cgroup v2 based ownership (reviewer round 7)
# ---------------------------------------------------------------------------

_CGROUP_V2_ROOT = Path("/sys/fs/cgroup")


def is_cgroup_v2_available() -> bool:
    """Check if cgroup v2 is mounted and available.

    Per reviewer round 7: 'session scan 只能找到仍属于原 SID 的进程。
    EngineCore 如果自己调用 setsid()，会立即进入新的 session' —
    cgroup v2 is the polling-independent attribution mechanism that survives
    setsid/reparent: processes stay in the cgroup from fork until exit,
    regardless of session changes.
    """
    return (_CGROUP_V2_ROOT / "cgroup.controllers").is_file()


def get_writable_cgroup_parent() -> Path | None:
    """Find a writable cgroup v2 parent directory.

    Tries (in order):
    1. /sys/fs/cgroup (root — available on the NPU server)
    2. /sys/fs/cgroup/user.slice/user-<uid>.slice (systemd user delegation)

    Returns the first writable parent, or None if cgroup v2 is unavailable
    or no writable parent is found.
    """
    if not is_cgroup_v2_available():
        return None

    # Try root cgroup (NPU server runs as root)
    if os.access(_CGROUP_V2_ROOT, os.W_OK):
        return _CGROUP_V2_ROOT

    # Try systemd user delegated cgroup
    uid = os.getuid()
    user_slice = _CGROUP_V2_ROOT / "user.slice" / f"user-{uid}.slice"
    if user_slice.is_dir() and os.access(user_slice, os.W_OK):
        return user_slice

    return None


def create_job_cgroup(job_id: str) -> Path | None:
    """Create a cgroup v2 directory for the job.

    Per reviewer round 7: '实现上需要 cgroup、runtime 同步 registry，或其他
    在它离开 launcher SID 前就完成的不可漏归属机制' — cgroup v2 membership
    is assigned at fork time (before the child can call setsid), making it
    a truly unforgeable attribution mechanism.

    Returns the cgroup path on success, or None if cgroup v2 is unavailable
    or the directory cannot be created.
    """
    parent = get_writable_cgroup_parent()
    if parent is None:
        return None

    cgroup_path = parent / f"vllm_hust_{job_id}"
    try:
        cgroup_path.mkdir(exist_ok=False)
    except (OSError, PermissionError):
        return None
    return cgroup_path


def add_pid_to_cgroup(pid: int, cgroup_path: Path) -> bool:
    """Add a PID to the cgroup by writing to cgroup.procs.

    After this call, the PID AND all its future descendants (forks) are
    members of the cgroup.  Descendants that call setsid() or reparent to
    init remain in the cgroup — this is the key property that makes cgroup
    attribution polling-independent and setsid-surviving.

    Returns True on success, False on failure.
    """
    procs_file = cgroup_path / "cgroup.procs"
    try:
        procs_file.write_text(f"{pid}\n")
        return True
    except (OSError, PermissionError):
        return False


def snapshot_cgroup_descendants(cgroup_path: Path) -> list[dict[str, str | int]]:
    """Read all PIDs from cgroup.procs and return their identity snapshots.

    Per reviewer round 7: '仅靠结束时扫描原 SID 无法补回已经离开的进程' —
    cgroup.procs contains ALL processes in the cgroup, including those that
    called setsid() and left the launcher's session.  This is the polling-
    independent scan that recovers setsid'd processes.

    Returns a list of ``{"pid": int, "starttime": str, "cmdline": str}``.
    """
    procs_file = cgroup_path / "cgroup.procs"
    snapshots: list[dict[str, str | int]] = []
    try:
        content = procs_file.read_text()
    except (OSError, PermissionError):
        return snapshots

    for line in content.split():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        st = get_starttime(pid)
        cmd = get_cmdline(pid)
        if st is not None and cmd is not None:
            snapshots.append({"pid": pid, "starttime": st, "cmdline": cmd})
    return snapshots


def cleanup_cgroup_descendants(
    cgroup_path: Path,
    signal: int,
) -> dict[str, list[int] | int]:
    """Kill all processes in the cgroup whose identity still matches.

    Reads cgroup.procs, verifies each process's PID + starttime + cmdline,
    and sends ``signal`` to those that still match.  This catches processes
    that called setsid() and left the launcher's session.

    Returns a summary dict: ``{"killed": [pid, ...], "skipped": count, "total": count}``.
    """
    snapshots = snapshot_cgroup_descendants(cgroup_path)
    killed: list[int] = []
    skipped = 0
    for entry in snapshots:
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
    return {"killed": killed, "skipped": skipped, "total": len(snapshots)}


def remove_cgroup(cgroup_path: Path) -> bool:
    """Remove the cgroup directory after cleanup.

    A cgroup can only be removed when it has no processes.  Call
    cleanup_cgroup_descendants first.

    Returns True on success, False on failure.
    """
    try:
        cgroup_path.rmdir()
        return True
    except (OSError, PermissionError):
        return False


def pid_in_cgroup(pid: int, cgroup_path: Path) -> bool:
    """Check if ``pid`` is a member of ``cgroup_path``.

    Reads ``cgroup.procs`` and returns True if ``pid`` is listed.
    """
    procs_file = cgroup_path / "cgroup.procs"
    try:
        content = procs_file.read_text()
    except (OSError, PermissionError):
        return False
    target = str(pid)
    for line in content.split():
        if line.strip() == target:
            return True
    return False


def exec_in_cgroup(
    cgroup_path: Path,
    ready_file: Path,
    cmd: list[str],
) -> int:
    """Join ``cgroup_path`` then ``exec`` ``cmd``.

    Per reviewer round 8: '请用受控 wrapper/握手让 launcher 自身在 exec
    benchmark、产生任何后代之前成功加入 job cgroup，加入失败时 fail closed'.

    This closes the startup race: the process joins the cgroup BEFORE
    exec'ing the benchmark, so all descendants inherit cgroup membership
    at fork time — BEFORE they can call ``setsid()`` or reparent.

    Steps:
    1. Write own PID to ``cgroup.procs`` (join the cgroup).
    2. Verify membership (fail closed if not).
    3. Write ``ready_file`` to signal the parent that the join succeeded.
    4. ``exec`` the command (replaces the process, keeps PID).

    If joining fails, exit with code 127 (fail closed) — do NOT exec
    the command, as descendants would not be tracked.

    Args:
        cgroup_path: Path to the job-owned cgroup v2 directory.
        ready_file: Path to a file that is created after successfully
            joining the cgroup.  The parent polls for this file to
            detect that the join completed.
        cmd: Command and arguments to exec after joining.

    Returns:
        Does not return on success (``exec`` replaces the process).
        Returns 127 on cgroup join failure or exec failure.
    """
    procs_file = cgroup_path / "cgroup.procs"
    my_pid = os.getpid()
    try:
        procs_file.write_text(f"{my_pid}\n")
    except (OSError, PermissionError) as exc:
        print(f"exec_in_cgroup: failed to join cgroup: {exc}", file=sys.stderr)
        return 127

    # Verify membership — fail closed if the write didn't take effect.
    if not pid_in_cgroup(my_pid, cgroup_path):
        print(
            "exec_in_cgroup: PID not in cgroup after join (write silently ignored?)",
            file=sys.stderr,
        )
        return 127

    # Signal to the parent that the join succeeded.
    try:
        ready_file.write_text(f"{my_pid}\n")
    except OSError:
        # Best effort — don't block exec on signaling failure.  The
        # parent will time out and fall back to session scan.
        pass

    # exec the command — replaces this process, keeps PID.  All forks
    # from the exec'd process inherit cgroup membership at fork time.
    try:
        os.execvp(cmd[0], cmd)
    except OSError as exc:
        print(f"exec_in_cgroup: exec failed: {exc}", file=sys.stderr)
        return 127
    return 127  # unreachable


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
        python process_identity.py get_sid <pid>
        python process_identity.py snapshot_session <sid>
        python process_identity.py cleanup <snapshot_file> <signal>
        python process_identity.py cgroup_available
        python process_identity.py create_cgroup <job_id>
        python process_identity.py add_pid <pid> <cgroup_path>
        python process_identity.py snapshot_cgroup <cgroup_path>
        python process_identity.py cleanup_cgroup <cgroup_path> <signal>
        python process_identity.py remove_cgroup <cgroup_path>
        python process_identity.py pid_in_cgroup <pid> <cgroup_path>
        python process_identity.py exec_in_cgroup <cgroup_path> <ready_file> <cmd> [args...]

    Output:
    - ``get_starttime`` / ``get_cmdline``: raw value on stdout (or empty).
    - ``verify``: exit 0 if identity matches, 1 if not (no stdout).
    - ``snapshot`` / ``snapshot_session`` / ``snapshot_cgroup``: JSON array on stdout.
    - ``get_sid``: SID integer on stdout (or empty).
    - ``cleanup`` / ``cleanup_cgroup``: JSON summary on stdout.
    - ``cgroup_available``: exit 0 if available, 1 if not.
    - ``create_cgroup``: cgroup path on stdout (or empty on failure).
    - ``add_pid``: exit 0 on success, 1 on failure.
    - ``remove_cgroup``: exit 0 on success, 1 on failure.
    - ``pid_in_cgroup``: exit 0 if PID is in cgroup, 1 if not.
    - ``exec_in_cgroup``: does not return on success (exec replaces the
      process); exits 127 on cgroup join failure.

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

    elif cmd == "get_sid" and len(sys.argv) == 3:
        pid = int(sys.argv[2])
        result = get_session_id(pid)
        if result is not None:
            print(result)
        return 0

    elif cmd == "snapshot_session" and len(sys.argv) == 3:
        sid = int(sys.argv[2])
        result = snapshot_session_descendants(sid)
        print(json.dumps(result))
        return 0

    elif cmd == "cleanup" and len(sys.argv) == 4:
        snapshot_file = Path(sys.argv[2])
        signal = int(sys.argv[3])
        summary = cleanup_descendants(snapshot_file, signal)
        print(json.dumps(summary))
        return 0

    elif cmd == "cgroup_available" and len(sys.argv) == 2:
        return 0 if is_cgroup_v2_available() else 1

    elif cmd == "create_cgroup" and len(sys.argv) == 3:
        job_id = sys.argv[2]
        result = create_job_cgroup(job_id)
        if result is not None:
            print(result)
            return 0
        return 1

    elif cmd == "add_pid" and len(sys.argv) == 4:
        pid = int(sys.argv[2])
        cgroup_path = Path(sys.argv[3])
        return 0 if add_pid_to_cgroup(pid, cgroup_path) else 1

    elif cmd == "snapshot_cgroup" and len(sys.argv) == 3:
        cgroup_path = Path(sys.argv[2])
        result = snapshot_cgroup_descendants(cgroup_path)
        print(json.dumps(result))
        return 0

    elif cmd == "cleanup_cgroup" and len(sys.argv) == 4:
        cgroup_path = Path(sys.argv[2])
        signal = int(sys.argv[3])
        summary = cleanup_cgroup_descendants(cgroup_path, signal)
        print(json.dumps(summary))
        return 0

    elif cmd == "remove_cgroup" and len(sys.argv) == 3:
        cgroup_path = Path(sys.argv[2])
        return 0 if remove_cgroup(cgroup_path) else 1

    elif cmd == "pid_in_cgroup" and len(sys.argv) == 4:
        pid = int(sys.argv[2])
        cgroup_path = Path(sys.argv[3])
        return 0 if pid_in_cgroup(pid, cgroup_path) else 1

    elif cmd == "exec_in_cgroup" and len(sys.argv) >= 5:
        cgroup_path = Path(sys.argv[2])
        ready_file = Path(sys.argv[3])
        exec_cmd = sys.argv[4:]
        return exec_in_cgroup(cgroup_path, ready_file, exec_cmd)

    else:
        print(f"Unknown command/args: {sys.argv[1:]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
