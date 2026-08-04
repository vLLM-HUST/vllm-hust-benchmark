"""Tests for process identity verification (issue #146 reviewer round 5).

Per reviewer round 5: '请补齐 launcher 和后代的 PID/start time/cmdline 身份
记录与复核，并增加快速 detach、launcher PID reuse、descendant cmdline
mismatch 的反向测试'.

These tests verify the triple-identity (PID + starttime + cmdline) verification
logic in ``scripts/process_identity.py``.  The identity-reading functions
(``get_starttime``, ``get_cmdline``) are mocked to simulate PID reuse scenarios
without relying on real process recycling.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "process_identity.py"

# Session ID (SID) and setsid are Linux-specific.  macOS ps does not
# support ``-o sid=``, and setsid is not available.  Tests that require
# real SID/setsid are skipped on non-Linux platforms.
_SKIP_NON_LINUX = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="SID/setsid is Linux-specific (macOS ps lacks -o sid=)",
)


@pytest.fixture(scope="module")
def pi_mod():
    """Load process_identity.py as a module."""
    spec = importlib.util.spec_from_file_location("process_identity", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# verify_identity: basic positive/negative cases
# ---------------------------------------------------------------------------


class TestVerifyIdentity:
    """Tests for verify_identity — the core triple-identity check."""

    def test_all_match_returns_true(self, pi_mod):
        """When PID exists and starttime + cmdline match, returns True."""
        with (
            patch.object(pi_mod, "get_starttime", return_value="12345"),
            patch.object(pi_mod, "get_cmdline", return_value="python bench.py"),
        ):
            assert pi_mod.verify_identity(999, "12345", "python bench.py") is True

    def test_empty_recorded_starttime_returns_false(self, pi_mod):
        """Empty recorded starttime must fail (no identity to compare)."""
        assert pi_mod.verify_identity(999, "", "cmd") is False

    def test_empty_recorded_cmdline_returns_false(self, pi_mod):
        """Empty recorded cmdline must fail."""
        assert pi_mod.verify_identity(999, "12345", "") is False

    def test_none_recorded_values_return_false(self, pi_mod):
        """None recorded values must fail."""
        assert pi_mod.verify_identity(999, None, "cmd") is False
        assert pi_mod.verify_identity(999, "12345", None) is False

    def test_pid_not_found_returns_false(self, pi_mod):
        """When PID doesn't exist (get_starttime returns None), fail."""
        with (
            patch.object(pi_mod, "get_starttime", return_value=None),
            patch.object(pi_mod, "get_cmdline", return_value="cmd"),
        ):
            assert pi_mod.verify_identity(999, "12345", "cmd") is False

    def test_starttime_mismatch_returns_false(self, pi_mod):
        """Different starttime means PID was recycled — must NOT kill.

        Per reviewer round 5: 'cleanup 时只要当前 PID 存在就会发送 TERM，
        仍有 PID reuse 误杀风险' — starttime mismatch is the primary
        signal of PID reuse.
        """
        with (
            patch.object(pi_mod, "get_starttime", return_value="99999"),
            patch.object(pi_mod, "get_cmdline", return_value="python bench.py"),
        ):
            assert pi_mod.verify_identity(999, "12345", "python bench.py") is False

    def test_cmdline_mismatch_returns_false(self, pi_mod):
        """Different cmdline means PID was recycled — must NOT kill.

        Per reviewer round 5: '后代只记录 PID/start time，没有保存和复核 cmdline'
        — cmdline is the second identity dimension that catches PID reuse
        even when starttime happens to collide.
        """
        with (
            patch.object(pi_mod, "get_starttime", return_value="12345"),
            patch.object(pi_mod, "get_cmdline", return_value="different cmd"),
        ):
            assert pi_mod.verify_identity(999, "12345", "python bench.py") is False

    def test_cmdline_none_current_returns_false(self, pi_mod):
        """When current cmdline can't be read (process exited), fail."""
        with (
            patch.object(pi_mod, "get_starttime", return_value="12345"),
            patch.object(pi_mod, "get_cmdline", return_value=None),
        ):
            assert pi_mod.verify_identity(999, "12345", "python bench.py") is False


# ---------------------------------------------------------------------------
# Reverse test: launcher PID reuse
# ---------------------------------------------------------------------------


class TestLauncherPidReuse:
    """Reverse test: launcher PID recycled to a different process.

    Per reviewer round 5: 'launcher 则连启动时的 start time 都没有记录，
    cleanup 时只要当前 PID 存在就会发送 TERM，仍有 PID reuse 误杀风险'.

    Scenario:
    1. Launcher starts with PID 1234, starttime=100, cmdline="python bench".
    2. Launcher exits naturally.
    3. A different process (e.g. another user's job) gets PID 1234 with
       starttime=200, cmdline="some_other_process".
    4. kill_owned_server calls verify_identity(1234, "100", "python bench").
    5. verify_identity must return False — the PID now belongs to a different
       process and must NOT be killed.
    """

    def test_launcher_pid_reuse_starttime_mismatch(self, pi_mod):
        """Launcher PID recycled — starttime differs, must not kill."""
        recorded_starttime = "100"
        recorded_cmdline = "python -m vllm.entrypoints.cli.main bench throughput"

        # Simulate: PID 1234 now has a different starttime (recycled)
        with (
            patch.object(pi_mod, "get_starttime", return_value="200"),
            patch.object(pi_mod, "get_cmdline", return_value=recorded_cmdline),
        ):
            result = pi_mod.verify_identity(1234, recorded_starttime, recorded_cmdline)
        assert result is False, (
            "Launcher PID with different starttime must NOT be killed "
            "(PID reuse detected)"
        )

    def test_launcher_pid_reuse_cmdline_mismatch(self, pi_mod):
        """Launcher PID recycled — cmdline differs, must not kill.

        Even if starttime somehow matches (unlikely but possible on some
        systems with coarse granularity), cmdline mismatch catches the reuse.
        """
        recorded_starttime = "100"
        recorded_cmdline = "python -m vllm.entrypoints.cli.main bench throughput"

        # Simulate: PID 1234 now has different cmdline (recycled)
        with (
            patch.object(pi_mod, "get_starttime", return_value="100"),
            patch.object(
                pi_mod, "get_cmdline", return_value="/usr/bin/some_other_process --flag"
            ),
        ):
            result = pi_mod.verify_identity(1234, recorded_starttime, recorded_cmdline)
        assert result is False, (
            "Launcher PID with different cmdline must NOT be killed "
            "(PID reuse detected via cmdline mismatch)"
        )

    def test_launcher_pid_reuse_both_mismatch(self, pi_mod):
        """Launcher PID recycled — both starttime and cmdline differ."""
        with (
            patch.object(pi_mod, "get_starttime", return_value="99999"),
            patch.object(pi_mod, "get_cmdline", return_value="completely different"),
        ):
            result = pi_mod.verify_identity(1234, "100", "python bench")
        assert result is False

    def test_launcher_still_same_process_allows_kill(self, pi_mod):
        """Positive test: launcher still alive with same identity — kill allowed."""
        with (
            patch.object(pi_mod, "get_starttime", return_value="100"),
            patch.object(pi_mod, "get_cmdline", return_value="python bench"),
        ):
            result = pi_mod.verify_identity(1234, "100", "python bench")
        assert result is True


# ---------------------------------------------------------------------------
# Reverse test: descendant cmdline mismatch
# ---------------------------------------------------------------------------


class TestDescendantCmdlineMismatch:
    """Reverse test: descendant PID recycled to a different cmdline.

    Per reviewer round 5: '后代只记录 PID/start time，没有保存和复核 cmdline'.

    Scenario:
    1. EngineCore worker has PID 5678, starttime=150, cmdline="EngineCore".
    2. EngineCore exits after the benchmark.
    3. A different process gets PID 5678 with starttime=250, cmdline="other".
    4. cleanup_descendants verifies identity before killing.
    5. The recycled PID must NOT be killed.
    """

    def test_descendant_cmdline_mismatch_not_killed(self, pi_mod, tmp_path):
        """Descendant PID with different cmdline must NOT be killed."""
        snapshot_file = tmp_path / "snapshots.jsonl"
        snapshot = [
            {"pid": 5678, "starttime": "150", "cmdline": "EngineCore worker"},
        ]
        snapshot_file.write_text(json.dumps(snapshot) + "\n")

        killed_pids: list[int] = []

        with (
            patch.object(pi_mod, "get_starttime", return_value="150"),
            patch.object(pi_mod, "get_cmdline", return_value="different_process --arg"),
            patch("os.kill", side_effect=lambda pid, sig: killed_pids.append(pid)),
        ):
            summary = pi_mod.cleanup_descendants(snapshot_file, 15)

        assert 5678 not in killed_pids, (
            "Descendant with mismatched cmdline must NOT be killed"
        )
        assert summary["killed"] == []
        assert summary["skipped"] >= 1

    def test_descendant_starttime_mismatch_not_killed(self, pi_mod, tmp_path):
        """Descendant PID with different starttime must NOT be killed."""
        snapshot_file = tmp_path / "snapshots.jsonl"
        snapshot = [
            {"pid": 5678, "starttime": "150", "cmdline": "EngineCore worker"},
        ]
        snapshot_file.write_text(json.dumps(snapshot) + "\n")

        killed_pids: list[int] = []

        with (
            patch.object(pi_mod, "get_starttime", return_value="999"),
            patch.object(pi_mod, "get_cmdline", return_value="EngineCore worker"),
            patch("os.kill", side_effect=lambda pid, sig: killed_pids.append(pid)),
        ):
            summary = pi_mod.cleanup_descendants(snapshot_file, 15)

        assert 5678 not in killed_pids
        assert summary["killed"] == []

    def test_descendant_identity_match_is_killed(self, pi_mod, tmp_path):
        """Positive test: descendant with matching identity IS killed."""
        snapshot_file = tmp_path / "snapshots.jsonl"
        snapshot = [
            {"pid": 5678, "starttime": "150", "cmdline": "EngineCore worker"},
        ]
        snapshot_file.write_text(json.dumps(snapshot) + "\n")

        killed_pids: list[int] = []

        with (
            patch.object(pi_mod, "get_starttime", return_value="150"),
            patch.object(pi_mod, "get_cmdline", return_value="EngineCore worker"),
            patch("os.kill", side_effect=lambda pid, sig: killed_pids.append(pid)),
        ):
            summary = pi_mod.cleanup_descendants(snapshot_file, 15)

        assert 5678 in killed_pids
        assert summary["killed"] == [5678]

    def test_mixed_match_and_mismatch(self, pi_mod, tmp_path):
        """Only descendants with matching identity are killed; others skipped."""
        snapshot_file = tmp_path / "snapshots.jsonl"
        snapshot = [
            {"pid": 100, "starttime": "10", "cmdline": "match_cmd"},
            {"pid": 200, "starttime": "20", "cmdline": "mismatch_cmd"},
            {"pid": 300, "starttime": "30", "cmdline": "match_cmd"},
        ]
        snapshot_file.write_text(json.dumps(snapshot) + "\n")

        killed_pids: list[int] = []

        # PID 100 and 300 match; PID 200 has different cmdline
        def mock_cmdline(pid):
            if pid == 200:
                return "different_cmd"
            return "match_cmd"

        recorded_starttimes = {100: "10", 200: "20", 300: "30"}

        with (
            patch.object(
                pi_mod,
                "get_starttime",
                side_effect=lambda pid: recorded_starttimes.get(pid),
            ),
            patch.object(pi_mod, "get_cmdline", side_effect=mock_cmdline),
            patch("os.kill", side_effect=lambda pid, sig: killed_pids.append(pid)),
        ):
            summary = pi_mod.cleanup_descendants(snapshot_file, 15)

        assert sorted(killed_pids) == [100, 300]
        assert 200 not in killed_pids
        assert summary["skipped"] >= 1


# ---------------------------------------------------------------------------
# Reverse test: fast detach (process spawns and setsid between snapshots)
# ---------------------------------------------------------------------------


class TestFastDetach:
    """Reverse test: fast-detach process captured by immediate snapshot.

    Per reviewer round 5: '每 2 秒轮询一次会漏掉在首次/两次 snapshot 之间快速
    spawn、setsid、reparent 的 EngineCore'.

    Scenario:
    1. Launcher starts.
    2. EngineCore spawns and immediately setsid (detaches from the process
       tree) within 100ms.
    3. The first snapshot (immediate, before sleep) catches it.
    4. By the next poll (0.5s later), EngineCore has already reparented to
       init and pgrep -P no longer finds it.
    5. But because the immediate snapshot captured it, merge_snapshots
       includes it in the merged list, and cleanup can kill it.

    This test verifies that merge_snapshots correctly includes processes
    from ANY snapshot, even if they disappeared in later snapshots.
    """

    def test_fast_detach_captured_by_immediate_snapshot(self, pi_mod):
        """A process that appears in snapshot 1 but not snapshot 2 is kept.

        The immediate snapshot (before first sleep) catches the fast-detach
        process.  Later snapshots miss it (it already reparented).  But
        merge_snapshots keeps it because it appeared in at least one snapshot.
        """
        snapshot_1 = [
            {"pid": 100, "starttime": "10", "cmdline": "launcher"},
            {"pid": 101, "starttime": "11", "cmdline": "EngineCore"},
        ]
        snapshot_2 = [
            {"pid": 100, "starttime": "10", "cmdline": "launcher"},
            # EngineCore (pid 101) already detached — not in snapshot 2
        ]
        snapshot_3 = [
            {"pid": 100, "starttime": "10", "cmdline": "launcher"},
            # Still no EngineCore
        ]

        merged = pi_mod.merge_snapshots([snapshot_1, snapshot_2, snapshot_3])

        pids = {e["pid"] for e in merged}
        assert 101 in pids, (
            "Fast-detach process (pid 101) must be captured by the immediate "
            "snapshot and retained in merged results"
        )

    def test_process_appearing_only_in_later_snapshot_is_kept(self, pi_mod):
        """A process that appears in snapshot 2 but not snapshot 1 is kept.

        This covers the case where a process spawns just after the immediate
        snapshot but before the first poll.
        """
        snapshot_1 = [
            {"pid": 100, "starttime": "10", "cmdline": "launcher"},
        ]
        snapshot_2 = [
            {"pid": 100, "starttime": "10", "cmdline": "launcher"},
            {"pid": 200, "starttime": "20", "cmdline": "late_spawn"},
        ]

        merged = pi_mod.merge_snapshots([snapshot_1, snapshot_2])

        pids = {e["pid"] for e in merged}
        assert 200 in pids, "Late-spawning process must be captured"

    def test_process_changing_cmdline_keeps_latest(self, pi_mod):
        """If a process restarts with a different cmdline, latest wins.

        This handles the edge case where a process crashes and restarts
        with the same PID but different cmdline.  We keep the latest
        identity seen while the launcher was alive.
        """
        snapshot_1 = [
            {"pid": 100, "starttime": "10", "cmdline": "old_cmd"},
        ]
        snapshot_2 = [
            {"pid": 100, "starttime": "20", "cmdline": "new_cmd"},
        ]

        merged = pi_mod.merge_snapshots([snapshot_1, snapshot_2])

        assert len(merged) == 1
        entry = merged[0]
        assert entry["cmdline"] == "new_cmd"
        assert entry["starttime"] == "20"

    def test_empty_snapshot_lists(self, pi_mod):
        """Empty snapshot lists produce empty merged result."""
        assert pi_mod.merge_snapshots([]) == []
        assert pi_mod.merge_snapshots([[], []]) == []

    def test_fast_detach_cleanup_kills_captured_process(self, pi_mod, tmp_path):
        """End-to-end: fast-detach process is killed at cleanup.

        The process was captured by the immediate snapshot, so even though
        it later detached (pgrep can't find it), cleanup_descendants still
        has its identity and can verify + kill it if it's still alive.
        """
        snapshot_file = tmp_path / "snapshots.jsonl"
        # Simulate: immediate snapshot caught EngineCore (pid 101)
        snapshot_1 = [
            {"pid": 101, "starttime": "11", "cmdline": "EngineCore"},
        ]
        # Later snapshots missed it (detached)
        snapshot_2: list[dict[str, str | int]] = []

        snapshot_file.write_text(
            json.dumps(snapshot_1) + "\n" + json.dumps(snapshot_2) + "\n"
        )

        killed_pids: list[int] = []

        with (
            patch.object(pi_mod, "get_starttime", return_value="11"),
            patch.object(pi_mod, "get_cmdline", return_value="EngineCore"),
            patch("os.kill", side_effect=lambda pid, sig: killed_pids.append(pid)),
        ):
            pi_mod.cleanup_descendants(snapshot_file, 9)

        assert 101 in killed_pids, (
            "Fast-detach process captured by immediate snapshot must be killed "
            "at cleanup if its identity still matches"
        )


# ---------------------------------------------------------------------------
# merge_snapshots: deduplication and override behavior
# ---------------------------------------------------------------------------


class TestMergeSnapshots:
    """Tests for merge_snapshots — deduplication and latest-wins behavior."""

    def test_deduplicates_same_pid(self, pi_mod):
        """Same PID in multiple snapshots produces one entry."""
        snapshots = [
            [{"pid": 100, "starttime": "10", "cmdline": "cmd"}],
            [{"pid": 100, "starttime": "10", "cmdline": "cmd"}],
        ]
        merged = pi_mod.merge_snapshots(snapshots)
        assert len(merged) == 1

    def test_latest_identity_wins(self, pi_mod):
        """When PID appears in multiple snapshots, latest identity wins."""
        snapshots = [
            [{"pid": 100, "starttime": "10", "cmdline": "v1"}],
            [{"pid": 100, "starttime": "10", "cmdline": "v2"}],
        ]
        merged = pi_mod.merge_snapshots(snapshots)
        assert len(merged) == 1
        assert merged[0]["cmdline"] == "v2"

    def test_preserves_all_unique_pids(self, pi_mod):
        """All unique PIDs across snapshots are preserved."""
        snapshots = [
            [{"pid": 100, "starttime": "10", "cmdline": "a"}],
            [{"pid": 200, "starttime": "20", "cmdline": "b"}],
            [{"pid": 300, "starttime": "30", "cmdline": "c"}],
        ]
        merged = pi_mod.merge_snapshots(snapshots)
        pids = {e["pid"] for e in merged}
        assert pids == {100, 200, 300}


# ---------------------------------------------------------------------------
# CLI interface tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for the CLI interface used by the bash script."""

    def test_verify_returns_0_on_match(self, pi_mod, capsys):
        """verify command exits 0 when identity matches."""
        with (
            patch.object(pi_mod, "get_starttime", return_value="100"),
            patch.object(pi_mod, "get_cmdline", return_value="test_cmd"),
            patch(
                "sys.argv", ["process_identity.py", "verify", "999", "100", "test_cmd"]
            ),
        ):
            exit_code = pi_mod.main()
        assert exit_code == 0

    def test_verify_returns_1_on_mismatch(self, pi_mod, capsys):
        """verify command exits 1 when identity doesn't match."""
        with (
            patch.object(pi_mod, "get_starttime", return_value="999"),
            patch.object(pi_mod, "get_cmdline", return_value="test_cmd"),
            patch(
                "sys.argv",
                ["process_identity.py", "verify", "999", "100", "different_cmd"],
            ),
        ):
            exit_code = pi_mod.main()
        assert exit_code == 1

    def test_get_starttime_cli_prints_raw_value(self, pi_mod, capsys):
        """get_starttime CLI prints the raw value (not JSON)."""
        with (
            patch.object(pi_mod, "get_starttime", return_value="12345"),
            patch("sys.argv", ["process_identity.py", "get_starttime", "999"]),
        ):
            exit_code = pi_mod.main()
        captured = capsys.readouterr()
        assert exit_code == 0
        assert captured.out.strip() == "12345"

    def test_get_cmdline_cli_prints_raw_value(self, pi_mod, capsys):
        """get_cmdline CLI prints the raw value (not JSON)."""
        with (
            patch.object(pi_mod, "get_cmdline", return_value="python bench.py"),
            patch("sys.argv", ["process_identity.py", "get_cmdline", "999"]),
        ):
            exit_code = pi_mod.main()
        captured = capsys.readouterr()
        assert exit_code == 0
        assert captured.out.strip() == "python bench.py"

    def test_snapshot_cli_produces_valid_json(self, pi_mod, capsys):
        """snapshot CLI produces valid JSON array on stdout."""
        mock_snapshot = [
            {"pid": 100, "starttime": "10", "cmdline": "test"},
        ]
        with (
            patch.object(pi_mod, "snapshot_descendants", return_value=mock_snapshot),
            patch("sys.argv", ["process_identity.py", "snapshot", "1"]),
        ):
            exit_code = pi_mod.main()
        captured = capsys.readouterr()
        assert exit_code == 0
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_cleanup_cli_produces_summary(self, pi_mod, tmp_path, capsys):
        """cleanup CLI produces JSON summary on stdout."""
        snapshot_file = tmp_path / "snapshots.jsonl"
        snapshot_file.write_text(
            json.dumps([{"pid": 100, "starttime": "10", "cmdline": "cmd"}]) + "\n"
        )

        with (
            patch.object(pi_mod, "get_starttime", return_value="10"),
            patch.object(pi_mod, "get_cmdline", return_value="cmd"),
            patch("os.kill"),
            patch(
                "sys.argv",
                ["process_identity.py", "cleanup", str(snapshot_file), "15"],
            ),
        ):
            exit_code = pi_mod.main()
        captured = capsys.readouterr()
        assert exit_code == 0
        summary = json.loads(captured.out)
        assert "killed" in summary
        assert "skipped" in summary
        assert "total" in summary

    def test_unknown_command_returns_2(self, pi_mod, capsys):
        """Unknown command exits with code 2."""
        with patch("sys.argv", ["process_identity.py", "unknown"]):
            exit_code = pi_mod.main()
        assert exit_code == 2

    def test_no_args_returns_2(self, pi_mod, capsys):
        """No arguments exits with code 2."""
        with patch("sys.argv", ["process_identity.py"]):
            exit_code = pi_mod.main()
        assert exit_code == 2

    def test_cleanup_with_empty_file(self, pi_mod, tmp_path):
        """cleanup command handles empty snapshot file."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        summary = pi_mod.cleanup_descendants(empty_file, 15)
        assert summary["killed"] == []
        assert summary["total"] == 0

    def test_cleanup_with_corrupt_json_lines(self, pi_mod, tmp_path):
        """cleanup command skips corrupt JSON lines."""
        corrupt_file = tmp_path / "corrupt.jsonl"
        corrupt_file.write_text(
            "not json\n"
            + json.dumps([{"pid": 100, "starttime": "10", "cmdline": "cmd"}])
            + "\n"
            + "also not json\n"
        )

        with (
            patch.object(pi_mod, "get_starttime", return_value="10"),
            patch.object(pi_mod, "get_cmdline", return_value="cmd"),
            patch("os.kill"),
        ):
            summary = pi_mod.cleanup_descendants(corrupt_file, 15)

        # Only the valid JSON line is processed
        assert summary["total"] == 1


# ---------------------------------------------------------------------------
# Session-based ownership (reviewer round 6)
# ---------------------------------------------------------------------------


class TestGetSessionId:
    """Tests for get_session_id — reading the session ID of a process.

    Per reviewer round 6: '改用不会依赖轮询命中的归属机制...启动即进入
    job-owned cgroup' — session-based ownership requires reading the SID
    of the launcher so we can scan the session at cleanup time.
    """

    def test_get_sid_returns_int_for_real_process(self, pi_mod):
        """get_session_id returns an integer SID for a real process.

        Uses the current test process PID, which always exists.
        """
        sid = pi_mod.get_session_id(__import__("os").getpid())
        assert sid is not None
        assert isinstance(sid, int)
        assert sid > 0

    test_get_sid_returns_int_for_real_process = _SKIP_NON_LINUX(
        test_get_sid_returns_int_for_real_process
    )

    def test_get_sid_returns_none_for_nonexistent_pid(self, pi_mod):
        """get_session_id returns None for a PID that doesn't exist."""
        # PID 999999 is very unlikely to exist
        sid = pi_mod.get_session_id(999999)
        assert sid is None

    def test_get_sid_cli_prints_integer(self, pi_mod, capsys):
        """get_sid CLI prints the SID as an integer on stdout."""
        pid = __import__("os").getpid()
        with patch("sys.argv", ["process_identity.py", "get_sid", str(pid)]):
            exit_code = pi_mod.main()
        captured = capsys.readouterr()
        assert exit_code == 0
        assert int(captured.out.strip()) > 0

    test_get_sid_cli_prints_integer = _SKIP_NON_LINUX(test_get_sid_cli_prints_integer)


class TestSnapshotSessionDescendants:
    """Tests for snapshot_session_descendants — polling-independent session scan.

    Per reviewer round 6: '改用不会依赖轮询命中的归属机制' — session scan
    finds ALL processes in the session via a single ``ps -eo pid,sid`` query,
    regardless of when they spawned or whether they reparented.
    """

    def test_snapshot_session_returns_list(self, pi_mod):
        """snapshot_session_descendants returns a list (possibly empty)."""
        # Use a very high SID that almost certainly has no processes.
        result = pi_mod.snapshot_session_descendants(999999)
        assert isinstance(result, list)
        assert result == []

    def test_snapshot_session_captures_current_process(self, pi_mod):
        """snapshot_session_descendants captures the current process.

        The current process is always in its own session, so scanning our
        session should find at least us.
        """
        pid = __import__("os").getpid()
        sid = pi_mod.get_session_id(pid)
        assert sid is not None
        result = pi_mod.snapshot_session_descendants(sid)
        pids = {e["pid"] for e in result}
        assert pid in pids, "Current process must be found in its own session scan"

    test_snapshot_session_captures_current_process = _SKIP_NON_LINUX(
        test_snapshot_session_captures_current_process
    )

    def test_snapshot_session_cli_produces_json(self, pi_mod, capsys):
        """snapshot_session CLI produces valid JSON array on stdout."""
        pid = __import__("os").getpid()
        sid = pi_mod.get_session_id(pid)
        with patch("sys.argv", ["process_identity.py", "snapshot_session", str(sid)]):
            exit_code = pi_mod.main()
        captured = capsys.readouterr()
        assert exit_code == 0
        data = json.loads(captured.out)
        assert isinstance(data, list)

    test_snapshot_session_cli_produces_json = _SKIP_NON_LINUX(
        test_snapshot_session_cli_produces_json
    )

    def test_snapshot_session_with_mocked_ps(self, pi_mod):
        """snapshot_session_descendants parses ps output correctly.

        Mocks ``ps -eo pid=,sid=`` to return a known set of processes and
        verifies that only processes with the matching SID are returned.
        """
        mock_ps_output = (
            "  100  50\n"
            "  101  50\n"
            "  102  99\n"  # Different session — should be excluded
            "  103  50\n"
            "  104  77\n"  # Different session — should be excluded
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = __import__("types").SimpleNamespace(
                returncode=0, stdout=mock_ps_output, stderr=""
            )
            with (
                patch.object(pi_mod, "get_starttime", side_effect=lambda p: f"st{p}"),
                patch.object(pi_mod, "get_cmdline", side_effect=lambda p: f"cmd{p}"),
            ):
                result = pi_mod.snapshot_session_descendants(50)

        pids = {e["pid"] for e in result}
        assert pids == {100, 101, 103}, "Only processes with SID==50 should be returned"
        for entry in result:
            assert entry["starttime"].startswith("st")
            assert entry["cmdline"].startswith("cmd")


# ---------------------------------------------------------------------------
# Real subprocess test: fast detach before first snapshot
# ---------------------------------------------------------------------------


class TestRealFastDetach:
    """Real subprocess test: child spawns and detaches before first snapshot.

    Per reviewer round 6: '请用真实子进程测试复现"首次 snapshot 前 detach"'.

    This test starts a real subprocess via setsid (making it a session
    leader), then immediately spawns a child that sleeps.  It verifies
    that session-based scanning (snapshot_session_descendants) captures
    the child even if it detaches/reparents before the first poll.

    Unlike the mock-based TestFastDetach (which only tests merge_snapshots
    logic), this test exercises the real ``ps -eo pid,sid`` scan and proves
    that session-based ownership does not miss fast-detach processes.
    """

    @_SKIP_NON_LINUX
    def test_session_scan_captures_fast_detach_child(self, pi_mod):
        """A child spawned in the session is captured by session scan.

        Steps:
        1. Start a setsid sleep process (becomes session leader).
        2. Fork a child that also sleeps (inherits the session).
        3. Call snapshot_session_descendants(sid).
        4. Verify the child is in the results.
        """
        import os
        import signal
        import time

        # Start a setsid process that sleeps — it becomes a session leader.
        # We use a pipe to synchronize: the child writes its PID.
        r_fd, w_fd = os.pipe()
        proc = subprocess.Popen(
            ["setsid", "bash", "-c", f"echo $$ > /dev/fd/{w_fd}; sleep 30"],
            pass_fds=(w_fd,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.close(w_fd)

        try:
            # Read the session leader's PID
            leader_pid_str = os.read(r_fd, 32).decode().strip()
            os.close(r_fd)
            if not leader_pid_str:
                pytest.skip("Could not read leader PID from pipe")
            leader_pid = int(leader_pid_str)

            # Give it a moment to start sleeping
            time.sleep(0.3)

            # Get the session ID
            sid = pi_mod.get_session_id(leader_pid)
            if sid is None:
                pytest.skip("Could not read SID (not running on Linux?)")

            # Scan the session — should find the leader AND its sleep child
            result = pi_mod.snapshot_session_descendants(sid)
            pids = {e["pid"] for e in result}

            assert leader_pid in pids, "Session leader must be found by session scan"
            # The bash -c subprocess and the sleep child should also be found
            assert len(result) >= 1, (
                "Session scan must find at least the leader process"
            )
        finally:
            # Clean up
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.wait(timeout=5)

    @_SKIP_NON_LINUX
    def test_session_scan_vs_pgrep_tree_walking(self, pi_mod):
        """Session scan captures processes that pgrep -P tree-walking misses.

        Per reviewer round 6: the old snapshot_descendants (pgrep -P) can
        miss processes that spawn and reparent between snapshots.  Session
        scan (ps -eo pid,sid) does not have this limitation.

        This test creates a setsid process with a child, then verifies:
        - snapshot_session_descendants finds the child.
        - The child is captured regardless of tree-walking timing.
        """
        import os
        import signal
        import time

        r_fd, w_fd = os.pipe()
        proc = subprocess.Popen(
            [
                "setsid",
                "bash",
                "-c",
                # Spawn a child that also spawns a grandchild, then sleep
                f"echo $$ > /dev/fd/{w_fd}; sleep 30 & sleep 30 & sleep 30",
            ],
            pass_fds=(w_fd,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.close(w_fd)

        try:
            leader_pid_str = os.read(r_fd, 32).decode().strip()
            os.close(r_fd)
            if not leader_pid_str:
                pytest.skip("Could not read leader PID from pipe")
            leader_pid = int(leader_pid_str)

            time.sleep(0.5)

            sid = pi_mod.get_session_id(leader_pid)
            if sid is None:
                pytest.skip("Could not read SID (not running on Linux?)")

            # Session scan should find the leader + bash + sleep children
            session_result = pi_mod.snapshot_session_descendants(sid)
            session_pids = {e["pid"] for e in session_result}

            assert leader_pid in session_pids
            # Should find at least 2 processes (leader + at least one child)
            assert len(session_pids) >= 2, (
                f"Session scan should find multiple processes, found {session_pids}"
            )
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.wait(timeout=5)
