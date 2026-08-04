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
    """Real subprocess test: child spawns and reparents within the session.

    Per reviewer round 6: '请用真实子进程测试复现"首次 snapshot 前 detach"'.

    This test starts a real subprocess via setsid (making it a session
    leader), then immediately spawns a child that sleeps.  It verifies
    that session-based scanning (snapshot_session_descendants) captures
    the child as long as it stays in the session.

    NOTE (reviewer round 7): these tests only cover reparent-WITHIN-session.
    They do NOT cover the setsid case where a child leaves the session.
    For setsid coverage, see TestRealCgroupSetsidCleanup and
    TestSessionScanMissesSetsid.
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


# ---------------------------------------------------------------------------
# Reviewer round 7: session scan CANNOT catch setsid'd processes
# ---------------------------------------------------------------------------


class TestSessionScanMissesSetsid:
    """Reverse test: session scan misses processes that called setsid().

    Per reviewer round 7: 'session scan 只能找到仍属于原 SID 的进程。
    EngineCore 如果自己调用 setsid()，会立即进入新的 session，之后
    ps -eo pid,sid 按 launcher SID 扫描同样找不到它'.

    This test proves that session-based scanning is INSUFFICIENT for
    catching setsid'd processes, justifying the need for cgroup v2.
    """

    @_SKIP_NON_LINUX
    def test_session_scan_misses_setsid_child(self, pi_mod):
        """A child that calls setsid() is NOT found by session scan.

        Steps:
        1. Start a setsid process (becomes session leader, SID=L).
        2. The process forks a child that calls setsid() (enters new session).
        3. Session scan of SID=L does NOT find the child.
        4. This proves session scan is insufficient.
        """
        import os
        import signal
        import time

        r_fd, w_fd = os.pipe()
        # The launcher starts a child that calls setsid() and sleeps.
        # The child writes its PID and new SID to the pipe.
        proc = subprocess.Popen(
            [
                "setsid",
                "bash",
                "-c",
                f"echo $$ > /dev/fd/{w_fd}; "
                # Fork a child that setsid's into a new session
                "bash -c 'setsid sleep 30 & echo $! > /tmp/setsid_child_pid; "
                "echo $$ > /tmp/setsid_child_sid' &"
                "sleep 30",
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

            time.sleep(1.0)  # Give the setsid child time to start

            sid = pi_mod.get_session_id(leader_pid)
            if sid is None:
                pytest.skip("Could not read SID (not running on Linux?)")

            # Session scan of the launcher's SID
            session_result = pi_mod.snapshot_session_descendants(sid)
            session_pids = {e["pid"] for e in session_result}

            # The setsid'd child should NOT be in the session scan
            # (it left the session by calling setsid)
            # Read the child's PID if available
            try:
                with open("/tmp/setsid_child_pid") as f:
                    child_pid = int(f.read().strip())
                assert child_pid not in session_pids, (
                    f"setsid'd child (pid {child_pid}) should NOT be found "
                    f"by session scan of SID {sid}, but it was. "
                    f"This means the child did not actually leave the session."
                )
            except (FileNotFoundError, ValueError):
                pytest.skip("Could not read setsid child PID")

        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.wait(timeout=5)
            # Also kill any orphaned setsid'd children
            try:
                with open("/tmp/setsid_child_pid") as f:
                    child_pid = int(f.read().strip())
                os.kill(child_pid, signal.SIGKILL)
            except (FileNotFoundError, ValueError, ProcessLookupError, OSError):
                pass


# ---------------------------------------------------------------------------
# Cgroup v2 based attribution (reviewer round 7)
# ---------------------------------------------------------------------------

# Cgroup v2 tests require Linux and writable /sys/fs/cgroup.
_SKIP_NO_CGROUP = pytest.mark.skipif(
    platform.system() != "Linux"
    or not Path("/sys/fs/cgroup/cgroup.controllers").is_file(),
    reason="cgroup v2 is required (not available on macOS or without cgroup v2)",
)


class TestCgroupFunctions:
    """Unit tests for cgroup v2 functions (mock-based, run on all platforms).

    Per reviewer round 7: '实现上需要 cgroup、runtime 同步 registry，或其他
    在它离开 launcher SID 前就完成的不可漏归属机制'.
    """

    def test_is_cgroup_v2_available_returns_bool(self, pi_mod):
        """is_cgroup_v2_available returns a bool."""
        result = pi_mod.is_cgroup_v2_available()
        assert isinstance(result, bool)

    def test_get_writable_cgroup_parent_returns_path_or_none(self, pi_mod):
        """get_writable_cgroup_parent returns a Path or None."""
        result = pi_mod.get_writable_cgroup_parent()
        assert result is None or isinstance(result, Path)

    def test_create_job_cgroup_returns_path_or_none(self, pi_mod):
        """create_job_cgroup returns a Path or None."""
        result = pi_mod.create_job_cgroup("test_unit")
        assert result is None or isinstance(result, Path)
        # Clean up if created
        if result is not None:
            pi_mod.remove_cgroup(result)

    @_SKIP_NO_CGROUP
    def test_create_add_snapshot_remove_cgroup(self, pi_mod):
        """End-to-end: create cgroup, add PID, snapshot, remove.

        This test creates a real cgroup, adds the current process,
        verifies the snapshot contains the current PID, then removes
        the cgroup.
        """
        import os

        job_id = f"test_{os.getpid()}_{id(pi_mod)}"
        cgroup_path = pi_mod.create_job_cgroup(job_id)
        if cgroup_path is None:
            pytest.skip("Could not create cgroup (no writable parent)")

        try:
            # Add current process to cgroup
            pid = os.getpid()
            assert pi_mod.add_pid_to_cgroup(pid, cgroup_path) is True

            # Snapshot should include the current process
            snapshots = pi_mod.snapshot_cgroup_descendants(cgroup_path)
            pids = {e["pid"] for e in snapshots}
            assert pid in pids, (
                f"Current process (pid {pid}) must be in cgroup snapshot, "
                f"got pids: {pids}"
            )
        finally:
            # Remove cgroup (may fail if processes are still in it)
            pi_mod.remove_cgroup(cgroup_path)

    def test_snapshot_cgroup_descendants_handles_missing_path(self, pi_mod):
        """snapshot_cgroup_descendants returns [] for non-existent path."""
        result = pi_mod.snapshot_cgroup_descendants(Path("/nonexistent/cgroup"))
        assert result == []

    def test_cleanup_cgroup_descendants_empty_cgroup(self, pi_mod, tmp_path):
        """cleanup_cgroup_descendants on empty cgroup returns zero killed."""
        # Mock the snapshot to return empty list
        with patch.object(pi_mod, "snapshot_cgroup_descendants", return_value=[]):
            summary = pi_mod.cleanup_cgroup_descendants(tmp_path, 15)
        assert summary["killed"] == []
        assert summary["total"] == 0

    def test_cleanup_cgroup_descendants_kills_matching(self, pi_mod, tmp_path):
        """cleanup_cgroup_descendants kills processes with matching identity."""
        mock_snapshots = [
            {"pid": 12345, "starttime": "100", "cmdline": "sleep 30"},
        ]
        killed_pids: list[int] = []

        with (
            patch.object(
                pi_mod,
                "snapshot_cgroup_descendants",
                return_value=mock_snapshots,
            ),
            patch.object(pi_mod, "verify_identity", return_value=True),
            patch("os.kill", side_effect=lambda pid, sig: killed_pids.append(pid)),
        ):
            summary = pi_mod.cleanup_cgroup_descendants(tmp_path, 15)

        assert 12345 in killed_pids
        assert summary["killed"] == [12345]

    def test_cleanup_cgroup_descendants_skips_mismatch(self, pi_mod, tmp_path):
        """cleanup_cgroup_descendants skips processes with mismatched identity."""
        mock_snapshots = [
            {"pid": 12345, "starttime": "100", "cmdline": "sleep 30"},
        ]

        with (
            patch.object(
                pi_mod,
                "snapshot_cgroup_descendants",
                return_value=mock_snapshots,
            ),
            patch.object(pi_mod, "verify_identity", return_value=False),
            patch("os.kill"),
        ):
            summary = pi_mod.cleanup_cgroup_descendants(tmp_path, 15)

        assert summary["killed"] == []
        assert summary["skipped"] >= 1

    def test_remove_cgroup_returns_bool(self, pi_mod, tmp_path):
        """remove_cgroup returns a bool."""
        result = pi_mod.remove_cgroup(tmp_path)
        assert isinstance(result, bool)

    def test_cgroup_available_cli(self, pi_mod, capsys):
        """cgroup_available CLI exits 0 or 1."""
        with patch("sys.argv", ["process_identity.py", "cgroup_available"]):
            exit_code = pi_mod.main()
        assert exit_code in (0, 1)

    def test_create_cgroup_cli(self, pi_mod, capsys):
        """create_cgroup CLI prints path or nothing."""
        with patch("sys.argv", ["process_identity.py", "create_cgroup", "test_cli"]):
            exit_code = pi_mod.main()
        captured = capsys.readouterr()
        # Exit 0 with path, or exit 1 with empty output
        if exit_code == 0:
            assert len(captured.out.strip()) > 0
            # Clean up
            cgroup_path = Path(captured.out.strip())
            if cgroup_path.is_dir():
                pi_mod.remove_cgroup(cgroup_path)
        else:
            assert exit_code == 1

    def test_pid_in_cgroup_returns_false_for_missing_path(self, pi_mod):
        """pid_in_cgroup returns False for a non-existent cgroup path."""
        result = pi_mod.pid_in_cgroup(99999, Path("/nonexistent/cgroup/path"))
        assert result is False

    def test_pid_in_cgroup_returns_true_when_pid_listed(self, pi_mod, tmp_path):
        """pid_in_cgroup returns True when the PID is in cgroup.procs."""
        procs_file = tmp_path / "cgroup.procs"
        procs_file.write_text("12345\n67890\n")
        assert pi_mod.pid_in_cgroup(12345, tmp_path) is True
        assert pi_mod.pid_in_cgroup(67890, tmp_path) is True

    def test_pid_in_cgroup_returns_false_when_pid_not_listed(self, pi_mod, tmp_path):
        """pid_in_cgroup returns False when the PID is NOT in cgroup.procs."""
        procs_file = tmp_path / "cgroup.procs"
        procs_file.write_text("12345\n67890\n")
        assert pi_mod.pid_in_cgroup(99999, tmp_path) is False

    def test_pid_in_cgroup_cli_exit_codes(self, pi_mod, tmp_path):
        """pid_in_cgroup CLI exits 0 if PID is in cgroup, 1 if not."""
        procs_file = tmp_path / "cgroup.procs"
        procs_file.write_text(f"{__import__('os').getpid()}\n")
        with patch(
            "sys.argv",
            [
                "process_identity.py",
                "pid_in_cgroup",
                str(__import__("os").getpid()),
                str(tmp_path),
            ],
        ):
            assert pi_mod.main() == 0
        with patch(
            "sys.argv",
            ["process_identity.py", "pid_in_cgroup", "99999", str(tmp_path)],
        ):
            assert pi_mod.main() == 1

    def test_exec_in_cgroup_fails_closed_on_join_error(self, pi_mod, tmp_path):
        """exec_in_cgroup returns 127 when cgroup join fails (fail closed).

        Per reviewer round 8: '加入失败时 fail closed' — if writing to
        cgroup.procs fails, the wrapper must NOT exec the command.
        """
        ready_file = tmp_path / "ready"
        with (
            patch("pathlib.Path.write_text", side_effect=PermissionError("denied")),
            patch("os.execvp") as mock_exec,
        ):
            rc = pi_mod.exec_in_cgroup(tmp_path, ready_file, ["echo", "hello"])
        assert rc == 127
        mock_exec.assert_not_called()
        assert not ready_file.exists()

    def test_exec_in_cgroup_fails_closed_on_membership_check(self, pi_mod, tmp_path):
        """exec_in_cgroup returns 127 when membership verification fails.

        Even if write_text succeeds, if pid_in_cgroup returns False (write
        was silently ignored), the wrapper must fail closed.
        """
        ready_file = tmp_path / "ready"
        with (
            patch.object(pi_mod, "pid_in_cgroup", return_value=False),
            patch("os.execvp") as mock_exec,
        ):
            rc = pi_mod.exec_in_cgroup(tmp_path, ready_file, ["echo", "hello"])
        assert rc == 127
        mock_exec.assert_not_called()

    def test_exec_in_cgroup_writes_ready_file_and_execs(self, pi_mod, tmp_path):
        """exec_in_cgroup writes ready_file and calls os.execvp on success."""
        ready_file = tmp_path / "ready"
        with (
            patch.object(pi_mod, "pid_in_cgroup", return_value=True),
            patch("os.execvp") as mock_exec,
        ):
            # On success, exec replaces the process and does not return.
            # Since execvp is mocked, the function falls through to the
            # unreachable return 127 — we ignore the return value and
            # assert the side effects instead.
            pi_mod.exec_in_cgroup(tmp_path, ready_file, ["echo", "hello"])
        # 1. ready_file was written (join signaled to parent)
        assert ready_file.exists()
        # 2. os.execvp was called with the correct command
        mock_exec.assert_called_once_with("echo", ["echo", "hello"])

    def test_exec_in_cgroup_returns_127_on_exec_failure(self, pi_mod, tmp_path):
        """exec_in_cgroup returns 127 if os.execvp raises OSError."""
        ready_file = tmp_path / "ready"
        with (
            patch.object(pi_mod, "pid_in_cgroup", return_value=True),
            patch("os.execvp", side_effect=OSError("no such command")),
        ):
            rc = pi_mod.exec_in_cgroup(tmp_path, ready_file, ["nonexistent_cmd"])
        assert rc == 127


class TestRealCgroupSetsidCleanup:
    """Real subprocess tests: descendant setsid + launcher exit + cgroup cleanup.

    Per reviewer round 7: '请让真实测试中的 descendant 确实建立新 session
    并让 launcher 先退出，确认清理仍能凭已同步登记的 PID/starttime/cmdline
    找到它；实现上需要 cgroup、runtime 同步 registry，或其他在它离开
    launcher SID 前就完成的不可漏归属机制。仅靠结束时扫描原 SID 无法补回
    已经离开的进程。'

    These tests prove that cgroup v2 catches setsid'd processes that session
    scan cannot.  They require:
    - Linux with cgroup v2
    - Writable /sys/fs/cgroup (root or systemd user delegation)
    - setsid command available
    """

    @_SKIP_NON_LINUX
    @_SKIP_NO_CGROUP
    def test_cgroup_catches_setsid_descendant(self, pi_mod):
        """Cgroup catches a descendant that called setsid().

        Per reviewer round 8: '请用受控 wrapper/握手让 launcher 自身在 exec
        benchmark、产生任何后代之前成功加入 job cgroup' — uses exec_in_cgroup
        to ensure the launcher joins the cgroup BEFORE forking descendants.

        Steps:
        1. Create a cgroup.
        2. Use exec_in_cgroup to start a launcher that joins the cgroup
           BEFORE exec'ing bash.
        3. The launcher forks a child that calls setsid() (new session).
        4. Read cgroup.procs — should include the setsid'd child via
           fork-time inheritance (NOT direct migration).
        5. Verify session scan of the launcher's SID does NOT find the child.
        """
        import os
        import signal
        import tempfile
        import time

        parent = pi_mod.get_writable_cgroup_parent()
        if parent is None:
            pytest.skip("No writable cgroup parent (requires root or delegation)")

        job_id = f"test_setsid_{os.getpid()}_{int(time.time())}"
        cgroup_path = pi_mod.create_job_cgroup(job_id)
        if cgroup_path is None:
            pytest.skip("Could not create cgroup")

        child_pid_file = tempfile.mktemp()
        ready_file = tempfile.mktemp()
        os.unlink(ready_file)  # remove so wrapper can create it

        try:
            # Launcher script: fork a setsid'd child, write its PID, stay alive.
            launcher_script = (
                f"setsid bash -c 'sleep 30' & echo $! > {child_pid_file}; sleep 30"
            )

            # Start via exec_in_cgroup: wrapper joins cgroup, then execs bash.
            # The launcher (bash) is in the cgroup BEFORE forking the child,
            # so the child inherits membership at fork time.
            proc = subprocess.Popen(
                [
                    "setsid",
                    sys.executable,
                    str(SCRIPT_PATH),
                    "exec_in_cgroup",
                    str(cgroup_path),
                    ready_file,
                    "bash",
                    "-c",
                    launcher_script,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for the wrapper to signal cgroup join.
            waited = 0
            while not os.path.exists(ready_file) and waited < 50:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                waited += 1

            if not os.path.exists(ready_file):
                pytest.skip("Wrapper did not join cgroup (may lack permissions)")

            # Wait for the child PID to be written.
            waited = 0
            while not os.path.exists(child_pid_file) and waited < 50:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                waited += 1

            if not os.path.exists(child_pid_file):
                pytest.skip("Could not read child PID")

            with open(child_pid_file) as f:
                child_pid = int(f.read().strip())

            time.sleep(0.5)  # Give the setsid child time to settle

            # CGROUP scan: should find the setsid'd child via INHERITANCE.
            cgroup_snapshots = pi_mod.snapshot_cgroup_descendants(cgroup_path)
            cgroup_pids = {e["pid"] for e in cgroup_snapshots}
            assert child_pid in cgroup_pids, (
                f"Cgroup scan MUST find setsid'd child (pid {child_pid}) via "
                f"fork-time inheritance. Got pids: {cgroup_pids}"
            )

            # SESSION scan: should NOT find the setsid'd child (it left the
            # session by calling setsid).  This proves cgroup is superior.
            launcher_sid = pi_mod.get_session_id(proc.pid)
            if launcher_sid is not None:
                session_snapshots = pi_mod.snapshot_session_descendants(launcher_sid)
                session_pids = {e["pid"] for e in session_snapshots}
                assert child_pid not in session_pids, (
                    f"Session scan should NOT find setsid'd child (pid {child_pid}), "
                    f"but it did. The child may not have actually called setsid()."
                )

        finally:
            pi_mod.cleanup_cgroup_descendants(cgroup_path, signal.SIGKILL)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.wait(timeout=5)
            pi_mod.remove_cgroup(cgroup_path)
            for f in [child_pid_file, ready_file]:
                try:
                    os.unlink(f)
                except OSError:
                    pass

    @_SKIP_NON_LINUX
    @_SKIP_NO_CGROUP
    def test_cgroup_cleanup_kills_setsid_after_launcher_exit(self, pi_mod):
        """Cleanup kills setsid'd descendant AFTER launcher exits.

        Per reviewer round 8: '再用真实测试只迁 launcher、让它之后
        fork→setsid→exit，确认 child 从继承关系进入 cgroup并被清理，
        不要直接迁 child。'

        This test ONLY migrates the launcher (via exec_in_cgroup).  The
        child enters the cgroup via fork-time inheritance — NOT via direct
        migration.  This proves the cgroup attribution mechanism works
        correctly for the real startup sequence.

        Steps:
        1. Create a cgroup.
        2. Use exec_in_cgroup to start a launcher that joins the cgroup
           BEFORE exec'ing bash.
        3. The launcher forks a child that calls setsid() (new session).
        4. The launcher EXITS immediately.
        5. The setsid'd child is still alive (orphaned, reparented to init).
        6. cleanup_cgroup_descendants finds and kills it via cgroup.
        """
        import os
        import signal
        import tempfile
        import time

        parent = pi_mod.get_writable_cgroup_parent()
        if parent is None:
            pytest.skip("No writable cgroup parent (requires root or delegation)")

        job_id = f"test_exit_{os.getpid()}_{int(time.time())}"
        cgroup_path = pi_mod.create_job_cgroup(job_id)
        if cgroup_path is None:
            pytest.skip("Could not create cgroup")

        child_pid_file = tempfile.mktemp()
        ready_file = tempfile.mktemp()
        os.unlink(ready_file)  # remove so wrapper can create it

        try:
            # Launcher script: fork a setsid'd child, write its PID, EXIT.
            # The child survives as an orphan in a new session.
            launcher_script = f"setsid bash -c 'sleep 30' & echo $! > {child_pid_file}"

            # Start via exec_in_cgroup: wrapper joins cgroup, then execs bash.
            # The launcher (bash) is in the cgroup BEFORE forking the child,
            # so the child inherits membership at fork time.
            proc = subprocess.Popen(
                [
                    "setsid",
                    sys.executable,
                    str(SCRIPT_PATH),
                    "exec_in_cgroup",
                    str(cgroup_path),
                    ready_file,
                    "bash",
                    "-c",
                    launcher_script,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for the wrapper to signal cgroup join.
            waited = 0
            while not os.path.exists(ready_file) and waited < 50:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                waited += 1

            if not os.path.exists(ready_file):
                pytest.skip("Wrapper did not join cgroup (may lack permissions)")

            # Wait for the child PID to be written.
            waited = 0
            while not os.path.exists(child_pid_file) and waited < 50:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                waited += 1

            if not os.path.exists(child_pid_file):
                pytest.skip("Could not read child PID")

            with open(child_pid_file) as f:
                child_pid = int(f.read().strip())

            # Wait for the launcher to exit.
            proc.wait(timeout=5)

            time.sleep(0.5)  # Ensure child is reparented

            # The child should still be alive (orphaned, in a new session).
            assert pi_mod.get_starttime(child_pid) is not None, (
                "Setsid'd child should still be alive after launcher exit"
            )

            # Verify the child is in the cgroup via INHERITANCE (not direct
            # migration).  Per reviewer round 8: '不要直接迁 child'.
            assert pi_mod.pid_in_cgroup(child_pid, cgroup_path), (
                f"Child (pid {child_pid}) must be in cgroup via fork-time "
                f"inheritance from the launcher, NOT direct migration."
            )

            # CGROUP cleanup should find and kill the child.
            summary = pi_mod.cleanup_cgroup_descendants(cgroup_path, signal.SIGTERM)
            assert child_pid in summary["killed"], (
                f"Cgroup cleanup must kill setsid'd child (pid {child_pid}) "
                f"after launcher exit. Summary: {summary}"
            )

            # Verify the child is dead.
            time.sleep(0.5)
            assert pi_mod.get_starttime(child_pid) is None, (
                "Setsid'd child must be dead after cgroup cleanup"
            )

        finally:
            # Ensure cleanup
            pi_mod.cleanup_cgroup_descendants(cgroup_path, signal.SIGKILL)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.wait(timeout=5)
            pi_mod.remove_cgroup(cgroup_path)
            for f in [child_pid_file, ready_file]:
                try:
                    os.unlink(f)
                except OSError:
                    pass

    @_SKIP_NON_LINUX
    @_SKIP_NO_CGROUP
    def test_cgroup_cleanup_pid_reuse_not_killed(self, pi_mod):
        """Cgroup cleanup does NOT kill processes with mismatched identity.

        Per reviewer round 7: PID reuse safety — even with cgroup, we verify
        PID + starttime + cmdline before killing.

        Steps:
        1. Create a cgroup and add a process.
        2. The process exits.
        3. A different process (different starttime/cmdline) somehow has the
           same PID (simulated by mocking verify_identity).
        4. cleanup_cgroup_descendants does NOT kill it.
        """
        import os
        import time

        parent = pi_mod.get_writable_cgroup_parent()
        if parent is None:
            pytest.skip("No writable cgroup parent")

        job_id = f"test_reuse_{os.getpid()}_{int(time.time())}"
        cgroup_path = pi_mod.create_job_cgroup(job_id)
        if cgroup_path is None:
            pytest.skip("Could not create cgroup")

        try:
            # Add current process to cgroup
            pid = os.getpid()
            assert pi_mod.add_pid_to_cgroup(pid, cgroup_path) is True

            # Mock verify_identity to return False (simulating PID reuse)
            killed_pids: list[int] = []
            with (
                patch.object(pi_mod, "verify_identity", return_value=False),
                patch("os.kill", side_effect=lambda p, s: killed_pids.append(p)),
            ):
                summary = pi_mod.cleanup_cgroup_descendants(cgroup_path, 15)

            assert pid not in killed_pids, (
                "Process with mismatched identity must NOT be killed"
            )
            assert summary["killed"] == []

        finally:
            pi_mod.remove_cgroup(cgroup_path)
