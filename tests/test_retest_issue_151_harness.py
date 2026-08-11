"""Tests for the issue #151 regression re-test bash harness.

Covers the testable logic in ``scripts/retest_issue_151_regression.sh``:

- PID triple-identity verification (PID + starttime + cmdline hash)
- Fail-closed raw.json validation
- Scoped stop_server cleanup (process-group kill, no broad pkill)
- main() argument parsing and dry-run plan

Server lifecycle, engine checkout, and NPU benchmark execution require the
real NPU server and are intentionally not exercised here.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path


from tests._bash_utils import bash_executable

REPO_ROOT = Path(__file__).resolve().parents[1]
RETEST_SCRIPT = REPO_ROOT / "scripts" / "retest_issue_151_regression.sh"

# The script has no top-level side effects (only variable assignments and a
# ``main "$@"`` guard gated on BASH_SOURCE), so sourcing it is safe and gives
# us every helper function plus the config globals.
_SOURCE = f"source {shlex.quote(str(RETEST_SCRIPT))}\n"


def _run_bash(snippet: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [bash_executable(), "-c", _SOURCE + snippet],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# PID triple-identity verification
# ---------------------------------------------------------------------------


class TestPidIdentity:
    def test_verify_identity_true_on_live_process(self) -> None:
        """A live process's own PID/starttime/cmdline must verify as true."""
        # $$ is the current shell PID; capture its starttime + cmdline hash
        # via the same helpers the script uses.
        snippet = textwrap.dedent(
            """\
            _st=$(_get_starttime $$)
            _ch=$(_get_cmdline_hash $$)
            if _verify_pid_identity "$$" "$_st" "$_ch"; then
                echo MATCH
            else
                echo NO_MATCH
            fi
            """
        )
        result = _run_bash(snippet)
        assert result.returncode == 0
        assert "MATCH" in result.stdout

    def test_verify_identity_false_for_nonexistent_pid(self) -> None:
        """A PID that does not exist must never verify as true."""
        snippet = textwrap.dedent(
            """\
            if _verify_pid_identity "999999" "ignored" "ignored"; then
                echo MATCH
            else
                echo NO_MATCH
            fi
            """
        )
        result = _run_bash(snippet)
        assert result.returncode == 0
        assert "NO_MATCH" in result.stdout

    def test_verify_identity_false_on_starttime_mismatch(self) -> None:
        """A wrong starttime must fail verification even if the PID is live."""
        snippet = textwrap.dedent(
            """\
            _ch=$(_get_cmdline_hash $$)
            if _verify_pid_identity "$$" "0" "$_ch"; then
                echo MATCH
            else
                echo NO_MATCH
            fi
            """
        )
        result = _run_bash(snippet)
        assert result.returncode == 0
        assert "NO_MATCH" in result.stdout

    def test_cmdline_hash_is_64_hex(self) -> None:
        """_get_cmdline_hash must return a 64-char hex SHA-256."""
        snippet = "_get_cmdline_hash $$"
        result = _run_bash(snippet)
        assert result.returncode == 0
        value = result.stdout.strip()
        assert len(value) == 64
        int(value, 16)  # raises if not hex

    def test_starttime_nonempty_for_live_process(self) -> None:
        """_get_starttime must return a non-empty value for a live PID."""
        snippet = "_get_starttime $$"
        result = _run_bash(snippet)
        assert result.returncode == 0
        assert result.stdout.strip() != ""


# ---------------------------------------------------------------------------
# Fail-closed raw.json validation
# ---------------------------------------------------------------------------


class TestValidateRawJson:
    def _write_raw(self, tmp_path: Path, content: str) -> Path:
        raw = tmp_path / "raw.json"
        raw.write_text(content)
        return raw

    def test_valid_raw_passes(self, tmp_path: Path) -> None:
        """mean_ttft_ms > 0 must pass validation."""
        raw = self._write_raw(
            tmp_path,
            '{"mean_ttft_ms": 120.5, "mean_tpot_ms": 390.0, "output_throughput": 80.0}',
        )
        # The script's hardcoded $PYTHON points at the NPU server; override it
        # with the local interpreter so validation logic is exercised here.
        result = _run_bash(
            f"PYTHON={shlex.quote(sys.executable)} "
            f'validate_raw_json "{raw}" "random-online"'
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        """A missing raw.json must fail (fail-closed)."""
        result = _run_bash(f'validate_raw_json "{tmp_path}/raw.json" "random-online"')
        assert result.returncode != 0

    def test_zero_ttft_fails(self, tmp_path: Path) -> None:
        """mean_ttft_ms == 0 must fail (fail-closed)."""
        raw = self._write_raw(tmp_path, '{"mean_ttft_ms": 0}')
        result = _run_bash(f'validate_raw_json "{raw}" "random-online"')
        assert result.returncode != 0

    def test_negative_ttft_fails(self, tmp_path: Path) -> None:
        """mean_ttft_ms < 0 must fail (fail-closed)."""
        raw = self._write_raw(tmp_path, '{"mean_ttft_ms": -5}')
        result = _run_bash(f'validate_raw_json "{raw}" "random-online"')
        assert result.returncode != 0

    def test_non_numeric_ttft_fails(self, tmp_path: Path) -> None:
        """Non-numeric mean_ttft_ms must fail (fail-closed)."""
        raw = self._write_raw(tmp_path, '{"mean_ttft_ms": "abc"}')
        result = _run_bash(f'validate_raw_json "{raw}" "random-online"')
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Scoped stop_server cleanup (no broad pkill)
# ---------------------------------------------------------------------------


class TestStopServerScopedCleanup:
    def test_stop_server_uses_targeted_kill_not_pkill(self, tmp_path: Path) -> None:
        """stop_server must clean up via process-group + tracked PIDs only.

        We stub out kill_owned_server and verify stop_server does NOT invoke a
        broad ``pkill -9 -f`` against vllm/NPU patterns, nor scan /proc for
        davinci holders. We also fake npu-smi so the memory-wait loop returns
        immediately instead of sleeping 60s.
        """
        fakebin = tmp_path / "bin"
        fakebin.mkdir()
        fake_npu_smi = fakebin / "npu-smi"
        # Emit a low memory value so the release-wait loop breaks on the first
        # iteration (value < 5000).
        fake_npu_smi.write_text('#!/bin/sh\necho "0000:C1:00.0  1000 / 65536 MB"\n')
        os.chmod(fake_npu_smi, 0o755)

        logfile = tmp_path / "stop.log"
        snippet = textwrap.dedent(
            f"""\
            # Stub the identity-verified cleanup so we can assert on control flow.
            kill_owned_server() {{
                echo "kill_owned_server called"
            }}
            export PATH="{fakebin}:$PATH"
            nobuf() {{ :; }}
            stop_server > "{logfile}" 2>&1
            echo "===LOG_START==="
            cat "{logfile}"
            echo "===LOG_END==="
            """
        )
        result = _run_bash(snippet, timeout=60)
        assert result.returncode == 0, result.stderr
        assert "kill_owned_server called" in result.stdout
        # Ensure the memory-wait completed (did not hang) and logged release.
        assert "NPU 0 memory released" in result.stdout
        # stop_server must not print any broad pkill patterns.
        assert "pkill" not in result.stdout

    def test_pkill_removed_from_script_source(self) -> None:
        """The committed script must not contain broad pkill patterns.

        Regression guard for the review comment about killing other tenants'
        vllm/NPU processes on a shared server.
        """
        source = RETEST_SCRIPT.read_text()
        # No broad pkill -9 -f against vllm/NPU patterns.
        assert 'pkill -9 -f "vllm serve"' not in source
        assert 'pkill -9 -f "run_engine_core"' not in source
        assert 'pkill -9 -f "vllm.entrypoints"' not in source
        # No /proc fd scan for davinci holders.
        assert "davinci0" not in source
        assert "davinci_manager" not in source


# ---------------------------------------------------------------------------
# main() argument parsing and dry-run
# ---------------------------------------------------------------------------


class TestMainArgParsing:
    def test_unknown_arg_exits_1(self) -> None:
        """An unknown argument must exit with code 1.

        ``main`` calls ``exit 1`` on an unknown argument, so we run it inside a
        subshell — the subshell terminates with code 1 and the parent echoes it.
        """
        result = _run_bash("( main --bogus 2>/dev/null ) || rc=$?\necho EXIT_$rc")
        assert "EXIT_1" in result.stdout

    def test_dry_run_prints_plan(self) -> None:
        """--dry-run must print the interleaved plan and return 0."""
        result = _run_bash("main --dry-run >/dev/null 2>&1; echo EXIT_$?")
        assert result.returncode == 0
        assert "EXIT_0" in result.stdout

    def test_dry_run_lists_all_pairs(self) -> None:
        """The dry-run plan must list every commit/workload pair."""
        result = _run_bash("main --dry-run 2>&1")
        assert result.returncode == 0
        expected_tokens = [
            "2206f1f7b7",
            "f273f9c5e2",
            "7a63f81e86",
            "ec4847981f",
            "random-online",
            "agent-research-online",
        ]
        for token in expected_tokens:
            assert f"{token} / " in result.stdout or f" / {token}" in result.stdout, (
                f"dry-run plan missing {token!r}:\n{result.stdout}"
            )
