"""Test that CI/CD release gates enforce the trend coverage contract.

These tests verify that:
1. The validate-trend CLI command rejects invalid entries with non-zero exit code.
2. The validate-trend CLI command accepts valid entries with zero exit code.
3. The scripts/validate_trend_entries.py script behaves identically (parity).
4. Both entry points produce consistent admission decisions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "validate_trend_entries.py"
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "trend_coverage"

VALID_DIR = FIXTURE_ROOT / "valid"
INVALID_DIR = FIXTURE_ROOT / "invalid"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Run the validate-trend CLI subcommand and return the result."""
    cmd = [sys.executable, "-m", "vllm_hust_benchmark.cli", "validate-trend"] + args
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def _run_script(args: list[str]) -> subprocess.CompletedProcess:
    """Run scripts/validate_trend_entries.py and return the result."""
    cmd = [sys.executable, str(SCRIPT_PATH)] + args
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def _entry_ids_from_output(output: str) -> set[str]:
    """Extract entry IDs from the output lines like '  default  uuid-...: ...'."""
    ids = set()
    for line in output.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and "-" in parts[1]:
            ids.add(parts[1])
    return ids


# ---------------------------------------------------------------------------
# CLI tests — validate-trend command
# ---------------------------------------------------------------------------

class TestCliValidateTrend:
    """Tests for the ``validate-trend`` CLI subcommand (manual patching entry point)."""

    def test_valid_fixtures_pass(self) -> None:
        """All valid trend fixtures must pass validation with exit code 0."""
        result = _run_cli(["--input", str(VALID_DIR)])
        assert result.returncode == 0, (
            f"CLI returned {result.returncode} for valid fixtures\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
        # All 18 entries in valid fixtures should be accounted for
        ids = _entry_ids_from_output(result.stdout)
        assert len(ids) == 18, f"Expected 18 entries, got {len(ids)}: {ids}"

    def test_invalid_fixtures_fail(self) -> None:
        """All invalid trend fixtures must be rejected with non-zero exit code."""
        result = _run_cli(["--input", str(INVALID_DIR)])
        assert result.returncode != 0, (
            f"CLI returned 0 for invalid fixtures — should have failed\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
        # All 4 invalid fixtures should be listed as invalid
        ids = _entry_ids_from_output(result.stdout)
        assert len(ids) == 4, f"Expected 4 invalid entries, got {len(ids)}: {ids}"

    def test_valid_single_file_passes(self) -> None:
        """A single valid entry file must pass."""
        path = VALID_DIR / "experimental.json"
        result = _run_cli(["--input", str(path)])
        assert result.returncode == 0, (
            f"CLI returned {result.returncode} for {path}\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )

    def test_invalid_single_file_fails(self) -> None:
        """A single invalid entry file must be rejected."""
        path = INVALID_DIR / "bad-version.json"
        result = _run_cli(["--input", str(path)])
        assert result.returncode != 0, (
            f"CLI returned 0 for {path} — expected failure\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
        assert "SCHEMA_INVALID" in result.stdout, "Expected SCHEMA_INVALID error"

    def test_nonexistent_input_fails_gracefully(self) -> None:
        """A non-existent input path should fail with a clear error."""
        result = _run_cli(["--input", str(ROOT / "tests" / "nonexistent.json")])
        assert result.returncode == 2, (
            f"Expected exit code 2 for missing input, got {result.returncode}"
        )

    def test_empty_directory_produces_no_errors(self) -> None:
        """An empty directory with no JSON files should exit 0."""
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            result = _run_cli(["--input", tmpdir])
        assert result.returncode == 0

    def test_print_diagnostics_entry_id(self) -> None:
        """Output must include entry IDs for actionable diagnostics."""
        result = _run_cli(["--input", str(INVALID_DIR)])
        # Each invalid entry should have an entry_id in the output
        assert "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa" in result.stdout
        assert "77777777-7777-4777-8777-777777777777" in result.stdout

    def test_print_diagnostics_error_code(self) -> None:
        """Output must include error codes like SCHEMA_INVALID."""
        result = _run_cli(["--input", str(INVALID_DIR)])
        assert "SCHEMA_INVALID" in result.stdout

    def test_print_diagnostics_release_gate_message(self) -> None:
        """On failure, output must clearly indicate the release gate is blocked."""
        result = _run_cli(["--input", str(INVALID_DIR)])
        assert "release gate blocked" in result.stderr or "release gate blocked" in result.stdout


# ---------------------------------------------------------------------------
# Script parity tests — scripts/validate_trend_entries.py must behave identically
# ---------------------------------------------------------------------------

class TestScriptParity:
    """The legacy script entry point must produce the same results as the CLI."""

    def test_script_valid_fixtures_pass(self) -> None:
        """The script must pass on all valid fixtures (same as CLI)."""
        cli = _run_cli(["--input", str(VALID_DIR)])
        script = _run_script(["--input", str(VALID_DIR)])
        assert script.returncode == 0, (
            f"Script returned {script.returncode} for valid fixtures\n"
            f"stderr: {script.stderr}\nstdout: {script.stdout}"
        )
        # Both exit 0
        assert cli.returncode == script.returncode

    def test_script_invalid_fixtures_fail(self) -> None:
        """The script must reject invalid fixtures (same as CLI)."""
        cli = _run_cli(["--input", str(INVALID_DIR)])
        script = _run_script(["--input", str(INVALID_DIR)])
        assert script.returncode != 0, (
            f"Script returned 0 for invalid fixtures — expected failure\n"
            f"stderr: {script.stderr}\nstdout: {script.stdout}"
        )
        # Both non-zero
        assert (cli.returncode != 0) == (script.returncode != 0)

    def test_script_and_cli_consistent_decisions(self) -> None:
        """Both entry points must produce the same admission decisions per entry."""
        cli = _run_cli(["--input", str(VALID_DIR)])
        script = _run_script(["--input", str(VALID_DIR)])

        def _parse_decisions(output: str) -> dict[str, str]:
            """Parse '  status  entry_id: reason' lines into {entry_id: status}."""
            result = {}
            for line in output.splitlines():
                stripped = line.strip()
                # Match status lines (start with a status word, contain entry_id)
                # Skip WARNING/ERROR lines from issues
                if stripped.startswith(("default", "blocked", "experimental", "excluded", "invalid", "pending")):
                    parts = stripped.split(None, 2)
                    if len(parts) >= 2:
                        entry_id = parts[1]
                        if "-" in entry_id:
                            result[entry_id] = parts[0]
            return result

        cli_decisions = _parse_decisions(cli.stdout)
        script_decisions = _parse_decisions(script.stdout)

        assert cli_decisions == script_decisions, (
            f"CLI and script gave different entry-level decisions\n"
            f"CLI: {cli_decisions}\nScript: {script_decisions}"
        )
        assert len(cli_decisions) == 18, f"Expected 18 entries, got {len(cli_decisions)}"


# ---------------------------------------------------------------------------
# CI/CD gate simulation tests
# ---------------------------------------------------------------------------

class TestCiCdGate:
    """Simulate the exact CI/CD gate logic to prove acceptance criteria."""

    def test_ci_would_fail_on_invalid_entry(self) -> None:
        """Emulate the CI 'Validate trend rejection fixtures' step."""
        result = _run_script(["--input", str(INVALID_DIR)])
        assert result.returncode != 0, (
            "CI gate would pass with invalid entries — WRONG: "
            "gate must fail when invalid entries are present"
        )

    def test_ci_would_pass_on_valid_entries(self) -> None:
        """Emulate the CI 'Validate trend admission fixtures' step."""
        result = _run_script(["--input", str(VALID_DIR)])
        assert result.returncode == 0, (
            f"CI gate would fail with valid entries — WRONG: "
            f"gate must pass valid entries (exit {result.returncode})"
        )

    def test_hf_publish_gate_rejects_invalid_snapshot(self) -> None:
        """Emulate the HF publish gate's post-sync snapshot validation."""
        # Build a synthetic "snapshot" with one valid + one invalid entry
        valid_data = json.loads((VALID_DIR / "experimental.json").read_text(encoding="utf-8"))
        invalid_data = json.loads((INVALID_DIR / "bad-version.json").read_text(encoding="utf-8"))
        payload = [valid_data, invalid_data]

        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir) / "leaderboard_single.json"
            snapshot.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            result = _run_cli(["--input", str(snapshot)])
        assert result.returncode != 0, (
            "HF publish gate would allow invalid entry through — WRONG"
        )

    def test_hf_publish_gate_accepts_clean_snapshot(self) -> None:
        """Emulate the HF publish gate with all-valid snapshot entries."""
        payload: list[dict] = []
        for path in sorted(VALID_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                payload.extend(data)
            else:
                payload.append(data)

        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir) / "leaderboard_single.json"
            snapshot.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            result = _run_cli(["--input", str(snapshot)])
        assert result.returncode == 0, (
            f"HF publish gate rejected clean snapshot (exit {result.returncode})"
        )
