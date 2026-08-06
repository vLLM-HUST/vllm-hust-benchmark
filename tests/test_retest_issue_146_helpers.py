"""Tests for bash helper functions in retest_issue_146_regression.sh.

Per reviewer round 1 (PR #153): '请统一做 trim、大小写归一化后的完整
sentinel 校验并补反例测试'.  These tests source the two helper functions
(is_placeholder_sentinel, resolve_ascend_version) directly from the script
and verify their behavior with positive and reverse (negative) cases.
"""

from __future__ import annotations

import shlex
import subprocess
import textwrap
from pathlib import Path

import pytest
from tests._bash_utils import bash_executable

REPO_ROOT = Path(__file__).resolve().parents[1]
RETEST_SCRIPT = REPO_ROOT / "scripts" / "retest_issue_146_regression.sh"


def _source_helpers(snippet: str) -> str:
    """Extract just the helper functions from the retest script and run snippet.

    Uses awk to extract the function definitions without triggering the
    script's top-level side effects (set -euo pipefail, exports, mkdir, etc.).
    """
    script_path = shlex.quote(str(RETEST_SCRIPT))
    return textwrap.dedent(
        f"""\
        source <(awk '
          /^is_placeholder_sentinel\\(\\) \\{{/ {{capture=1}}
          /^resolve_ascend_version\\(\\) \\{{/ {{capture=1}}
          capture {{print}}
          /^\\}}/ && capture {{capture=0}}
        ' {script_path})
        {snippet}
        """
    )


def _run_bash(snippet: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [bash_executable(), "-c", _source_helpers(snippet)],
        capture_output=True,
        text=True,
        timeout=10,
    )


# ---------------------------------------------------------------------------
# is_placeholder_sentinel
# ---------------------------------------------------------------------------


class TestIsPlaceholderSentinel:
    """Verify is_placeholder_sentinel catches all sentinel variants.

    Per reviewer round 1: '实现对 CANN/driver 只判断空值和精确小写 unknown，
    对 Python 甚至只判断空值；请统一做 trim、大小写归一化后的完整 sentinel
    校验并补反例测试'.
    """

    @pytest.mark.parametrize(
        "value",
        [
            "unknown",
            "Unknown",
            "UNKNOWN",
            "not available",
            "NOT AVAILABLE",
            "Not Available",
            "n/a",
            "N/A",
            "N/a",
            "none",
            "None",
            "NONE",
            "null",
            "NULL",
            "Null",
            "",
            "   ",
            "  unknown  ",
            "\tunknown\t",
            "\n",
        ],
        ids=lambda v: f"sentinel_{repr(v)}",
    )
    def test_sentinels_detected(self, value: str):
        """All placeholder sentinels must return 0 (true = is sentinel)."""
        # Use printf to preserve whitespace in the value
        result = _run_bash(
            f'if is_placeholder_sentinel "$(printf %s {shlex.quote(value)})"; '
            f"then echo SENTINEL; else echo VALID; fi"
        )
        assert "SENTINEL" in result.stdout, (
            f"Expected {value!r} to be detected as sentinel, "
            f"got: {result.stdout.strip()}"
        )

    @pytest.mark.parametrize(
        "value",
        [
            "9.0.0",
            "26.0.rc1",
            "3.11.6",
            "Python 3.11.6",
            "8.5.RC1",
            "v1.0.0",
            "24.0.0",
        ],
        ids=lambda v: f"valid_{v}",
    )
    def test_real_values_pass(self, value: str):
        """Real version strings must return 1 (false = not a sentinel)."""
        result = _run_bash(
            f'if is_placeholder_sentinel "$(printf %s {shlex.quote(value)})"; '
            f"then echo SENTINEL; else echo VALID; fi"
        )
        assert "VALID" in result.stdout, (
            f"Expected {value!r} to be a valid version, got: {result.stdout.strip()}"
        )


# ---------------------------------------------------------------------------
# resolve_ascend_version
# ---------------------------------------------------------------------------


class TestResolveAscendVersion:
    """Verify resolve_ascend_version parses 'Version=' lines from candidate files.

    Per reviewer round 1: '/usr/local/Ascend/ascend-toolkit/latest/version.cfg
    已确认该路径在当前环境不存在，实际应从 install.info 解析'.
    """

    def test_parses_version_from_opp_version_info(self, tmp_path: Path):
        """The real CANN path (opp/version.info) has 'Version=9.0.0'."""
        version_file = tmp_path / "opp" / "version.info"
        version_file.parent.mkdir(parents=True)
        version_file.write_text(
            "Version=9.0.0\n"
            "version_dir=cann\n"
            'required_package_runtime_version=">=8.5"\n'
        )
        result = _run_bash(f'resolve_ascend_version "{version_file}" && echo OK')
        assert "9.0.0" in result.stdout
        assert result.returncode == 0

    def test_parses_driver_version_info(self, tmp_path: Path):
        """The driver path (driver/version.info) has 'Version=26.0.rc1'."""
        version_file = tmp_path / "driver" / "version.info"
        version_file.parent.mkdir(parents=True)
        version_file.write_text(
            "Version=26.0.rc1\nascendhal_version=7.35.23\naicpu_version=1.0\n"
        )
        result = _run_bash(f'resolve_ascend_version "{version_file}" && echo OK')
        assert "26.0.rc1" in result.stdout
        assert result.returncode == 0

    def test_parses_quoted_version(self, tmp_path: Path):
        """Version values wrapped in quotes must be extracted correctly."""
        version_file = tmp_path / "version.info"
        version_file.write_text('Version="8.5.RC1.2"\n')
        result = _run_bash(f'resolve_ascend_version "{version_file}"')
        assert result.stdout.strip() == "8.5.RC1.2"

    def test_case_insensitive_version_key(self, tmp_path: Path):
        """'version=' (lowercase) must also be parsed."""
        version_file = tmp_path / "version.info"
        version_file.write_text("version=7.0.0\n")
        result = _run_bash(f'resolve_ascend_version "{version_file}"')
        assert result.stdout.strip() == "7.0.0"

    def test_falls_back_to_second_candidate(self, tmp_path: Path):
        """If the first candidate doesn't exist, try the next one."""
        missing = tmp_path / "nonexistent" / "version.cfg"
        real = tmp_path / "opp" / "version.info"
        real.parent.mkdir(parents=True)
        real.write_text("Version=9.0.0\n")
        candidates = f"{missing} {real}"
        result = _run_bash(f'resolve_ascend_version "{candidates}"')
        assert result.stdout.strip() == "9.0.0"

    def test_returns_empty_when_no_candidate_exists(self, tmp_path: Path):
        """When no candidate file exists, return empty (fail-closed)."""
        result = _run_bash(
            f'resolve_ascend_version "{tmp_path}/nope1 {tmp_path}/nope2" || echo FAILED'
        )
        assert result.stdout.strip() == "FAILED"

    def test_returns_empty_when_no_version_line(self, tmp_path: Path):
        """When the file exists but has no 'Version=' line, return empty.

        E.g. driver/build.info contains only a timestamp like '2026-04-28 00:00:00'.
        """
        version_file = tmp_path / "build.info"
        version_file.write_text("2026-04-28 00:00:00\n")
        result = _run_bash(f'resolve_ascend_version "{version_file}" || echo FAILED')
        assert result.stdout.strip() == "FAILED"

    def test_skips_sentinel_version_value(self, tmp_path: Path):
        """If the Version= line contains a sentinel, it's still returned;
        is_placeholder_sentinel is responsible for rejecting it afterwards."""
        version_file = tmp_path / "version.info"
        version_file.write_text("Version=unknown\n")
        result = _run_bash(f'resolve_ascend_version "{version_file}"')
        # resolve_ascend_version returns the raw value; the caller
        # (write_env_manifest) is responsible for calling
        # is_placeholder_sentinel to reject it.
        assert result.stdout.strip() == "unknown"

    def test_version_cfg_still_works_if_present(self, tmp_path: Path):
        """If version.cfg exists and has Version= line, it's used as fallback.

        This ensures backward compatibility on environments that DO have
        version.cfg (older Ascend installs).
        """
        version_file = tmp_path / "version.cfg"
        version_file.write_text("Version=8.0.0\n")
        result = _run_bash(f'resolve_ascend_version "{version_file}"')
        assert result.stdout.strip() == "8.0.0"
