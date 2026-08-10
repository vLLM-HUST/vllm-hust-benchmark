"""Shell script syntax + help tests for scripts/run_readiness_slo_matrix.sh.

Per the project memory constraint: ``CI lint job must include
`bash -n scripts/*.sh` to check shell script syntax`` and
``TestShellScriptSyntax unit test must be added to verify shell script
syntax``. The readiness SLO matrix runner is a new shell entry point and
must be covered by the same guard.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests._bash_utils import bash_executable

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_readiness_slo_matrix.sh"


def test_readiness_slo_matrix_script_has_valid_syntax() -> None:
    """`bash -n` must accept the matrix runner with no syntax errors."""
    result = subprocess.run(
        [bash_executable(), "-n", str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_readiness_slo_matrix_script_help_works() -> None:
    """`--help` must exit 0 and document the matrix options."""
    result = subprocess.run(
        [bash_executable(), str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout
    assert "--workloads" in result.stdout
    assert "--load-profiles" in result.stdout
    assert "--repetitions" in result.stdout
    assert "--engine-repo" in result.stdout
    assert "--output-dir" in result.stdout


def test_readiness_slo_matrix_script_rejects_missing_engine_repo() -> None:
    """Without --engine-repo (or VLLM_HUST_REPO) the script must fail closed."""
    env = {
        "PATH": "/usr/bin:/bin",
        "VLLM_HUST_REPO": "",
    }
    result = subprocess.run(
        [bash_executable(), str(SCRIPT), "--repetitions", "3"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode != 0
    assert "--engine-repo is required" in result.stderr


def test_readiness_slo_matrix_script_rejects_low_repetitions() -> None:
    """--repetitions must be >= 3 (independent process restarts)."""
    result = subprocess.run(
        [bash_executable(), str(SCRIPT), "--repetitions", "2", "--engine-repo", "/tmp"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "repetitions must be an integer >= 3" in result.stderr
