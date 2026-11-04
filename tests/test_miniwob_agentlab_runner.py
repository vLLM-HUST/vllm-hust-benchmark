"""Check the MiniWoB++ launcher without a browser or model dependency."""

import json
import runpy
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("terminated", "truncated", "reward", "error", "valid"),
    [
        (True, False, 0.0, None, True),
        (False, True, 0.0, None, True),
        (False, False, 1.0, None, False),
        (True, False, "1.0", None, False),
        (True, False, 1.0, "browser failed", False),
    ],
)
def test_terminal_summary(tmp_path, terminated, truncated, reward, error, valid):
    script = Path(__file__).resolve().parents[1] / "scripts/run_miniwob_agentlab.py"
    read_summary = runpy.run_path(str(script))["terminal_summary"]
    (tmp_path / "summary_info.json").write_text(
        json.dumps(
            {
                "n_steps": 1,
                "cum_reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "err_msg": error,
            }
        ),
        encoding="utf-8",
    )
    if not valid:
        with pytest.raises(ValueError, match="invalid BrowserGym terminal summary"):
            read_summary(tmp_path)
        return
    assert read_summary(tmp_path) == {
        "steps": 1,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }


def run_cli(tmp_path, endpoint):
    script = Path(__file__).resolve().parents[1] / "scripts/run_miniwob_agentlab.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--task",
            "miniwob.click-dialog",
            "--site",
            "http://127.0.0.1:8001/miniwob/",
            "--endpoint",
            endpoint,
            "--model",
            "fixture",
            "--output",
            str(tmp_path / "result"),
            "--max-steps",
            "5",
            "--max-output-tokens",
            "64",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_default_plan_creates_no_result_directory(tmp_path):
    result = run_cli(tmp_path, "http://127.0.0.1:8002/v1")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "PLAN_ONLY"
    assert not (tmp_path / "result").exists()


def test_non_loopback_endpoint_is_rejected(tmp_path):
    result = run_cli(tmp_path, "http://192.0.2.1:8002/v1")
    assert result.returncode == 2
    assert "loopback" in result.stderr
