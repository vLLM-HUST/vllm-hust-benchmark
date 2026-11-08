"""Verify the local adapter preserves author reward and action boundaries."""

from pathlib import Path
import runpy
import subprocess
import sys

import pytest


def test_saved_author_budget_rejects_unpinned_output(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_webshop_original.py"
    result_file = tmp_path / "partial.jsonl"
    result_file.write_bytes(b'{"session":0}\n')
    result = subprocess.run(
        [sys.executable, str(script), "--verify-saved-author-budget", str(result_file)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2
    assert "does not match the pinned author-budget run" in result.stderr
    assert "SAVED_EPISODES_VERIFIED" not in result.stdout


def test_author_episode_scoring_distinguishes_expiry_from_partial_probe():
    score = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/run_webshop_original.py")
    )["author_episode_score"]
    assert score({"terminated": True, "reward": 0.4, "steps": [{}]}) == 0.4
    assert score({"terminated": False, "steps": [{}] * 100}) == 0.0
    with pytest.raises(ValueError, match="incomplete author episode"):
        score({"terminated": False, "steps": [{}] * 15})


def test_fixed_session_preserves_author_reward():
    run = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/run_webshop_original.py")
    )["run_session"]

    class Env:
        def reset(self, session):
            assert session == 2
            return "original observation", None

        def get_available_actions(self):
            return {"has_search_bar": False, "clickables": ["buy now"]}

        def step(self, action):
            assert action == "click[buy now]"
            return "terminal", 0.0, True, {}

    actions = iter([None, "search[shoes]", "click[unknown]", "click[buy now]"])
    result = run(Env(), 2, lambda *_: next(actions), 4)
    assert result["terminated"] and result["reward"] == 0.0
    assert all(step["invalid"] for step in result["steps"][:3])


def test_step_cap_does_not_invent_reward():
    run = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/run_webshop_original.py")
    )["run_session"]

    class Env:
        def reset(self, session):
            return "observation", None

        def get_available_actions(self):
            return {"has_search_bar": True, "clickables": []}

    assert run(Env(), 0, lambda *_: None, 1) == {
        "session": 0,
        "steps": [{"action": None, "invalid": True}],
        "terminated": False,
        "reward": None,
    }
