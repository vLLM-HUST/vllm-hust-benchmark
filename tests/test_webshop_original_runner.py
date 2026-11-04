"""Verify the local adapter preserves author reward and action boundaries."""

from pathlib import Path
import runpy


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
