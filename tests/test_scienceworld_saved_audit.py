"""Saved score interpretation must retain the author's negative-score rule."""

import ast
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "scripts/verify_scienceworld_original_saved.py"
SPEC = importlib.util.spec_from_file_location("scienceworld_saved", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "scores,expected",
    [(["0", "0.5", "1"], 100), (["0", "0.5", "-1"], 50), (["0", "0"], 0)],
)
def test_author_no_stop_rule(scores, expected):
    assert MODULE.author_score([{"score": score} for score in scores]) == expected


def test_original_score_summary():
    assert MODULE.parse_scores(
        "Task name:testScores:[0, 50, 100] Average score:50.0 Args:{}"
    ) == [0, 50, 100]


def test_author_initial_credit_is_not_last_agent_score():
    assert MODULE.author_score([{"score": "0.08333333333333333"}, {"score": "-1"}]) == 0
    assert MODULE.author_score([{"score": "0.2"}]) == 0
    assert MODULE.parse_scores(
        "Task name:testScores: [0, 50, 100] Average score: 50.0 Args: {}"
    ) == [0, 50, 100]


@pytest.mark.parametrize(
    "text",
    [
        "missing",
        "Scores:[0] Average score:0 Args:{}",
        "Scores:[0,50,100] Average score:100 Args:{}",
    ],
)
def test_incomplete_or_changed_scores_fail(text):
    with pytest.raises(ValueError):
        MODULE.parse_scores(text)


def test_auditor_has_no_model_or_network_client():
    tree = ast.parse(PATH.read_text("utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module)
    assert set(modules) <= {"argparse", "ast", "json", "pathlib", "re", "scienceworld"}


def make_saved_fixture(root):
    import json

    (root / "native-public.json").write_text(
        json.dumps(
            {
                "status": "AUTHOR_SELECTED_TASK_LOOPS_RETURNED_SAVED_SCORING_AUDIT_PENDING",
                "task_indices": [3, 4, 5],
                "completed_task_indices": [3, 4, 5],
                "model_calls": 9,
            }
        )
    )
    for task, variations in MODULE.FIXED.items():
        folder = root / f"task-{task}"
        folder.mkdir()
        packed = {
            str(var): {
                "episodeIdx": var,
                "history": {
                    "taskIdx": str(task),
                    "variationIdx": str(var),
                    "simplifications": "easy",
                    "history": [
                        {"action": "look around", "score": "0"},
                        {"action": "fixture", "score": "0.5"},
                    ],
                },
            }
            for var in variations
        }
        (folder / f"task{task}-fixture.json").write_text(json.dumps(packed))
        config = {
            "env_step_limit": 100,
            "simplification_str": "easy",
            "set": "test_mini",
            "no_stop": True,
            "model_name": "gpt-4",
        }
        (folder / f"task{task}-score.txt").write_text(
            f"Scores: [50, 50, 50] Average score: 50.0 Args: {config!r}"
        )


def test_saved_score_audit_does_not_claim_environment_replay(tmp_path):
    make_saved_fixture(tmp_path)
    result = MODULE.audit_saved_scores(tmp_path)
    assert len(result["episodes"]) == 9
    assert result["model_calls"] == 0
    assert result["environment_replay_performed"] is False
    assert result["source_checkbox_eligible"] is False


def test_saved_score_audit_rejects_protocol_change(tmp_path):
    make_saved_fixture(tmp_path)
    path = tmp_path / "task-3/task3-score.txt"
    path.write_text(path.read_text().replace("'no_stop': True", "'no_stop': False"))
    with pytest.raises(ValueError, match="protocol"):
        MODULE.audit_saved_scores(tmp_path)
