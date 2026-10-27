"""Replay saved author ScienceWorld actions on CPU to verify scores; no model API."""

import argparse
import ast
import json
from pathlib import Path
import re

FIXED = {3: [27, 28, 29], 4: [27, 28, 29], 5: [225, 226, 227]}


def author_score(history):
    scores = [int(round(100 * float(step["score"]))) for step in history]
    if not scores:
        raise ValueError("missing original environment history")
    # The pinned author's --no_stop profile rolls a negative terminal score back.
    # The reset observation may already have nonzero environment credit, but
    # the author initializes score/last_score to0 before the first agent action.
    if len(scores) == 1:
        return 0
    return scores[-1] if scores[-1] >= 0 else (scores[-2] if len(scores) > 2 else 0)


def parse_scores(text):
    match = re.search(r"Scores:\s*(\[.*?\]) Average score:\s*([^ ]+) Args:", text)
    if not match:
        raise ValueError("missing author score summary")
    scores = ast.literal_eval(match.group(1))
    if len(scores) != 3 or not all(isinstance(value, (int, float)) for value in scores):
        raise ValueError("exactly three original episode scores required")
    if abs(float(match.group(2)) - sum(scores) / 3) > 1e-8:
        raise ValueError("author average score inconsistent")
    return scores


def audit_saved_scores(root):
    """Validate saved native coverage/scores, without claiming environment replay."""
    root = Path(root)
    native = json.loads((root / "native-public.json").read_text())
    if (
        native["status"]
        != "AUTHOR_SELECTED_TASK_LOOPS_RETURNED_SAVED_SCORING_AUDIT_PENDING"
        or native["task_indices"] != list(FIXED)
        or native["completed_task_indices"] != list(FIXED)
    ):
        raise ValueError("fixed original native loops not complete")
    episodes = []
    for task, variations in FIXED.items():
        folder = root / f"task-{task}"
        files = list(folder.glob(f"task{task}-*.json"))
        if len(files) != 1:
            raise ValueError("one original history per task required")
        packed = json.loads(files[0].read_text())
        if set(packed) != {str(v) for v in variations}:
            raise ValueError("fixed original coverage differs")
        text = (folder / f"task{task}-score.txt").read_text()
        scores = parse_scores(text)
        config = ast.literal_eval(text.split(" Args:", 1)[1].strip())
        if any(
            config[key] != value
            for key, value in {
                "env_step_limit": 100,
                "simplification_str": "easy",
                "set": "test_mini",
                "no_stop": True,
                "model_name": "gpt-4",
            }.items()
        ):
            raise ValueError("saved author protocol differs")
        for offset, variation in enumerate(variations):
            entry = packed[str(variation)]
            saved = entry["history"]
            if (
                entry["episodeIdx"] != variation
                or int(saved["taskIdx"]) != task
                or int(saved["variationIdx"]) != variation
                or saved["simplifications"] != "easy"
            ):
                raise ValueError("saved native identity/profile differs")
            if not saved["history"] or saved["history"][0]["action"] != "look around":
                raise ValueError("saved original reset absent")
            score = author_score(saved["history"])
            if score != scores[offset]:
                raise ValueError("saved author score differs from environment history")
            episodes.append(
                {
                    "task_index": task,
                    "variation": variation,
                    "author_score": score,
                    "raw_terminal_environment_score": saved["history"][-1]["score"],
                    "saved_environment_steps": len(saved["history"]),
                }
            )
    return {
        "source_id": "SCIENCEWORLD",
        "status": "NINE_SAVED_NATIVE_EPISODES_AND_AUTHOR_SCORES_VERIFIED_NOT_ENVIRONMENT_REPLAY",
        "source_checkbox_eligible": False,
        "model_calls": 0,
        "environment_replay_performed": False,
        "original_model_call_attempts": native["model_calls"],
        "episodes": episodes,
        "scope": "saved identity/protocol/scoring consistency only; prior strict replay mismatch remains unresolved and is not waived",
    }


def audit(root):
    from scienceworld import ScienceWorldEnv, __version__

    if __version__ != "1.1.3":
        raise ValueError("original author environment1.1.3 required")
    root = Path(root)
    native = json.loads((root / "native-public.json").read_text())
    if (
        native["status"]
        != "AUTHOR_SELECTED_TASK_LOOPS_RETURNED_SAVED_SCORING_AUDIT_PENDING"
    ):
        raise ValueError("original native loop not complete")
    if native["task_indices"] != list(FIXED) or native[
        "completed_task_indices"
    ] != list(FIXED):
        raise ValueError("fixed original coverage incomplete")
    results = []
    env = ScienceWorldEnv(envStepLimit=100)
    names = env.getTaskNames()
    for task, variations in FIXED.items():
        folder = root / f"task-{task}"
        history_files = list(folder.glob(f"task{task}-*.json"))
        if len(history_files) != 1:
            raise ValueError("one original buffered history file per task required")
        packed = json.loads(history_files[0].read_text())
        if set(packed) != {str(v) for v in variations}:
            raise ValueError("original variation set differs")
        expected_scores = parse_scores((folder / f"task{task}-score.txt").read_text())
        for index, variation in enumerate(variations):
            episode = packed[str(variation)]
            saved = episode["history"]
            if (
                episode["episodeIdx"] != variation
                or int(saved["taskIdx"]) != task
                or int(saved["variationIdx"]) != variation
                or saved["taskName"] != names[task]
            ):
                raise ValueError("original episode identity differs")
            actions = saved["history"]
            if not actions or actions[0]["action"] != "look around":
                raise ValueError("original reset history missing")
            env.load(names[task], variation, "easy", generateGoldPath=True)
            env.reset()
            if env.getRunHistory()["history"] != actions[:1]:
                raise ValueError("original reset replay differs")
            for offset, record in enumerate(actions[1:], start=1):
                env.step(record["action"])
                if env.getRunHistory()["history"][-1] != record:
                    raise ValueError(
                        f"original action/observation/score replay differs at {task}/{variation}/{offset}"
                    )
            replayed = env.getRunHistory()
            if replayed != saved:
                raise ValueError("full original environment history replay differs")
            score = author_score(actions)
            if score != expected_scores[index]:
                raise ValueError("native score differs from environment history")
            results.append(
                {
                    "task_index": task,
                    "variation": variation,
                    "environment_steps_replayed": len(actions),
                    "author_score": score,
                    "raw_terminal_environment_score": actions[-1]["score"],
                    "replay_exact": True,
                }
            )
    return {
        "source_id": "SCIENCEWORLD",
        "status": "NINE_ORIGINAL_EPISODES_CPU_REPLAY_AND_AUTHOR_SCORE_VERIFIED",
        "source_checkbox_eligible": False,
        "model_calls": 0,
        "episodes": results,
        "scope": "saved action replay and scoring only; public package/research consumption closure still required",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--saved-scores-only",
        action="store_true",
        help="offline saved-score consistency only; does not pass or replace strict environment replay",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("fresh audit output required")
    result = (
        audit_saved_scores(args.native_root)
        if args.saved_scores_only
        else audit(args.native_root)
    )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
