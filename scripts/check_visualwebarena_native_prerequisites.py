"""Verify pinned VisualWebArena prerequisites, without tasks, models or installs."""

import argparse
import json
from pathlib import Path
import platform
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_crosscodeeval_original_assets import verify_git_blob
from prepare_xsum_original_assets import bound_path


PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/visualwebarena-author-source.json"
)
TASK_FILES = (
    "config_files/vwa/test_classifieds.raw.json",
    "config_files/vwa/test_reddit.raw.json",
    "config_files/vwa/test_shopping.raw.json",
)


def inspect_original(author_root, *, pins_path=PINS):
    pins = json.loads(Path(pins_path).read_text("utf-8"))
    if pins.get("source_id") != "VISUALWEBARENA" or not pins.get("revision"):
        raise ValueError("pinned original VisualWebArena profile required")
    files = pins["files"]
    if len({row["path"] for row in files}) != len(files):
        raise ValueError("distinct original source paths required")
    root = Path(author_root).resolve(strict=True)
    for row in files:
        verify_git_blob(bound_path(root, "", row), row)
    required = {*TASK_FILES, "run.py", "environment_docker/README.md"}
    if not required.issubset({row["path"] for row in files}):
        raise ValueError("original tasks, launcher and site deployment pins required")
    launcher = (root / "run.py").read_text("utf-8")
    if (
        "if DATASET == 'visualwebarena':" not in launcher
        or "eval_caption_image_fn = image_utils.get_captioning_fn(" not in launcher
        or 'choices=["cpu", "cuda"]' not in launcher
    ):
        raise ValueError("review changed original eager captioner/device contract")
    groups = {}
    for name in TASK_FILES:
        tasks = json.loads((root / name).read_text("utf-8"))
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("nonempty original task lists required")
        ids = []
        for task in tasks:
            if (
                not isinstance(task, dict)
                or type(task.get("task_id")) is not int
                or not isinstance(task.get("eval"), dict)
                or not isinstance(task.get("sites"), list)
            ):
                raise ValueError("original task/site/evaluator objects required")
            ids.append(task["task_id"])
        if len(set(ids)) != len(ids):
            raise ValueError("distinct original task IDs within each split required")
        groups[name] = tasks
    fixed = groups[TASK_FILES[0]][:3]
    if [row["task_id"] for row in fixed] != [0, 1, 2]:
        raise ValueError("fixed original first three Classifieds tasks required")
    return {
        "source_id": "VISUALWEBARENA",
        "status": "ORIGINAL_PREREQUISITES_INSPECTED_NOT_NATIVE_READY",
        "publisher_revision": pins["revision"],
        "verified_original_files": len(files),
        "host_architecture": platform.machine(),
        "original_task_definitions": sum(len(rows) for rows in groups.values()),
        "original_split_counts": {name: len(rows) for name, rows in groups.items()},
        "fixed_original_tasks": [
            {
                "task_id": row["task_id"],
                "sites": row["sites"],
                "eval_types": row["eval"].get("eval_types", []),
                "has_original_goal_image": bool(row.get("image")),
                "require_reset": row.get("require_reset"),
            }
            for row in fixed
        ],
        "original_launcher_initializes_eval_captioner_even_for_url_match": True,
        "original_eval_captioner": "Salesforce/blip2-flan-t5-xl",
        "original_cli_eval_captioner_devices": ["cpu", "cuda"],
        "npu_captioner_execution_qualified": False,
        "url_only_metric_does_not_prove_captioner_free_original_launcher": True,
        "site_runtime_and_login_qualified": False,
        "actor_tokenizer_and_vision_protocol_qualified": False,
        "model_calls": 0,
        "original_task_or_scoring_executed": False,
        "source_checkbox_eligible": False,
        "remaining": [
            "Provision the original Classifieds site, complete author initialization and native login",
            "Qualify NPU-only captioner and actor compatibility without changing original tasks or metrics",
            "Preserve selected native Agent workflow and explicit budgets; obtain actual original scores",
            "Verify installed public capability and unchanged research-side original input/reference relations",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path)
    args = parser.parse_args()
    if args.author_root is None:
        result = {
            "source_id": "VISUALWEBARENA",
            "status": "PLAN_ONLY",
            "source_profile": str(PINS),
            "model_calls": 0,
            "source_checkbox_eligible": False,
        }
    else:
        result = inspect_original(args.author_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
