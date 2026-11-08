"""Run the unchanged original evaluator on saved predictions in an isolated CPU container.

This does not execute generated programs, call models, overwrite original scores,
or certify source delivery. Default prints a verified plan; --execute scores into
a fresh directory, separately from earlier Agent submissions.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess

from vllm_hust_workload_data.naturebench import original_cpu_packages


def plan(args):
    data = args.data_root.resolve(strict=True)
    predictions = args.predictions.resolve(strict=True)
    output = args.output.resolve()
    if not predictions.is_dir():
        raise ValueError("saved prediction directory required")
    if output.exists():
        raise FileExistsError("fresh scoring output required; retain earlier scores")
    for root in (data, predictions):
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError(
                "scoring output must be separate from data and predictions"
            )
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image):
        raise ValueError("explicit qualified immutable CPU image ID required")
    packages = list(original_cpu_packages(data, task_ids=[args.task_id]))
    task = data / "tasks" / args.task_id
    evaluator = task / "evaluation/evaluator.py"
    if not evaluator.is_file():
        raise ValueError("selected original task has no evaluator.py entry")
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--cpus",
        "4",
        "--memory",
        "8g",
        "--pids-limit",
        "256",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=512m",
        "--mount",
        f"type=bind,src={task / 'evaluation'},dst=/evaluation,readonly",
        "--mount",
        f"type=bind,src={predictions},dst=/predictions,readonly",
        "--mount",
        f"type=bind,src={output},dst=/grade",
        "--env",
        "OUTPUT_DIR=/predictions",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--workdir",
        "/grade",
        args.image,
        "python",
        "/evaluation/evaluator.py",
    ]
    return {
        "source_id": "NATUREBENCH",
        "task_id": args.task_id,
        "source": packages[0]["source"],
        "command": command,
        "status": "VERIFIED_SAVED_PREDICTION_SCORING_PLAN_NOT_EXECUTED",
        "source_checkbox_eligible": False,
        "new_model_calls": 0,
        "generated_program_execution": False,
        "boundary": "Separate final saved-prediction score, not replacement of earlier submissions",
    }


def execute(args, prepared):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    receipt = dict(prepared)
    try:
        with (
            (output / "stdout.private.log").open("xb") as stdout,
            (output / "stderr.private.log").open("xb") as stderr,
        ):
            result = subprocess.run(prepared["command"], stdout=stdout, stderr=stderr)
        receipt["exit_code"] = result.returncode
        receipt["status"] = "ORIGINAL_SAVED_PREDICTION_EVALUATOR_TERMINAL"
        if result.returncode:
            return result.returncode
        score = json.loads((output / "score.json").read_text("utf-8"))
        if not isinstance(score, dict) or not score:
            raise ValueError("nonempty original score object required")
        receipt["original_scores"] = score
        # Recheck registered original assets after execution.
        list(original_cpu_packages(args.data_root, task_ids=[args.task_id]))
        return 0
    finally:
        (output / "scoring-receipt.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    prepared = plan(args)
    if args.execute:
        raise SystemExit(execute(args, prepared))
    print(json.dumps(prepared, indent=2))


if __name__ == "__main__":
    main()
