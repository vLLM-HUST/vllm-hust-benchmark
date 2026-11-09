"""Run unchanged author scoring on complete saved offline answers, never infer.

Only the four tasks in the author's stats.sh are supported. Optional streaming
variants need their own qualified original scoring protocol, not a shape rewrite.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys

from prepare_jsonschemabench_author_source import verify_source
from vllm_hust_workload_data.assets import asset_path, verify_file
from vllm_hust_workload_data.registry import source_spec

PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/streamingbench-author-source.json"
)


def validate_answers(original, answers, *, model, task):
    if task not in {"real", "omni", "sqa", "proactive"} or not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", model
    ):
        raise ValueError(
            "official offline task and safe native model result key required"
        )
    if not isinstance(original, list) or not original or not isinstance(answers, list):
        raise ValueError("nonempty original and saved answer arrays required")
    clean = deepcopy(answers)
    count = 0
    for row in clean:
        groups = row if task == "sqa" else [row]
        if not isinstance(groups, list) or not groups:
            raise ValueError("complete original offline task shape required")
        for group in groups:
            if not isinstance(group, dict) or not isinstance(
                group.get("questions"), list
            ):
                raise ValueError("complete original question group required")
            for question in group["questions"]:
                if not isinstance(question, dict) or model not in question:
                    raise ValueError(
                        "missing saved original model result; no silent scorer skip"
                    )
                value = question.pop(model)
                if task == "proactive":
                    if (
                        not isinstance(value, dict)
                        or not isinstance(value.get("dialog_history"), list)
                        or not value["dialog_history"]
                    ):
                        raise ValueError(
                            "original proactive full dialog history required"
                        )
                    last = value["dialog_history"][-1]
                    if (
                        not isinstance(last, dict)
                        or type(last.get("time")) not in {int, float}
                        or not isinstance(last.get("content"), str)
                    ):
                        raise ValueError(
                            "original proactive final time/content required"
                        )
                elif not isinstance(value, str) or not value:
                    raise ValueError(
                        "empty/malformed result cannot be silently excluded by original scorer"
                    )
                count += 1
    if clean != original or count == 0:
        raise ValueError(
            "saved answers changed original questions, references or nested order"
        )
    return {
        "original_question_occurrences": count,
        "all_original_fields_preserved": True,
    }


def score(
    *, author_root, data_root, original_subset, answers_path, model, task, output
):
    author_root = verify_source(author_root, PROFILE)
    original_subset, answers_path = (
        Path(original_subset).resolve(),
        Path(answers_path).resolve(),
    )
    original = json.loads(original_subset.read_text("utf-8"))
    source = source_spec("STREAMINGBENCH")
    item = next(
        item
        for item in source["files"]
        if item["path"] == f"src/data/questions_{task}.json"
    )
    verify_file(data_root, item)
    published = json.loads(asset_path(data_root, item["path"]).read_text("utf-8"))
    if original != published and original != published[:3]:
        raise ValueError(
            "original subset must match full publisher file or fixed first3 complete groups"
        )
    coverage = validate_answers(
        original, json.loads(answers_path.read_text("utf-8")), model=model, task=task
    )
    output = Path(output).resolve()
    if (
        output.is_relative_to(PROFILE.parents[2])
        or output == author_root
        or output.is_relative_to(author_root)
        or author_root.is_relative_to(output)
        or any(
            path == output or path.is_relative_to(output)
            for path in (original_subset, answers_path)
        )
    ):
        raise ValueError(
            "fresh external output separate from originals and author source required"
        )
    output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-I",
        str(author_root / "src/data/count.py"),
        "--task",
        task,
        "--model",
        model,
        "--src",
        str(answers_path),
    ]
    result = subprocess.run(
        command, cwd=output, capture_output=True, text=True, timeout=60
    )
    (output / "scorer-stdout.log").write_text(result.stdout, encoding="utf-8")
    (output / "scorer-stderr.log").write_text(result.stderr, encoding="utf-8")
    receipt = {
        "source_id": "STREAMINGBENCH",
        "original_scorer_unchanged": True,
        "exit_code": result.returncode,
        "task": task,
        "model_result_key": model,
        "coverage": coverage,
        "model_calls": 0,
        "native_inference_executed_by_this_command": False,
        "source_checkbox_eligible": False,
    }
    if result.returncode == 0:
        receipt["original_stats_json"] = f"{model}_stats.json"
        receipt["original_stats_csv"] = f"{model}_stats.csv"
        if (
            not (output / receipt["original_stats_json"]).is_file()
            or not (output / receipt["original_stats_csv"]).is_file()
        ):
            raise ValueError("original scorer did not produce both declared outputs")
    (output / "scoring-public.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(
            "original scorer failed; preserved logs, no replacement grade"
        )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--original-subset", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--task", choices=["real", "omni", "sqa", "proactive"], required=True
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            score(
                author_root=args.author_root,
                data_root=args.data_root,
                original_subset=args.original_subset,
                answers_path=args.answers,
                model=args.model,
                task=args.task,
                output=args.output,
            )
        )
    )
