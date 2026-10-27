"""Offline saved-result audit; no model, network or judge calls."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

from prepare_longbench_v2_original_capacity import ARCHIVE_SHA, DATA_SHA
from run_longbench_v2_original_samples import validate_capacity_selection


def audit_case(
    original, prompt, request, response, prediction, summary, extract, model
):
    if request != {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 128,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }:
        raise ValueError("saved request differs from full original prompt/profile")
    choice = response["choices"][0]
    content = choice["message"]["content"]
    if not isinstance(content, str) or not content:
        raise ValueError("missing original model output")
    answer = extract(content.strip())
    expected = dict(original)
    expected.update(
        response=content.strip(),
        pred=answer,
        judge=answer == original["answer"],
        context=original["context"][:1000],
    )
    if prediction != expected:
        raise ValueError("saved original scorer result or task fields differ")
    usage = response["usage"]
    if (
        summary["id"] != original["_id"]
        or hashlib.sha256(prompt.encode()).hexdigest() != summary["prompt_sha256"]
        or summary["full_prompt_tokens"] != usage["prompt_tokens"]
        or summary["completion_tokens"] != usage["completion_tokens"]
        or summary["finish_reason"] != choice["finish_reason"]
        or summary["full_input_verified"] is not True
        or not 0 <= usage["completion_tokens"] <= 128
    ):
        raise ValueError("saved response usage/identity differs")
    return bool(prediction["judge"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--capacity-selection", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "PLAN_ONLY", "model_calls": 0}))
        return
    for path, digest in [
        (args.data_file, DATA_SHA),
        (args.official_archive, ARCHIVE_SHA),
    ]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("original source identity mismatch")
    run = args.run.resolve()
    output = args.output.resolve()
    if output == run or run in output.parents or output in run.parents:
        raise ValueError("audit output must be disjoint from original run")
    if output.exists():
        raise ValueError("fresh audit output required")
    receipt = json.loads((run / "receipt.json").read_text("utf-8"))
    if receipt["status"] != "END_TO_END_EXECUTED_AND_SCORED":
        raise ValueError("terminal native result required")
    capacity = json.loads(args.capacity_selection.read_text("utf-8"))
    selected = validate_capacity_selection(
        capacity, receipt["model"], receipt["model_max_length"]
    )
    rows = json.loads(args.data_file.read_text("utf-8"))
    prefix = "THUDM-LongBench-2e00731/"
    with zipfile.ZipFile(args.official_archive) as archive:
        source = archive.read(prefix + "pred.py").decode()
        template = archive.read(prefix + "prompts/0shot.txt").decode()
        for name in ["pred.py", "prompts/0shot.txt"]:
            if (run / "official-source" / prefix / name).read_bytes() != archive.read(
                prefix + name
            ):
                raise ValueError("executed original prompt/scorer source changed")
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "extract_answer"
    )
    namespace = {"re": re, "__builtins__": {}}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "pinned_original_extract_answer",
            "exec",
        ),
        namespace,
    )
    predictions = [
        json.loads(line)
        for line in (run / "predictions.private.jsonl").read_text("utf-8").splitlines()
    ]
    if len(predictions) != len(selected) or len(receipt["requests"]) != len(selected):
        raise ValueError("incomplete original selection")
    correct = []
    for number, (item, prediction, summary) in enumerate(
        zip(selected, predictions, receipt["requests"], strict=True), 1
    ):
        original = rows[item["row_index"]]
        prompt = template
        for marker, field in [
            ("$DOC$", "context"),
            ("$Q$", "question"),
            ("$C_A$", "choice_A"),
            ("$C_B$", "choice_B"),
            ("$C_C$", "choice_C"),
            ("$C_D$", "choice_D"),
        ]:
            prompt = prompt.replace(marker, original[field].strip())
        if (
            item["id"] != original["_id"]
            or item["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest()
            or item["full_prompt_tokens"] != summary["full_prompt_tokens"]
        ):
            raise ValueError("frozen original capacity selection changed")
        request = json.loads(
            (run / f"request-{number}.private.json").read_text("utf-8")
        )
        response = json.loads(
            (run / f"response-{number}.private.json").read_text("utf-8")
        )
        correct.append(
            audit_case(
                original,
                prompt,
                request,
                response,
                prediction,
                summary,
                namespace["extract_answer"],
                receipt["model"],
            )
        )
    if (
        sum(correct) != receipt["correct_tasks"]
        or [item["correct"] for item in receipt["tasks"]] != correct
    ):
        raise ValueError("saved aggregate scoring differs")
    output.mkdir(mode=0o700, parents=True)
    result = {
        "source_id": "LONGBENCH_V2",
        "status": "ORIGINAL_FULL_PROMPTS_SAVED_RESPONSES_AND_NATIVE_SCORING_VERIFIED",
        "tasks": len(correct),
        "correct": sum(correct),
        "new_model_calls": 0,
        "original_run_modified": False,
        "source_checkbox_eligible": False,
    }
    (output / "receipt-public.json").write_text(json.dumps(result, indent=2), "utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
