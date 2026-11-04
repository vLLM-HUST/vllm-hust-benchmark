"""Score saved outputs for the three fixed API-Bank dialogues using upstream code."""

import argparse
import json
import os
from pathlib import Path
import sys


FILES = (
    "AddAgenda-AddAlarm-GetUserToken-level-2-1.jsonl",
    "AddAgenda-AddMeeting-GetUserToken-level-2-2.jsonl",
    "AddAgenda-AddMeeting-GetUserToken-level-2-3.jsonl",
)
LOCAL_TOOLS = {"GetUserToken", "AddAgenda", "AddAlarm", "AddMeeting"}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    root = args.author_root.resolve(strict=True)
    predictions = {}
    try:
        rows = read_jsonl(args.predictions)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read prediction JSONL: {exc}")
    for line_number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            parser.error(f"line {line_number}: prediction must be an object")
        key = (row.get("file"), row.get("id"))
        if (
            key[0] not in FILES
            or type(key[1]) is not int
            or key[1] < 0
            or key in predictions
            or not isinstance(row.get("output"), str)
        ):
            parser.error(f"line {line_number}: invalid or duplicate prediction")
        predictions[key] = row["output"]

    os.chdir(root)
    sys.path.insert(0, str(root))
    from api_call_extraction import get_api_call, parse_api_call
    from evaluator_by_json import Evaluator, Sample

    cases = {}
    for name in FILES:
        rows = read_jsonl(root / "lv1-lv2-samples/level-1-given-desc" / name)
        if any(
            row.get("role") == "API" and row.get("api_name") not in LOCAL_TOOLS
            for row in rows
        ):
            parser.error("unexpected tool in fixed dialogue")
        samples = Sample.from_chat_history(rows)
        evaluator = Evaluator(samples)
        for i, sample in enumerate(samples):
            if sample.ground_truth["role"] in {"API", "AI"}:
                cases[name, i] = (evaluator, sample.ground_truth["role"])
    if predictions.keys() != cases.keys():
        parser.error("predictions must match every fixed scoring position exactly")
    results = []
    for (name, i), (evaluator, role) in cases.items():
        output = predictions[name, i]
        if role == "API":
            call = get_api_call(output)
            score = False
            if call and parse_api_call(call)[0] in LOCAL_TOOLS:
                score = bool(evaluator.evaluate(i, call)[0])
        else:
            score = evaluator.evaluate(i, output)
        results.append({"file": name, "id": i, "role": role, "score": score})
    print(json.dumps(results))


if __name__ == "__main__":
    main()
