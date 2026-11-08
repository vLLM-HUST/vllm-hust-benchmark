"""Bind the pinned WorkBench runner to an actual private vLLM model."""

import argparse
import ast
import json
import os
from pathlib import Path
import sys

import pandas as pd

from appworld_model_transport import model_proxy


POSITIONS = [0, 10, 20]
MODEL = "Qwen/Qwen3.5-9B"
MODEL_ALIAS = "qwen3.5-9b-local"


def plan(author, cohort):
    path = (
        author / "data/processed/tasks_and_outcomes/v1/calendar_tasks_and_outcomes.csv"
    )
    table = pd.read_csv(path)
    selected = table.drop_duplicates("base_template", keep="first").head(3)
    if len(table) != 110 or list(selected.index) != POSITIONS:
        raise ValueError("pinned original calendar sample differs from control")
    with cohort.open("x") as stream:
        selected[["task", "domains"]].to_csv(stream, index=False)
    return selected


def execute(author, cohort, output, origin):
    data = pd.read_csv(cohort)
    if list(data.columns) != ["task", "domains"] or len(data) != 3:
        raise ValueError("expected a fixed three-row task-only cohort")
    if (author / "data/processed/tasks_and_outcomes").exists():
        raise ValueError("author outcome files must stay outside the agent input tree")
    os.chdir(author)
    sys.path.insert(0, str(author))
    from src.evals.agent import (
        MODEL_REGISTRY,
        _PROVIDER_API_KEY_ENV,
        _PROVIDER_BASE_URLS,
        ModelConfig,
        resolve_route,
    )
    from src.evals.inference import generate_results

    MODEL_REGISTRY[MODEL_ALIAS] = ModelConfig("local/" + MODEL, True, "local")
    _PROVIDER_API_KEY_ENV["local"] = "WORKBENCH_PRIVATE_LOOPBACK"
    _PROVIDER_BASE_URLS["local"] = origin
    os.environ["WORKBENCH_PRIVATE_LOOPBACK"] = "loopback"
    route = resolve_route(MODEL_ALIAS)
    if (route.model_id, route.base_url, route.provider) != (MODEL, origin, "local"):
        raise ValueError("author route does not match the verified loopback model")
    predictions = generate_results(str(cohort), MODEL_ALIAS, workers=1, log_traces=True)
    if len(predictions) != 3 or predictions["task"].tolist() != data["task"].tolist():
        raise ValueError("author inference did not retain the three original task rows")
    with output.open("x") as stream:
        predictions.to_csv(stream, index=False)


def score(author, predictions, output):
    os.chdir(author)
    sys.path.insert(0, str(author))
    from src.evals.metrics import compute_metrics

    baseline = pd.read_csv(
        "data/processed/tasks_and_outcomes/v1/calendar_tasks_and_outcomes.csv"
    )
    truth = baseline.iloc[POSITIONS].copy()
    truth["outcome"] = truth["outcome"].map(ast.literal_eval)
    actual = pd.read_csv(predictions)
    if actual["task"].tolist() != truth["task"].tolist():
        raise ValueError("predictions differ from the preselected task order")
    actual["function_calls"] = actual["function_calls"].map(ast.literal_eval)
    result = compute_metrics(truth, actual)
    report = {
        "original_calendar_positions": POSITIONS,
        "errors": [bool(error) for error in result["error"]],
        "original_state_correct": [bool(value) for value in result["correct"]],
        "original_unwanted_side_effects": [
            bool(value) for value in result["unwanted_side_effects"]
        ],
        "source_checkbox_eligible": False,
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "execute", "score"))
    parser.add_argument("author", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--queue", type=Path)
    args = parser.parse_args()
    if args.mode == "plan":
        plan(args.author.resolve(strict=True), args.input.resolve())
    elif args.output is None:
        parser.error("a new output is required")
    elif args.mode == "execute":
        if args.queue is None:
            parser.error("an isolated model queue is required")
        with model_proxy(args.queue, port=0, timeout=900, expected_model=MODEL) as port:
            execute(
                args.author.resolve(strict=True),
                args.input.resolve(strict=True),
                args.output.resolve(),
                f"http://127.0.0.1:{port}/v1",
            )
    else:
        score(
            args.author.resolve(strict=True),
            args.input.resolve(strict=True),
            args.output.resolve(),
        )
