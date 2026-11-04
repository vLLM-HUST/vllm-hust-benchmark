"""Score saved fixed original LongVideoBench responses with installed lmms-eval."""

import argparse
import ast
import hashlib
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

UPSTREAM_SHA256 = "b5fad766cab8a0a0a5a2305a61882c158c7c2ebe509d7c7441dbbc2644f3d454"


def pinned_scorer(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Unexpected upstream scorer digest")
    names = {
        "parse_multi_choice_response",
        "evaluate_longvideobench",
        "eval_multi_choice",
        "calculate_ins_level_acc",
        "longvideobench_process_results",
        "longvideobench_aggregate_results",
    }
    tree = ast.parse(raw)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise ValueError("Incomplete upstream scoring functions")
    scope = {
        "random": random,
        "re": re,
        "defaultdict": defaultdict,
        "eval_logger": logging.getLogger("longvideobench"),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    return scope["longvideobench_process_results"], scope[
        "longvideobench_aggregate_results"
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--upstream-utils",
        type=Path,
        help="Exact pinned utils.py for dependency-free scoring",
    )
    args = parser.parse_args()
    if args.upstream_utils:
        process, aggregate = pinned_scorer(args.upstream_utils)
    else:
        from lmms_eval.tasks.longvideobench.utils import (
            longvideobench_aggregate_results as aggregate,
            longvideobench_process_results as process,
        )

    rows = json.loads(args.validation.read_text(encoding="utf-8"))
    if len(rows) != 1337:
        raise ValueError("Expected the pinned original validation split")
    random.seed(args.seed)
    results = []
    for index in range(3):
        original = rows[index]
        saved = json.loads(
            (args.responses / f"{index}.json").read_text(encoding="utf-8")
        )
        if saved["id"] != original["id"] or saved["reference"] != chr(
            65 + original["correct_choice"]
        ):
            raise ValueError(f"Saved result does not match original position {index}")
        response = saved["response"]["choices"][0]["message"]["content"]
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"No final answer at original position {index}")
        candidates = original["candidates"]
        if not 2 <= len(candidates) <= 5 or not 0 <= original["correct_choice"] < len(
            candidates
        ):
            raise ValueError(f"Invalid original candidates at position {index}")
        # The publisher JSON has a candidates list; lmms-eval expects option0..4.
        scorer_doc = {
            **original,
            **{
                f"option{i}": candidates[i] if i < len(candidates) else "N/A"
                for i in range(5)
            },
        }
        results.append(process(scorer_doc, [response])["lvb_acc"])
    print(
        json.dumps(
            {
                "fixed_positions": [0, 1, 2],
                "seed": args.seed,
                "lvb_acc": aggregate(results),
            }
        )
    )


if __name__ == "__main__":
    main()
