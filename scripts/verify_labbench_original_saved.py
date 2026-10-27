"""Reconstruct original LAB-Bench tasks and rescore saved responses without API calls.

Run inside the pinned publisher environment. Inputs and generated answers remain
external; stdout contains counts and scores only. Partial audits never complete a
source. Failed/retried attempts are rejected for separate review, not discarded.
"""

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

from run_labbench_original import CATEGORIES, select_instances, verify_source


def compare(actual, expected, label):
    if actual != expected:
        raise ValueError(f"saved evidence mismatch: {label}")


async def audit_category(labbench, category, receipt, root):
    evaluator = labbench.Evaluator(labbench.Eval(category))
    selected = select_instances(evaluator.eval_set.instances, receipt["per_subset"])
    evaluator.eval_set.instances = selected
    compare(
        [str(item.id) for _, item in selected],
        category_record(receipt, category)["selected_ids"],
        "fixed original task ids",
    )
    saved = json.loads((root / f"{category}-results.private.json").read_text())
    buffer = json.loads((root / f"{category}-agent-buffer.private.json").read_text())
    attempts = sorted(root.glob(f"{category}-attempt-*.private.json"))
    compare(
        len(attempts),
        len(selected),
        "one retained response per original task; retries require review",
    )
    options = {
        "model": receipt["model"],
        "max_tokens": receipt["declared_output_budget"],
    }
    if receipt["thinking_override"] is False:
        options["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    agent = labbench.OpenAIZeroShotAgent(
        use_cot=True, open_answer=False, model_kwargs=options
    )
    index = 0
    finish_reasons = Counter()

    async def saved_create(**request):
        nonlocal index
        attempt = json.loads(attempts[index].read_text())
        index += 1
        if "error_type" in attempt or "response" not in attempt:
            raise ValueError("failed or incomplete attempt requires separate review")
        compare(
            request,
            attempt["request"],
            "original prompt, encoded images and model profile",
        )
        response = attempt["response"]
        compare(len(response["choices"]), 1, "single original response")
        finish_reasons[str(response["choices"][0].get("finish_reason"))] += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=response["choices"][0]["message"]["content"]
                    )
                )
            ]
        )

    # Reuse author prompt/image encoding/parser/evaluator, replacing only transport.
    # No saved code or model output is executed.
    agent.client.chat.completions.create = saved_create
    random.seed(receipt["sample_seed"])
    try:
        rescored = await evaluator.score_agent(agent.run_task, n_threads=1)
    finally:
        await agent.client.close()
    normalized = {
        str(task_id): {
            **result,
            "instance": result["instance"].model_dump(),
            "input": result["input"].model_dump(),
        }
        for task_id, result in rescored["results"].items()
    }
    compare(
        json.loads(json.dumps(normalized, default=str)),
        saved["results"],
        "original inputs, targets, parser and per-task scores",
    )
    compare(
        json.loads(json.dumps(agent.task_buffer, default=str)),
        buffer,
        "original prompt and response parsing buffer",
    )
    for key, value in rescored.items():
        if key != "results":
            compare(value, saved[key], "official aggregate and subset metrics")
    compare(
        rescored["metrics_all"],
        category_record(receipt, category)["metrics"],
        "receipt metrics",
    )
    return {
        "category": category,
        "saved_tasks_verified": index,
        "metrics": rescored["metrics_all"],
        "finish_reasons": dict(finish_reasons),
        "parsed_answers": sum(
            item["agent_output"] is not None for item in normalized.values()
        ),
    }


def category_record(receipt, category):
    records = [item for item in receipt["categories"] if item["category"] == category]
    if len(records) != 1:
        raise ValueError("missing or duplicate category receipt")
    return records[0]


async def execute(args):
    verify_source(args.source_root, args.archive)
    os.environ["PUBLIC_RELEASE"] = "True"
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:1/v1"
    os.environ["OPENAI_API_KEY"] = (
        "model-free-unused-value"  # pragma: allowlist secret - unused fake local SDK value
    )
    sys.path.insert(0, str(args.source_root.resolve()))
    import labbench

    compare(
        Path(labbench.__file__).resolve(),
        args.source_root.resolve() / "labbench/__init__.py",
        "pinned publisher import",
    )
    receipt = json.loads((args.run_root / "receipt-private.json").read_text())
    compare(receipt["original_cot_template"], True, "original CoT template")
    results = []
    for category in args.categories:
        compare(
            category_record(receipt, category)["status"],
            "OFFICIAL_SCORING_COMPLETED_PENDING_SAVED_AUDIT",
            "completed category",
        )
        results.append(await audit_category(labbench, category, receipt, args.run_root))
    verify_source(args.source_root, args.archive)
    return {
        "source_id": "LABBENCH",
        "status": "SAVED_SELECTED_CATEGORIES_RECONSTRUCTED_AND_RESCORED",
        "categories": results,
        "new_model_calls": 0,
        "source_checkbox_eligible": False,
        "note": "Source completion also requires all planned categories and separate delivery/consumer evidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--categories", choices=CATEGORIES, nargs="+", required=True)
    args = parser.parse_args()
    if len(set(args.categories)) != len(args.categories):
        parser.error("duplicate categories are not accepted")
    print(json.dumps(asyncio.run(execute(args)), indent=2))


if __name__ == "__main__":
    main()
