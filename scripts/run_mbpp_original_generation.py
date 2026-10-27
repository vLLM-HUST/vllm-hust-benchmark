"""Original MBPP full-test tasks with publisher three-shot text prompts.

Generation only. Never executes predicted/reference Python code on the host.
"""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_mbpp_original_assets import PINS
from prepare_xsum_original_assets import bound_path, verify
from run_oasst_original_continuation import post, validate_endpoint


def original_prompt(task, examples):
    if [item["task_id"] for item in examples] != [2, 3, 4]:
        raise ValueError("publisher paper demonstration IDs 2/3/4 required")
    if type(task.get("task_id")) is not int or not 11 <= task["task_id"] <= 510:
        raise ValueError("original test task ID required")

    def block(item, solved):
        tests = item.get("test_list")
        if (
            not isinstance(item.get("text"), str)
            or not isinstance(tests, list)
            or len(tests) != 3
            or any(not isinstance(t, str) for t in tests)
        ):
            raise ValueError("original full task text and all three tests required")
        prefix = (
            "You are an expert Python programmer, and here is your task: "
            + item["text"]
            + " Your code should pass these tests:\n\n"
            + "\n".join(tests)
            + "\n[BEGIN]\n"
        )
        if solved:
            if not isinstance(item.get("code"), str):
                raise ValueError("original demonstration solution required")
            return prefix + item["code"] + "\n[DONE]"
        return prefix

    return "\n\n".join([*(block(item, True) for item in examples), block(task, False)])


def completion_request(task, examples, model, max_tokens):
    return {
        "model": model,
        "prompt": original_prompt(task, examples),
        "temperature": 0,
        "max_tokens": max_tokens,
        "n": 1,
        "stop": ["[DONE]"],
    }


def selected_tasks(rows, *, start=0, count=3, full=False):
    ids = [row.get("task_id") for row in rows]
    if (
        len(ids) != 500
        or any(type(i) is not int for i in ids)
        or set(ids) != set(range(11, 511))
    ):
        raise ValueError("complete unchanged original full test task IDs required")
    if full:
        if start != 0 or count != 3:
            raise ValueError("full cannot be mixed with range selection")
        return list(enumerate(rows))
    if start < 0 or count < 3 or start + count > len(rows):
        raise ValueError("fixed range of at least three original test tasks required")
    return list(enumerate(rows))[start : start + count]


def generate(
    selected,
    examples,
    *,
    output,
    endpoint,
    model,
    max_tokens,
    context,
    transport=post,
    final_verify=None,
    source_identity=None,
):
    endpoint = validate_endpoint(endpoint)
    if not model or type(max_tokens) is not int or not 0 < max_tokens < context:
        raise ValueError(
            "explicit model, output budget and real context capacity required"
        )
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "MBPP",
        "status": "STARTED",
        "variant": "full",
        "selected_original_rows": [index for index, _ in selected],
        "selected_task_ids": [task["task_id"] for _, task in selected],
        "demonstration_task_ids": [2, 3, 4],
        "model": model,
        "max_tokens": max_tokens,
        "context_capacity": context,
        "transport": "raw_completions_no_chat_template",
        "prompt_protocol": "publisher README text template; all three tests, original demonstrations 2/3/4; blocks joined by two newlines",
        "paper_model_or_sampling_replication": False,
        "samples": [],
        "functional_tests_executed": False,
        "source_checkbox_eligible": False,
        "source_identity": source_identity,
    }

    def save(name, value):
        path = output / name
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        path.chmod(0o600)

    save("receipt-private.json", receipt)
    try:
        for index, task in selected:
            saved = {
                "original_row_index": index,
                "original_task": task,
                "status": "PREPARING",
            }
            name = f"task-{task['task_id']}.private.json"
            save(name, saved)
            try:
                request = completion_request(task, examples, model, max_tokens)
                prompt = request["prompt"]
                saved["request"] = request
                save(name, saved)
                tokens = transport(
                    endpoint.removesuffix("/v1") + "/tokenize",
                    {"model": model, "prompt": prompt},
                )
                saved["tokenize"] = tokens
                save(name, saved)
                count = tokens.get("count")
                if type(count) is not int or count <= 0 or count + max_tokens > context:
                    raise ValueError(
                        "complete original prompt does not fit declared context"
                    )
                result = transport(endpoint + "/completions", request)
                saved["response"] = result
                save(name, saved)
                choices, usage = result.get("choices"), result.get("usage", {})
                if (
                    not isinstance(choices, list)
                    or len(choices) != 1
                    or not isinstance(choices[0].get("text"), str)
                    or choices[0].get("finish_reason") not in {"stop", "length"}
                    or usage.get("prompt_tokens") != count
                    or type(usage.get("completion_tokens")) is not int
                    or not 0 <= usage["completion_tokens"] <= max_tokens
                ):
                    raise ValueError("terminal raw completion integrity not verified")
                saved["status"] = "GENERATED_NOT_FUNCTIONALLY_SCORED"
                receipt["samples"].append(
                    {
                        "original_row_index": index,
                        "task_id": task["task_id"],
                        "finish_reason": choices[0]["finish_reason"],
                        "usage": usage,
                        "private_file": name,
                    }
                )
            except Exception as error:
                saved.update(
                    status="FAILED_NOT_REPLACED", error_type=type(error).__name__
                )
                raise
            finally:
                save(name, saved)
                save("receipt-private.json", receipt)
        if not receipt["samples"]:
            raise ValueError("no original test tasks selected")
        if final_verify:
            final_verify()
        receipt["status"] = "ORIGINAL_TEST_GENERATION_COMPLETE_SCORING_PENDING"
    except Exception as error:
        receipt.update(status="FAILED_NOT_COMPLETE", error_type=type(error).__name__)
        raise
    finally:
        save("receipt-private.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--context-capacity", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    root, out = args.assets.resolve(), args.output.resolve()
    if root.is_relative_to(out) or out.is_relative_to(root):
        parser.error("fresh private results must be separate from original assets")
    pins = json.loads(PINS.read_text("utf-8"))

    def verify_all():
        for item in pins["dataset"]["files"]:
            verify(bound_path(root, "data", item), item)

    verify_all()
    import pyarrow.parquet as pq

    rows = pq.read_table(root / "data/full/test-00000-of-00001.parquet").to_pylist()
    prompt_rows = pq.read_table(
        root / "data/full/prompt-00000-of-00001.parquet"
    ).to_pylist()
    if len(prompt_rows) != 10 or {item["task_id"] for item in prompt_rows} != set(
        range(1, 11)
    ):
        raise ValueError("complete original prompt split required")
    lookup = {item["task_id"]: item for item in prompt_rows}
    selected = selected_tasks(rows, start=args.start, count=args.count, full=args.full)
    receipt = generate(
        selected,
        [lookup[i] for i in (2, 3, 4)],
        output=out,
        endpoint=args.endpoint,
        model=args.model,
        max_tokens=args.max_tokens,
        context=args.context_capacity,
        final_verify=verify_all,
        source_identity={
            "repo_id": pins["dataset"]["repo_id"],
            "revision": pins["dataset"]["revision"],
            "asset_manifest_sha256": hashlib.sha256(PINS.read_bytes()).hexdigest(),
            "files": pins["dataset"]["files"],
        },
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "tasks": len(receipt["samples"]),
                "source_checkbox_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
