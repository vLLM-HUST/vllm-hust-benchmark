"""Original GPQA zero-shot task formatting, seeded choices and answer parser.

Preserves publisher system instruction, 1000-token budget and default index69
skip. Explicit --include-index69 executes that unchanged original task as a
separate compatibility profile; never pretend the publisher loop did so.
Only model transport changes to a declared local served model. This is not the
publisher GPT-4 baseline or the retrieval/CoT/self-consistency protocols.
"""

import argparse
import ast
from collections import namedtuple
import hashlib
import inspect
import json
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace
import typing
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SOURCE_COMMIT = "56686c06f5e19865c153de0fdb11be3890014df7"  # pragma: allowlist secret - publisher commit
SOURCE_SHA256 = {
    "baselines/utils.py": "63189efa76292b0f67d80ecf416f38d3db582b6570bdec433df8869868a136d4",  # pragma: allowlist secret - publisher code digest
    "baselines/run_baseline.py": "8d9225c2446b215dcb5fd5230491ee7f49cc4a40ba836c3f379693a7d74d0660",  # pragma: allowlist secret - publisher code digest
}
DATA_SHA256 = {
    "main": "acdeeac8f622267f2cd727d7d474202ea08dec80f7d3c3593b3ef8644f19b8e3",  # pragma: allowlist secret - publisher data digest
    "diamond": "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305",  # pragma: allowlist secret - publisher data digest
    "extended": "0926ee24949d02ed6748eb75a2611546c34479e30ddc42efd01d6f1681aaa48a",  # pragma: allowlist secret - publisher data digest
}
ROW_COUNTS = {"main": 448, "diamond": 198, "extended": 546}


def endpoint(value):
    p = urlparse(value)
    if (
        p.scheme != "http"
        or p.hostname not in {"localhost", "127.0.0.1"}
        or p.path.rstrip("/") != "/v1"
        or p.username
        or p.password
        or p.query
        or p.fragment
    ):
        raise ValueError("local credential-free HTTP /v1 endpoint required")
    return value.rstrip("/")


def post(url, body):
    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=3600) as response:
        return json.load(response)


def original_components(source):
    trees = {}
    for relative, expected in SOURCE_SHA256.items():
        path = source / relative
        raw = path.read_bytes()
        if path.is_symlink() or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("publisher GPQA source identity mismatch")
        trees[relative] = ast.parse(raw)
    import pandas as pd

    namespace = {
        **vars(typing),
        "pd": pd,
        "random": random,
        "re": re,
        "OPENAI_MODELS": ["gpt-4"],
        "CHAT_MODELS": ["gpt-4"],
        "Example": namedtuple(
            "Example",
            ["question", "choice1", "choice2", "choice3", "choice4", "correct_index"],
        ),
    }
    names = {
        "load_examples",
        "base_prompt",
        "zero_shot_prompt",
        "select_and_call_model",
    }
    nodes = [
        n
        for n in trees["baselines/utils.py"].body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    classes = [
        n
        for n in trees["baselines/run_baseline.py"].body
        if isinstance(n, ast.ClassDef) and n.name == "AnswerPredictor"
    ]
    if len(nodes) != 4 or len(classes) != 1:
        raise ValueError("original GPQA component inventory changed")
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=nodes + classes, type_ignores=[])
            ),
            "<pinned-gpqa-components>",
            "exec",
        ),
        namespace,
    )
    budget = (
        inspect.signature(namespace["select_and_call_model"])
        .parameters["max_tokens"]
        .default
    )
    if budget != 1000:
        raise ValueError("publisher generation budget changed")
    return namespace, budget


def run(args):
    api = endpoint(args.api_base)
    namespace, budget = original_components(args.source)
    path = args.data_root / f"gpqa_{args.split}.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATA_SHA256[args.split]:
        raise ValueError("original GPQA CSV identity mismatch")
    examples = namespace["load_examples"](str(path), args.seed)
    if len(examples) != ROW_COUNTS[args.split]:
        raise ValueError("original GPQA split row count mismatch")
    offsets = list(range(len(examples))) if args.full else [0, 1, 2]
    args.output.mkdir(mode=0o700, exist_ok=False)
    receipt = {
        "source_id": "GPQA",
        "source_commit": SOURCE_COMMIT,
        "split": args.split,
        "data_sha256": digest,
        "model": args.model,
        "publisher_protocol": "zero_shot",
        "seed": args.seed,
        "selected_original_offsets": offsets,
        "generation_budget": budget,
        "publisher_index69_skip_retained": not getattr(args, "include_index69", False),
        "index69_compatibility": "EXECUTE_UNCHANGED_ORIGINAL_ROW_SKIPPED_BY_PUBLISHER_LOOP"
        if getattr(args, "include_index69", False)
        else None,
        "status": "GENERATING",
        "completed_generations": 0,
        "correct": 0,
        "samples": [],
        "automatic_retries": 0,
        "publisher_model_baseline_reproduced": False,
        "source_checkbox_eligible": False,
        "raw_payloads_published": False,
        "started_at_unix": time.time(),
        "reused_generations": 0,
        "model_generation_calls_attempted": 0,
    }

    def save():
        temporary = args.output / "receipt.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output / "receipt.json")

    save()
    current = {}
    cached = {}

    def fill_cache():
        for previous in getattr(args, "reuse", []) or []:
            prior = json.loads((previous / "receipt.json").read_text("utf-8"))
            if (
                prior["source_commit"] != SOURCE_COMMIT
                or prior["model"] != args.model
                or prior["publisher_protocol"] != "zero_shot"
                or prior["generation_budget"] != budget
                or prior["data_sha256"] not in DATA_SHA256.values()
                or prior["status"]
                != "ORIGINAL_ZERO_SHOT_EXECUTION_AND_SCORING_COMPLETED_REQUIRES_AUDIT"
            ):
                raise ValueError(
                    "prior original GPQA run is not compatible for exact-request reuse"
                )
            for sample in prior["samples"]:
                if sample.get("publisher_skipped"):
                    continue
                file = previous / f"row-{sample['original_offset']}.private.json"
                raw = file.read_bytes()
                if hashlib.sha256(raw).hexdigest() != sample["private_record_sha256"]:
                    raise ValueError("prior GPQA private record identity mismatch")
                record = json.loads(raw)
                cached[json.dumps(record["request"], sort_keys=True)] = record[
                    "response"
                ]

    def transport(**original_request):
        if (
            original_request["model"] != "gpt-4"
            or original_request["max_tokens"] != budget
            or original_request["temperature"] != 0
            or original_request["stop"] is not None
        ):
            raise ValueError("original GPQA request configuration changed")
        request = {**original_request, "model": args.model}
        count = post(
            api.removesuffix("/v1") + "/tokenize",
            {"model": args.model, "messages": request["messages"]},
        )["count"]
        if type(count) is not int or count + budget > args.context_window:
            raise ValueError(
                "complete original GPQA prompt exceeds model context; no truncation"
            )
        key = json.dumps(request, sort_keys=True)
        reused = key in cached
        if reused:
            response = cached[key]
            receipt["reused_generations"] += 1
        else:
            receipt["model_generation_calls_attempted"] += 1
            save()
            response = post(api + "/chat/completions", request)
            cached[key] = response
        answer = response["choices"][0]["message"]["content"]
        if (
            not isinstance(answer, str)
            or response["model"] != args.model
            or response["usage"]["prompt_tokens"] != count
        ):
            raise ValueError("invalid model response or input identity mismatch")
        current.update(
            request=request,
            response=response,
            input_tokens=count,
            reused_saved_response=reused,
        )
        return answer

    namespace["client"] = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=transport))
    )
    try:
        fill_cache()
        for offset in offsets:
            example = examples[offset]
            if offset == 69 and not getattr(args, "include_index69", False):
                receipt["samples"].append(
                    {
                        "original_offset": offset,
                        "publisher_skipped": True,
                        "correct": False,
                    }
                )
                save()
                continue
            prompt = namespace["zero_shot_prompt"](example)
            answer = namespace["select_and_call_model"](prompt, "gpt-4", "sample")
            parsed = namespace["AnswerPredictor"].parse_sampled_answer(answer)
            correct = (
                parsed is not None
                and namespace["AnswerPredictor"].LETTER_TO_INDEX[parsed]
                == example.correct_index
            )
            record = {
                "original_offset": offset,
                "original_shuffled_example": example._asdict(),
                **current,
                "original_parsed_answer": parsed,
                "correct": correct,
            }
            saved = args.output / f"row-{offset}.private.json"
            saved.write_text(
                json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            receipt["samples"].append(
                {
                    "original_offset": offset,
                    "private_record_sha256": hashlib.sha256(
                        saved.read_bytes()
                    ).hexdigest(),
                    "correct": correct,
                    "answer_extracted": parsed is not None,
                }
            )
            receipt["completed_generations"] += 1
            receipt["correct"] += int(correct)
            save()
        receipt["accuracy"] = receipt["correct"] / len(
            examples if args.full else offsets
        )
        receipt["status"] = (
            "ORIGINAL_ZERO_SHOT_EXECUTION_AND_SCORING_COMPLETED_REQUIRES_AUDIT"
        )
    except Exception as exc:
        receipt["status"] = "FAILED_NO_REPLACEMENT_OR_RETRY"
        receipt["error_type"], receipt["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        receipt["finished_at_unix"] = time.time()
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=list(DATA_SHA256), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--context-window", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--include-index69",
        action="store_true",
        help="explicit compatibility mode: execute unchanged original row69 rather than preserve publisher loop skip; all other task settings stay identical",
    )
    parser.add_argument(
        "--reuse",
        type=Path,
        nargs="*",
        default=[],
        help="reuse only hash-checked byte-equivalent native requests from completed original runs",
    )
    print(json.dumps(run(parser.parse_args())))


if __name__ == "__main__":
    main()
