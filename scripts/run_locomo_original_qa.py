"""Pinned LoCoMo text QA generator/scorer with an explicit missing-key shim.

Original data and publisher files stay external and byte unchanged. This is not
RAG, image inference, research observation, or a publisher-model reproduction.
"""

import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import string
import time
from types import SimpleNamespace
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DATA_SHA = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"  # pragma: allowlist secret - publisher data checksum
GPT_SHA = "5fc977375878199735acd28fba5ae6f4d657fa0e000c0d2918a90c07b6035793"  # pragma: allowlist secret - publisher source checksum
EVAL_SHA = "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd"  # pragma: allowlist secret - publisher source checksum


def existing_answer_or_cat5_distractor(qa):
    """Keep every working original lookup; only missing cat5 gets its own field."""
    if "answer" in qa:
        return qa["answer"]
    if qa.get("category") == 5 and "adversarial_answer" in qa:
        return qa["adversarial_answer"]
    raise KeyError("answer")


class MissingCat5Answer(ast.NodeTransformer):
    def __init__(self):
        self.replacements = 0

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in {"qa", "line"}
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "answer"
            and isinstance(node.ctx, ast.Load)
        ):
            self.replacements += 1
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="_original_answer", ctx=ast.Load()),
                    args=[node.value],
                    keywords=[],
                ),
                node,
            )
        return node


def selected_definitions(path, checksum, functions, constants, namespace, compat):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("publisher source identity mismatch")
    nodes, found = [], set()
    for node in ast.parse(raw).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
            found.add(node.name)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in constants
        ):
            ast.literal_eval(node.value)
            nodes.append(node)
            found.add(node.targets[0].id)
    if found != functions | constants:
        raise ValueError("expected publisher definitions missing")
    tree = ast.Module(body=nodes, type_ignores=[])
    patch = MissingCat5Answer()
    if compat:
        tree = patch.visit(tree)
        namespace["_original_answer"] = existing_answer_or_cat5_distractor
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace, patch.replacements


def select_questions(data, *, indices=None, category=None, full=False):
    rows = [
        (p, q, qa) for p, parent in enumerate(data) for q, qa in enumerate(parent["qa"])
    ]
    if indices is not None:
        if category is not None or full:
            raise ValueError("explicit original indices cannot mix with category/full")
        if not indices or any(
            type(i) is not int or not 0 <= i < len(rows) for i in indices
        ):
            raise ValueError("original index outside source")
        if indices != sorted(set(indices)):
            raise ValueError("explicit indices must be unique original file order")
        return [(i, *rows[i]) for i in indices]
    filtered = [
        (i, *row)
        for i, row in enumerate(rows)
        if category is None or row[2]["category"] == category
    ]
    if not filtered:
        raise ValueError("empty original selection")
    return filtered if full else filtered[:3]


def local_endpoint(value):
    url = urlparse(value)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost"}
        or url.path.rstrip("/") != "/v1"
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("local HTTP /v1 endpoint without credentials required")
    return value.rstrip("/")


def load_functions(gpt_source, eval_source, *, compat, seed, preflight, callback):
    import numpy as np
    import regex
    import tiktoken
    from nltk.stem import PorterStemmer
    from tqdm import tqdm

    gpt, patched_gpt = selected_definitions(
        gpt_source,
        GPT_SHA,
        {"get_input_context", "get_cat_5_answer", "get_gpt_answers"},
        {
            "MAX_LENGTH",
            "PER_QA_TOKEN_BUDGET",
            "QA_PROMPT",
            "QA_PROMPT_CAT_5",
            "QA_PROMPT_BATCH",
            "CONV_START_PROMPT",
        },
        {
            "tiktoken": tiktoken,
            "random": random.Random(seed),
            "time": SimpleNamespace(sleep=lambda _: None) if preflight else time,
            "tqdm": tqdm,
            "run_chatgpt": callback,
        },
        compat,
    )
    scorer, patched_eval = selected_definitions(
        eval_source,
        EVAL_SHA,
        {"normalize_answer", "f1_score", "f1", "eval_question_answering"},
        set(),
        {
            "regex": regex,
            "string": string,
            "ps": PorterStemmer(),
            "Counter": Counter,
            "np": np,
        },
        compat,
    )
    if compat and (patched_gpt, patched_eval) != (4, 2):
        raise ValueError("unexpected compatibility patch surface")
    original_context = gpt["get_input_context"]

    def full_context_guard(conversation, question_tokens, encoding, args):
        actual = original_context(conversation, question_tokens, encoding, args)
        limit = gpt["MAX_LENGTH"][args.model]
        try:
            gpt["MAX_LENGTH"][args.model] = 10**9
            complete = original_context(conversation, question_tokens, encoding, args)
        finally:
            gpt["MAX_LENGTH"][args.model] = limit
        if actual != complete:
            raise ValueError(
                "full original conversation exceeds publisher context; no truncation"
            )
        return actual

    gpt["get_input_context"] = full_context_guard
    return gpt, scorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--gpt-source", type=Path, required=True)
    parser.add_argument("--evaluation-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument("--flat-index", type=int, action="append")
    parser.add_argument("--category", type=int, choices=range(1, 6))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-cat5-missing-answer-compat", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    endpoint = local_endpoint(args.endpoint)
    raw = args.data_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA:
        raise ValueError("original dataset identity mismatch")
    data = json.loads(raw)
    selection = select_questions(
        data, indices=args.flat_index, category=args.category, full=args.full
    )
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "LOCOMO",
        "status": "STARTED",
        "selected_original_flat_indices": [r[0] for r in selection],
        "publisher_profile": "gpt-4-turbo, no RAG, batch_size1, original32-token budget",
        "model": args.model,
        "compatibility_missing_answer_lookup_enabled": args.allow_cat5_missing_answer_compat,
        "original_source_files_rewritten": False,
        "original_dataset_rewritten": False,
        "random_seed": args.seed,
        "preflight_only": args.preflight_only,
        "native_publisher_sleep_preserved": not args.preflight_only,
        "diagnostic_placeholder_answers_are_native_results": False,
        "model_calls": 0,
        "samples": [],
    }
    current = []

    def post(path, body):
        request = Request(
            path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=300) as response:
            return json.load(response)

    def generate(query, **kwargs):
        if (
            kwargs["num_tokens_request"] != 32
            or kwargs["num_gen"] != 1
            or kwargs["temperature"] != 0
            or kwargs["model"] != "gpt-4-turbo"
            or kwargs["use_16k"] is not False
        ):
            raise ValueError("unexpected original generation configuration")
        flat, parent_index, question_index, qa = current.pop(0)
        item = {
            "original_flat_index": flat,
            "parent_index": parent_index,
            "question_index": question_index,
            "category": qa["category"],
            "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        }
        if args.preflight_only:
            receipt["samples"].append(item)
            return ""  # Only CPU prompt preparation; never scored or called a model.
        messages = [{"role": "user", "content": query}]
        tokenized = post(
            endpoint.removesuffix("/v1") + "/tokenize",
            {
                "model": args.model,
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        if tokenized["count"] + 32 > 131072:
            raise ValueError(
                "full original input plus output budget exceeds deployed context"
            )
        body = {
            "model": args.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 32,
            "n": 1,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        receipt["model_calls"] += 1
        response = post(endpoint + "/chat/completions", body)
        path = args.output / f"row-{flat}.private.json"
        path.write_text(
            json.dumps(
                {"request": body, "response": response, "original_qa": qa},
                ensure_ascii=False,
            )
        )
        if (
            len(response["choices"]) != 1
            or response["usage"]["prompt_tokens"] != tokenized["count"]
        ):
            raise ValueError("native response/token identity mismatch")
        choice = response["choices"][0]
        if choice["finish_reason"] not in {"stop", "length"} or not isinstance(
            choice["message"]["content"], str
        ):
            raise ValueError("incomplete native response")
        item.update(
            prompt_tokens=tokenized["count"],
            completion_tokens=response["usage"]["completion_tokens"],
            finish_reason=choice["finish_reason"],
            private_response_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        receipt["samples"].append(item)
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
        return choice["message"]["content"]

    try:
        generator, scorer = load_functions(
            args.gpt_source,
            args.evaluation_source,
            compat=args.allow_cat5_missing_answer_compat,
            seed=args.seed,
            preflight=args.preflight_only,
            callback=generate,
        )
        for parent_index in sorted({row[1] for row in selection}):
            group = [row for row in selection if row[1] == parent_index]
            current[:] = group
            inputs = deepcopy(data[parent_index])
            inputs["qa"] = [deepcopy(row[3]) for row in group]
            output = deepcopy(inputs)
            native_args = SimpleNamespace(
                model="gpt-4-turbo", use_rag=False, batch_size=1, overwrite=True
            )
            generator["get_gpt_answers"](inputs, output, "prediction", native_args)
            if current:
                raise ValueError(
                    "original generator did not consume complete selection"
                )
            if not args.preflight_only:
                (
                    args.output / f"parent-{parent_index}-scoring.private.json"
                ).write_text(json.dumps(output, ensure_ascii=False))
                scores, _, _ = scorer["eval_question_answering"](
                    output["qa"], eval_key="prediction"
                )
                if len(scores) != len(group):
                    raise ValueError("official scorer output size mismatch")
                for item, score in zip(
                    receipt["samples"][-len(group) :], scores, strict=True
                ):
                    item["publisher_score"] = float(score)
        if hashlib.sha256(args.data_file.read_bytes()).hexdigest() != DATA_SHA:
            raise ValueError("original dataset changed during execution")
        if (
            hashlib.sha256(args.gpt_source.read_bytes()).hexdigest() != GPT_SHA
            or hashlib.sha256(args.evaluation_source.read_bytes()).hexdigest()
            != EVAL_SHA
        ):
            raise ValueError("original publisher source changed during execution")
        receipt["status"] = (
            "ORIGINAL_PROMPTS_PREPARED_NO_GENERATION"
            if args.preflight_only
            else "ORIGINAL_QA_GENERATED_AND_PUBLISHER_SCORED"
        )
    except BaseException as error:
        receipt.update(status="EXECUTION_ERROR", error_type=type(error).__name__)
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
