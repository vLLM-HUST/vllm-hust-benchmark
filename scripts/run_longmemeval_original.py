"""Pinned LongMemEval full-history/oracle generation, never shortened histories.

Uses the unchanged publisher prepare_prompt function and generation defaults.
The local model/tokenizer transport replaces the publisher CLI's fixed model
allowlist. Saved hypotheses can be evaluated by the original evaluate_qa.py;
generation alone is explicitly not a completed benchmark or official score.
"""

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SOURCE_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"  # pragma: allowlist secret - public publisher commit
GENERATOR_SHA256 = "4f1eb3c69d7ad40f04065b9c0bc86f6582441018fc6ff751d162d66c95baf672"  # pragma: allowlist secret - publisher code digest
DATA_FILES = {
    "oracle": (
        "longmemeval_oracle.json",
        "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",  # pragma: allowlist secret - publisher data digest
    ),
    "s": (
        "longmemeval_s_cleaned.json",
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",  # pragma: allowlist secret - publisher data digest
    ),
    "m": (
        "longmemeval_m_cleaned.json",
        "9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f",  # pragma: allowlist secret - publisher data digest
    ),
}


def endpoint(value):
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("only credential-free local HTTP /v1 endpoint permitted")
    return value.rstrip("/")


def post(url, body):
    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=3600) as response:
        return json.load(response)


def publisher_prompt(source):
    path = source / "src/generation/run_generation.py"
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != GENERATOR_SHA256:
        raise ValueError("publisher generation source identity mismatch")
    functions = [
        node
        for node in ast.parse(raw).body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_prompt"
    ]
    if len(functions) != 1:
        raise ValueError("unique publisher prepare_prompt required")
    namespace = {"json": json}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
            "<pinned-longmemeval-prompt>",
            "exec",
        ),
        namespace,
    )
    return namespace["prepare_prompt"]


class NoTruncationTokenizer:
    """Original function only reads input_ids lengths; no tensor or decode needed."""

    def __init__(self, api, model):
        self.api, self.model = api, model

    def __call__(self, text, *, max_length, truncation, return_tensors):
        if truncation:
            raise ValueError(
                "original history exceeds publisher retrieval budget; truncation prohibited"
            )
        result = post(
            self.api.removesuffix("/v1") + "/tokenize",
            {"model": self.model, "prompt": text},
        )
        tokens = result["tokens"]
        if not isinstance(tokens, list) or result["count"] != len(tokens):
            raise ValueError("invalid tokenizer response")
        return {"input_ids": [tokens]}


def selection(rows, offsets):
    if offsets is None:
        offsets = list(range(len(rows)))
    if (
        not offsets
        or len(offsets) != len(set(offsets))
        or any(type(i) is not int or i < 0 or i >= len(rows) for i in offsets)
    ):
        raise ValueError("nonempty unique physical offsets required")
    return offsets


def complete_prompt(function, row, tokenizer, context_window, *, cot=True):
    if context_window <= 1800:
        raise ValueError("context window too small")
    budget = 800 if cot else 500
    # Original orig-session and oracle-session branches are identical: all
    # sessions from the explicitly selected original variant, never oracle
    # substitution for S/M. topk=len prevents implicit latest-50 selection.
    return function(
        deepcopy(row),
        "orig-session",
        len(row["haystack_sessions"]),
        False,
        "json",
        cot,
        tokenizer,
        "huggingface",
        context_window - budget - 1000,
        "none",
        con=False,
    ), budget


def run(args):
    api = endpoint(args.api_base)
    function = publisher_prompt(args.source)
    name, expected = DATA_FILES[args.variant]
    path = args.data_root / name
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("original data identity mismatch")
    rows = json.loads(raw)
    del raw
    if len(rows) != 500 or len({r["question_id"] for r in rows}) != 500:
        raise ValueError("original 500 unique questions required")
    offsets = selection(rows, None if args.full else [0, 1, 2])
    args.output.mkdir(mode=0o700, exist_ok=False)
    receipt = {
        "source_id": "LONGMEMEVAL",
        "source_commit": SOURCE_COMMIT,
        "data_sha256": expected,
        "variant": args.variant,
        "model": args.model,
        "selected_physical_offsets": offsets,
        "generation_budget": 500 if args.direct else 800,
        "complete_history_no_topk_reduction": True,
        "history_truncation_permitted": False,
        "status": "PREFLIGHT",
        "completed_rows": 0,
        "samples": [],
        "official_judge_executed": False,
        "source_checkbox_eligible": False,
        "raw_payloads_published": False,
    }

    def save():
        temporary = args.output / "receipt.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output / "receipt.json")

    save()
    try:
        tokenizer = NoTruncationTokenizer(api, args.model)
        prepared = []
        for offset in offsets:
            prompt, budget = complete_prompt(
                function,
                rows[offset],
                tokenizer,
                args.context_window,
                cot=not args.direct,
            )
            count = post(
                api.removesuffix("/v1") + "/tokenize",
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )["count"]
            if type(count) is not int or count + budget > args.context_window:
                raise ValueError(
                    "complete original prompt exceeds deployed context; no replacement row"
                )
            prepared.append((offset, prompt, budget, count))
        receipt["status"] = "GENERATING"
        save()
        with (args.output / "hypotheses.private.jsonl").open(
            "x", encoding="utf-8"
        ) as hypotheses:
            for offset, prompt, budget, count in prepared:
                request = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "n": 1,
                    "temperature": 0,
                    "max_tokens": budget,
                }
                response = post(api + "/chat/completions", request)
                answer = response["choices"][0]["message"]["content"]
                if (
                    not isinstance(answer, str)
                    or response["usage"]["prompt_tokens"] != count
                ):
                    raise ValueError(
                        "invalid response or input token identity mismatch"
                    )
                saved = args.output / f"row-{offset}.private.json"
                record = {
                    "original_row": rows[offset],
                    "request": request,
                    "response": response,
                    "input_tokens": count,
                }
                saved.write_text(
                    json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                hypotheses.write(
                    json.dumps(
                        {
                            "question_id": rows[offset]["question_id"],
                            "hypothesis": answer.strip(),
                        }
                    )
                    + "\n"
                )
                hypotheses.flush()
                receipt["samples"].append(
                    {
                        "original_offset": offset,
                        "private_record_sha256": hashlib.sha256(
                            saved.read_bytes()
                        ).hexdigest(),
                        "finish_reason": response["choices"][0]["finish_reason"],
                    }
                )
                receipt["completed_rows"] += 1
                save()
        receipt["status"] = "ORIGINAL_GENERATIONS_COMPLETED_SCORING_PENDING"
    except Exception as exc:
        receipt["status"] = "BLOCKED_OR_FAILED_NO_RETRY"
        receipt["error_type"], receipt["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--variant", choices=list(DATA_FILES), required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-window", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="publisher direct500 instead of default CoT800",
    )
    print(json.dumps(run(parser.parse_args())))


if __name__ == "__main__":
    main()
