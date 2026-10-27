"""Fixed original BigCodeBench Instruct generation only; never executes code.

Requires separately acquired publisher files and original v0.1.4 data. Functional
scoring, pass@1, a complete publisher environment and performance are not claimed.
"""

import argparse
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DATA_SHA = "d9a4965821c9507ebdfb551c288656b2d5fe553234f5183044333ca8a4018267"  # pragma: allowlist secret - original public data checksum
UTILITY_SHA = "a58690bedcdcb0bac9062bcef94c0ecbe1ef204e7fb7bebf6a6423e6a5900eaf"  # pragma: allowlist secret - publisher file checksum
REQUEST_SHA = "f7d147d3ee01dc77db9c45f1b567b76614331d8ddf9b46db0cfc39081648457e"  # pragma: allowlist secret - publisher file checksum
CODE_REVISION = "09dd993f46c3fbf3a799465bb96d524edcb0b199"  # pragma: allowlist secret - public upstream revision
INSTRUCTION_PREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
RESPONSE_PREFIX = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_indices(total, *, indices=None, start=None, count=None, full=False):
    require(type(total) is int and total >= 3, "at least three original tasks required")
    if indices is not None:
        require(
            not full and start is None and count is None, "conflicting selection modes"
        )
        chosen = list(indices)
    elif full:
        require(start is None and count is None, "conflicting selection modes")
        chosen = list(range(total))
    else:
        start = 0 if start is None else start
        count = 3 if count is None else count
        require(type(start) is int and type(count) is int, "integer range required")
        require(
            start >= 0 and 3 <= count <= total and start + count <= total,
            "invalid task range",
        )
        chosen = list(range(start, start + count))
    require(
        len(chosen) >= 3
        and all(type(i) is int and 0 <= i < total for i in chosen)
        and chosen == sorted(set(chosen)),
        "selection must contain at least three sorted unique original indices",
    )
    return chosen


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def origin(value):
    parsed = urlparse(value)
    require(
        parsed.scheme == "http"
        and parsed.hostname in {"localhost", "127.0.0.1"}
        and parsed.path in {"", "/"}
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment,
        "local HTTP origin without credentials required",
    )
    return value.rstrip("/")


def load_original(name, path, expected):
    require(digest(path) == expected, "publisher source identity mismatch")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_prompt(formatter, prompt):
    require(
        isinstance(prompt, str) and bool(prompt.strip()),
        "nonempty original prompt required",
    )
    return formatter.make_raw_chat_prompt(
        task_prompt=prompt,
        subset="full",
        split="instruct",
        instruction_prefix=INSTRUCTION_PREFIX,
        response_prefix=RESPONSE_PREFIX,
        tokenizer=None,
    )


def response_summary(response, expected_tokens):
    choices = response.get("choices")
    require(
        isinstance(choices, list) and len(choices) == 1,
        "one generation choice required",
    )
    choice = choices[0]
    require(isinstance(choice, dict), "invalid generation choice")
    content = choice.get("message", {}).get("content")
    require(isinstance(content, str) and bool(content.strip()), "empty generation")
    usage = response.get("usage", {})
    require(
        isinstance(usage, dict)
        and type(usage.get("prompt_tokens")) is int
        and usage["prompt_tokens"] == expected_tokens,
        "full prompt usage mismatch",
    )
    count = usage.get("completion_tokens")
    require(type(count) is int and 0 < count <= 1280, "invalid completion token count")
    require(
        usage.get("total_tokens") == expected_tokens + count, "inconsistent token usage"
    )
    finish = choice.get("finish_reason")
    return {
        "finish_reason": finish,
        "prompt_tokens": expected_tokens,
        "completion_tokens": count,
        "full_prompt_token_count_matches": True,
        "status": "GENERATION_TERMINATED"
        if finish == "stop"
        else "TRUNCATED_GENERATION"
        if finish == "length"
        else "INCOMPLETE_GENERATION",
        "generated_code_executed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--official-code", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--origin", default="http://127.0.0.1:19466", type=origin)
    parser.add_argument("--context-capacity", required=True, type=int)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--indices", type=int, nargs="+")
    selection.add_argument("--full", action="store_true")
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    os.umask(0o077)
    out = args.output.resolve()
    for root in [args.data_root.resolve(), args.official_code.resolve()]:
        require(
            not out.is_relative_to(root) and not root.is_relative_to(out),
            "output and original source trees must be separate",
        )
    path = args.data_root / "data/v0.1.4-00000-of-00001.parquet"
    require(digest(path) == DATA_SHA, "original data identity mismatch")
    utility_path = args.official_code / "bigcodebench/provider/utility.py"
    request_path = args.official_code / "bigcodebench/gen/util/openai_request.py"
    require(
        digest(utility_path) == UTILITY_SHA and digest(request_path) == REQUEST_SHA,
        "fixed original publisher files required",
    )
    import openai
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    require(len(rows) == 1140, "original v0.1.4 task count mismatch")
    indices = select_indices(
        len(rows),
        indices=args.indices,
        start=args.start,
        count=args.count,
        full=args.full,
    )
    selected = [rows[i] for i in indices]
    require(
        [row["task_id"] for row in rows] == [f"BigCodeBench/{i}" for i in range(1140)],
        "original release-order task selection mismatch",
    )
    formatter = load_original("bcb_original_formatter", utility_path, UTILITY_SHA)
    transport = load_original("bcb_original_transport", request_path, REQUEST_SHA)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "BIGCODEBENCH",
        "status": "STARTED",
        "samples": [],
        "code_revision": CODE_REVISION,
        "split": "instruct",
        "original_task_count": 1140,
        "selected_original_row_indices": indices,
        "model": args.model,
        "software": {
            p: metadata.version(p) for p in ["openai", "transformers", "pyarrow"]
        },
        "publisher_prompt_and_make_request_unchanged": True,
        "publisher_retry_loop_not_used": True,
        "full_reference_environment_reproduced": False,
        "max_completion_tokens": 1280,
        "temperature": 0,
        "top_p": 0.95,
        "server_thinking_setting": "existing service default; not changed by this entrypoint",
        "generated_code_executed": False,
        "functional_scoring": "NOT_RUN_REQUIRES_VERIFIED_SANDBOX",
        "pass_at_1_claim": False,
        "full_suite_claim": False,
        "performance_claim": False,
    }
    receipt_path = out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    client = openai.OpenAI(
        base_url=args.origin + "/v1",
        api_key="unused-local-service",  # pragma: allowlist secret - fixed dummy for the local service, not a credential
        timeout=300,
        max_retries=0,
    )
    try:
        for index, row in zip(indices, selected):
            sample = {"row_index": index, "task_id": row["task_id"]}
            try:
                message = format_prompt(formatter, row["instruct_prompt"])
                messages = [{"role": "user", "content": message}]
                request = Request(
                    args.origin + "/tokenize",
                    data=json.dumps(
                        {
                            "model": args.model,
                            "messages": messages,
                            "add_generation_prompt": True,
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=60) as response:
                    count = json.load(response)["count"]
                require(
                    type(count) is int
                    and count > 0
                    and count + 1280 <= args.context_capacity,
                    "full prompt exceeds declared context capacity",
                )
                reply = transport.make_request(
                    client,
                    message=message,
                    model=args.model,
                    max_tokens=1280,
                    temperature=0,
                    n=1,
                ).model_dump(mode="json")
                (out / f"{index}.private.json").write_text(
                    json.dumps(
                        {
                            "task_id": row["task_id"],
                            "messages": messages,
                            "response": reply,
                        }
                    ),
                    encoding="utf-8",
                )
                sample.update(response_summary(reply, count))
            except Exception as error:
                sample.update(
                    status="GENERATION_FAILED", error_type=type(error).__name__
                )
            receipt["samples"].append(sample)
            receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        require(digest(path) == DATA_SHA, "original data changed during execution")
        receipt["status"] = (
            "ORIGINAL_GENERATION_PROFILE_ATTEMPTED_NOT_FUNCTIONALLY_SCORED"
        )
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt))
        if not all(
            sample["status"] == "GENERATION_TERMINATED" for sample in receipt["samples"]
        ):
            raise SystemExit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
