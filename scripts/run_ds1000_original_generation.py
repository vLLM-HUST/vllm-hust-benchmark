"""Selected original DS-1000 raw completions on a local service; no code execution.

Publisher CUDA runner is parsed as data, never imported. Generation keeps its
1024-token greedy budget and only splits end markers after decoding finishes.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DATA_SHA = "26a492838b05f9d030b2f08a5715c6bafecafb601c157da912bf417fd8b0d363"  # pragma: allowlist secret - original public dataset checksum
SOURCE_SHA = "8f1883705ace8cb79d5b84f7c281cd02631b8cba6077acbf6ba891964de49d59"  # pragma: allowlist secret - publisher source checksum
CODE_REVISION = "b39aab71da6d23ef8d3cac59a7c5f834516ab334"  # pragma: allowlist secret - public source revision


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def origin(value):
    parsed = urlparse(value)
    require(
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost"}
        and parsed.path in {"", "/"}
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment,
        "local HTTP origin without credentials required",
    )
    return value.rstrip("/")


def parse_publisher_profile(text):
    values = {}
    for node in ast.walk(ast.parse(text)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name in {"generation_config", "stop_words"}:
                require(name not in values, "ambiguous publisher configuration")
                values[name] = ast.literal_eval(node.value)
    config = values.get("generation_config")
    require(
        isinstance(config, dict)
        and type(config.get("do_sample")) is bool
        and type(config.get("max_new_tokens")) is int
        and type(config.get("num_beams")) is int,
        "invalid publisher configuration types",
    )
    require(
        values
        == {
            "generation_config": {
                "do_sample": False,
                "max_new_tokens": 1024,
                "num_beams": 1,
            },
            "stop_words": ["</code>", "# SOLUTION END"],
        },
        "unsupported publisher profile",
    )
    return values


def request_body(prompt, model):
    require(
        isinstance(prompt, str) and bool(prompt.strip()),
        "nonempty original prompt required",
    )
    return {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 1024,
        "n": 1,
        "stream": False,
        "add_special_tokens": True,
        "skip_special_tokens": True,
    }


def postprocess(text, words):
    for word in words:
        text = text.split(word)[0]
    return text


def response_summary(response, expected_tokens, words):
    choices = response.get("choices")
    require(
        isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], dict),
        "one completion choice required",
    )
    choice = choices[0]
    text = choice.get("text")
    require(isinstance(text, str), "raw completion text required")
    usage = response.get("usage")
    require(
        isinstance(usage, dict)
        and type(usage.get("prompt_tokens")) is int
        and usage["prompt_tokens"] == expected_tokens,
        "full prompt usage mismatch",
    )
    count = usage.get("completion_tokens")
    require(type(count) is int and 0 <= count <= 1024, "invalid completion usage")
    require(
        usage.get("total_tokens") == expected_tokens + count, "inconsistent token usage"
    )
    candidate = postprocess(text, words)
    return {
        "status": "GENERATED_NOT_FUNCTIONALLY_SCORED",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": expected_tokens,
        "completion_tokens": count,
        "full_prompt_token_count_matches": True,
        "postprocess_only_after_full_generation": True,
        "end_markers_seen": [word for word in words if word in text],
        "raw_output_nonempty": bool(text.strip()),
        "postprocessed_solution_nonempty": bool(candidate.strip()),
    }, candidate


def select_indices(total, *, indices=None, start=None, count=None, full=False):
    """Freeze original positions before any model call; never replace failures."""
    if indices is not None or full:
        require(
            start is None and count is None, "explicit selection conflicts with range"
        )
        require(
            not (indices is not None and full), "full and explicit selection conflict"
        )
        selected = list(range(total)) if full else list(indices)
    else:
        first, size = (0 if start is None else start), (3 if count is None else count)
        require(
            type(first) is int
            and type(size) is int
            and first >= 0
            and size >= 3
            and first + size <= total,
            "valid original range of at least3 required",
        )
        selected = list(range(first, first + size))
    require(
        len(selected) >= 3
        and all(type(i) is int and 0 <= i < total for i in selected)
        and selected == sorted(set(selected)),
        "sorted unique original positions, at least3, required",
    )
    return selected


def profile_exit_code(samples, expected_count=3):
    if len(samples) != expected_count or any(
        row.get("status") != "GENERATED_NOT_FUNCTIONALLY_SCORED" for row in samples
    ):
        return 1
    if any(row["finish_reason"] not in {"stop", "length"} for row in samples):
        return 1
    return 2 if any(row["finish_reason"] == "length" for row in samples) else 0


def post(base, path, body):
    request = Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--publisher-inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--origin", type=origin, default="http://127.0.0.1:19466")
    parser.add_argument("--context-capacity", type=int, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--indices", type=int, nargs="+")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    os.umask(0o077)
    root, out, publisher = (
        args.data_root.resolve(),
        args.output.resolve(),
        args.publisher_inference.resolve(),
    )
    require(
        not out.is_relative_to(root)
        and not root.is_relative_to(out)
        and not publisher.is_relative_to(out),
        "output must be separate from original inputs",
    )
    path = root / "test.jsonl"
    require(digest(path) == DATA_SHA, "original dataset identity mismatch")
    require(digest(publisher) == SOURCE_SHA, "publisher source identity mismatch")
    profile = parse_publisher_profile(publisher.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(
        len(rows) == 1000
        and [row["metadata"]["problem_id"] for row in rows[:3]] == [0, 1, 2],
        "original test task order/count mismatch",
    )
    selected = select_indices(
        len(rows),
        indices=args.indices,
        start=args.start,
        count=args.count,
        full=args.full,
    )
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "DS1000",
        "status": "STARTED",
        "samples": [],
        "code_revision": CODE_REVISION,
        "selected_original_row_indices": selected,
        "model": args.model,
        "publisher_literal_config": profile,
        "api_stop_parameter_omitted": True,
        "no_chat_wrapper_or_prompt_rewrite": True,
        "transport": "local NPU raw completions, not publisher local HF CUDA batch",
        "generated_code_executed": False,
        "code_context_executed": False,
        "scorer_runs": 0,
        "full_reference_environment_reproduced": False,
        "full_suite_claim": False,
        "performance_claim": False,
        "source_checkbox_eligible": False,
    }
    receipt_path = out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    for index in selected:
        row = rows[index]
        sample = {
            "row_index": index,
            "original_problem_id": row["metadata"]["problem_id"],
            "original_library": row["metadata"].get("library"),
        }
        try:
            body = request_body(row["prompt"], args.model)
            count = post(
                args.origin,
                "/tokenize",
                {
                    "model": args.model,
                    "prompt": row["prompt"],
                    "add_special_tokens": True,
                },
            )["count"]
            require(
                type(count) is int
                and count > 0
                and count + 1024 <= args.context_capacity,
                "full prompt exceeds declared context capacity",
            )
            response = post(args.origin, "/v1/completions", body)
            (out / f"{index}.private.json").write_text(
                json.dumps({"request": body, "response": response}), encoding="utf-8"
            )
            summary, candidate = response_summary(
                response, count, profile["stop_words"]
            )
            sample.update(summary)
            (out / f"{index}.postprocessed.private.json").write_text(
                json.dumps(
                    {"id": index, "code": candidate, "metadata": row["metadata"]}
                ),
                encoding="utf-8",
            )
        except Exception as error:
            if isinstance(error, HTTPError):
                (out / f"{index}.error.private.json").write_bytes(error.read())
            sample.update(
                status="REQUEST_OR_PROTOCOL_FAILED", error_type=type(error).__name__
            )
        receipt["samples"].append(sample)
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    require(
        digest(path) == DATA_SHA and digest(publisher) == SOURCE_SHA,
        "original inputs changed",
    )
    exit_code = profile_exit_code(receipt["samples"], expected_count=len(selected))
    receipt["status"] = "ORIGINAL_COMPLETION_PROFILE_NOT_FUNCTIONALLY_SCORED"
    receipt["exit_code"] = exit_code
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
