"""Pinned RULER generated rows -> original client -> original metric.

Transport-only binding for a local completion service; not the unchanged NeMo CLI.
Run the publisher's data/prepare.py first. Outputs are private and never overwritten.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SOURCE_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"  # pragma: allowlist secret - publisher commit
SOURCE_SHA256 = {
    "scripts/pred/client_wrappers.py": "4530a6b53f451d1af8f29f6ee794fca789feffa92b2e838af2c67a4568244b2c",  # pragma: allowlist secret - publisher file hash
    "scripts/eval/evaluate.py": "b9a5fcbded7209663d97670496c74dbcd6358ddbe3d9887a0c5f0c4f974bfe1c",  # pragma: allowlist secret - publisher file hash
    "scripts/eval/synthetic/constants.py": "6740467c17b8dc06b6b30f4f97e54ce8de81db0dd879f1538d0b6b5727f4bd5f",  # pragma: allowlist secret - publisher file hash
    "scripts/data/synthetic/constants.py": "6296e901d495ec6200dc3f68993ea13d8282e3c0dbe1a8c47967f111105d1fde",  # pragma: allowlist secret - publisher file hash
    "scripts/synthetic.yaml": "34bc71dcacdc41a829a170f04b528fbf48d62c616338005ab4991680fbf8cb0b",  # pragma: allowlist secret - publisher file hash
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
        raise ValueError("uncredentialed loopback /v1 endpoint required")
    return value.rstrip("/")


def load_module(source, relative, name):
    spec = importlib.util.spec_from_file_location(name, source / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_components(source, task):
    # Check before importing any publisher Python, not after execution.
    for relative, expected in SOURCE_SHA256.items():
        path = source / relative
        if (
            path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"modified or mismatched publisher file: {relative}")
    import yaml

    config = yaml.safe_load((source / "scripts/synthetic.yaml").read_text())[task]
    family = config["task"]
    data = load_module(source, "scripts/data/synthetic/constants.py", "ruler_data")
    metrics = load_module(source, "scripts/eval/synthetic/constants.py", "ruler_metric")
    clients = load_module(source, "scripts/pred/client_wrappers.py", "ruler_client")
    tree = ast.parse((source / "scripts/eval/evaluate.py").read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "postprocess_pred"
    ]
    if len(nodes) != 1:
        raise ValueError("publisher postprocess function not unique")
    namespace = {"re": re}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            "<pinned-ruler-postprocess>",
            "exec",
        ),
        namespace,
    )
    return (
        clients.VLLMClient,
        data.TASKS[family]["tokens_to_generate"],
        namespace["postprocess_pred"],
        metrics.TASKS[family]["metric_fn"],
    )


def read_selection(path, expected_sha256, offsets):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("generated input digest mismatch")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if offsets is None:
        offsets = list(range(len(rows)))
    if (
        not offsets
        or any(type(i) is not int or i < 0 or i >= len(rows) for i in offsets)
        or len(set(offsets)) != len(offsets)
    ):
        raise ValueError("nonempty unique physical row offsets required")
    for row in rows:
        if (
            not isinstance(row.get("input"), str)
            or "index" not in row
            or not isinstance(row.get("answer_prefix", ""), str)
            or not isinstance(row.get("outputs"), list)
            or not row["outputs"]
            or any(not isinstance(v, str) for v in row["outputs"])
        ):
            raise ValueError("invalid original generated row")
    # Original index can repeat: never deduplicate or rewrite it.
    return rows, offsets


def completion_request(body, prompt, budget, model):
    expected = {
        "prompt": prompt,
        "max_tokens": budget,
        "temperature": 0.0,
        "top_k": 32,
        "top_p": 1.0,
        "stop": [],
    }
    if body != expected or type(body.get("max_tokens")) is not int:
        raise ValueError("original request or budget changed")
    return {**body, "model": model, "stream": False}


def suffix_response(result, input_tokens, budget):
    choices, usage = result.get("choices"), result.get("usage", {})
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("one completion required")
    choice = choices[0]
    if (
        not isinstance(choice.get("text"), str)
        or choice.get("finish_reason") not in {"stop", "length"}
        or type(usage.get("prompt_tokens")) is not int
        or usage["prompt_tokens"] != input_tokens
        or type(usage.get("completion_tokens")) is not int
        or not 0 <= usage["completion_tokens"] <= budget
    ):
        raise ValueError("invalid completion or token accounting")
    # Unlike some legacy clients, RULER expects the suffix ONLY. Prefix leaks gold.
    return {"text": [choice["text"]]}


def post(url, body):
    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=3600) as response:
        return json.load(response)


def run(args):
    api = endpoint(args.api_base)
    source = args.source.resolve()
    client_type, budget, postprocess, metric = original_components(source, args.task)
    offsets = None if args.rows is None else [int(v) for v in args.rows.split(",")]
    rows, offsets = read_selection(args.data, args.data_sha256, offsets)
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    receipt = {
        "source_id": "RULER",
        "source_commit": SOURCE_COMMIT,
        "task": args.task,
        "data_sha256": args.data_sha256,
        "selected_row_offsets": offsets,
        "original_output_budget": budget,
        "model": args.model,
        "status": "STARTED",
        "completed_rows": 0,
        "samples": [],
        "source_checkbox_eligible": False,
        "full_original_prediction_cli_claim": False,
        "binding": "original VLLMClient and scoring; stdlib JSONL replaces NeMo reader; local completion transport only",
        "source_index_field_rewritten": False,
        "performance_or_full_ruler_claim": False,
    }
    target = args.output / "receipt.json"

    def save():
        temp = args.output / "receipt.json.tmp"
        temp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temp.replace(target)

    client = client_type(
        server_host="127.0.0.1",
        server_port="0",
        tokens_to_generate=budget,
        temperature=0.0,
        top_p=1.0,
        top_k=32,
        random_seed=0,
        stop=[],
    )
    current = [None]

    def transport(body):
        offset, row, prompt = current[0]
        request = completion_request(body, prompt, budget, args.model)
        tokens = post(
            api.removesuffix("/v1") + "/tokenize",
            {"model": args.model, "prompt": prompt},
        )["count"]
        if (
            type(tokens) is not int
            or tokens <= 0
            or tokens + budget > args.context_window
        ):
            raise ValueError(
                "full original input and output exceed deployment capacity; no truncation"
            )
        record = {
            "original_generated_row": row,
            "native_request": request,
            "input_tokens": tokens,
        }
        path = args.output / f"row-{offset}.private.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        result = post(api + "/completions", request)
        record["native_response"] = result
        path.write_text(json.dumps(record), encoding="utf-8")
        response = suffix_response(result, tokens, budget)
        receipt["samples"].append(
            {
                "original_row_offset": offset,
                "private_record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        return response

    client._send_request = transport
    predictions = []
    save()
    try:
        with (args.output / "predictions.private.jsonl").open(
            "x", encoding="utf-8"
        ) as output:
            for offset in offsets:
                row = rows[offset]
                prompt = row["input"] + row.get("answer_prefix", "")
                current[0] = offset, row, prompt
                prediction = {
                    "index": row["index"],
                    "pred": client(prompt)["text"][0],
                    "input": prompt,
                    "outputs": row["outputs"],
                    "others": row.get("others", {}),
                    "truncation": row.get("truncation", -1),
                    "length": row.get("length", -1),
                }
                output.write(json.dumps(prediction) + "\n")
                output.flush()
                predictions.append(prediction)
                receipt["completed_rows"] = len(predictions)
                save()
        receipt["original_metric_score"] = metric(
            [postprocess(p["pred"], {}) for p in predictions],
            [p["outputs"] for p in predictions],
        )
        receipt["status"] = "ORIGINAL_PROFILE_COMPLETED_REQUIRES_AUDIT"
    except BaseException as error:
        receipt.update(status="EXECUTION_ERROR", error_type=type(error).__name__)
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--rows", help="predeclared physical offsets, comma-separated; default all rows"
    )
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-window", type=int, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="fresh private directory"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
