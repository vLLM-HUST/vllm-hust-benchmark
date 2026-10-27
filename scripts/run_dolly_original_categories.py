"""Original Dolly category coverage, not an official quality benchmark.

Reuse the pinned publisher-template integration. Data, requests and outputs stay
outside the checkout. Existing responses can be audited without model calls.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]


def original_runner():
    spec = importlib.util.spec_from_file_location(
        "dolly_original", Path(__file__).with_name("run_dolly_original_samples.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_indices(rows, *, full=False):
    categories = defaultdict(list)
    for index, row in enumerate(rows):
        category = row.get("category")
        if not isinstance(category, str) or not category.strip():
            raise ValueError("original category required")
        if full or len(categories[category]) < 3:
            categories[category].append(index)
    return sorted(index for group in categories.values() for index in group), dict(
        categories
    )


def reusable_indices(receipt, *, runner, model):
    if (
        receipt.get("source_id") != "DOLLY"
        or receipt.get("revision") != runner.REVISION
        or receipt.get("dataset_sha256") != runner.DATA_SHA
        or receipt.get("publisher_constants_sha256") != runner.CONSTS_SHA
        or receipt.get("model") != model
        or receipt.get("generation_config") != {"temperature": 0, "max_tokens": 2048}
        or receipt.get("status")
        != "ORIGINAL_FIELDS_GENERATED_AND_INTEGRITY_VERIFIED_NO_QUALITY_SCORE"
    ):
        raise ValueError(
            "reuse requires the same source, template, model and generation protocol"
        )
    result = set()
    for sample in receipt["samples"]:
        index = sample["row_index"]
        if (
            type(index) is not int
            or index < 0
            or index in result
            or sample["full_prompt_tokens"] != sample["preflight_prompt_tokens"]
            or sample["finish_reasons"] != ["stop"]
            or sample["response_validation"]["status"] != "RESPONSE_INTEGRITY_VERIFIED"
            or not sample["input_fields_roundtrip"]
        ):
            raise ValueError("reuse evidence is incomplete or ambiguous")
        result.add(index)
    if sorted(result) != sorted(receipt["selected_row_indices"]):
        raise ValueError("reuse indices disagree with samples")
    return result


def external(path):
    path = Path(path).resolve()
    if path == ROOT or path.is_relative_to(ROOT):
        raise ValueError(
            "datasets and private results must remain outside the checkout"
        )
    return path


def validate_endpoint(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("only a local credential-free /v1 endpoint is permitted")
    return value.rstrip("/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--publisher-constants", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify-existing-run", type=Path)
    parser.add_argument("--reuse-receipt", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument("--max-model-len", type=int, required=True)
    args = parser.parse_args()
    endpoint = validate_endpoint(args.endpoint)
    data_root = external(args.data_root)
    if args.max_model_len <= 2048:
        parser.error("declared deployment context must exceed generation budget")
    runner = original_runner()
    data = data_root / "databricks-dolly-15k.jsonl"
    if hashlib.sha256(data.read_bytes()).hexdigest() != runner.DATA_SHA:
        raise ValueError("original dataset SHA mismatch")
    if (
        hashlib.sha256(args.publisher_constants.read_bytes()).hexdigest()
        != runner.CONSTS_SHA
    ):
        raise ValueError("publisher constants SHA mismatch")
    spec = importlib.util.spec_from_file_location(
        "publisher_constants", args.publisher_constants
    )
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    rows = [
        json.loads(line)
        for line in data.read_text("utf-8").splitlines()
        if line.strip()
    ]
    selected, categories = selected_indices(rows, full=args.full)
    reused = (
        reusable_indices(
            json.loads(args.reuse_receipt.read_text()), runner=runner, model=args.model
        )
        if args.reuse_receipt
        else set()
    )
    pending = [index for index in selected if index not in reused]
    from vllm_hust_workload_data import iter_workload
    from vllm_hust_benchmark.workload_response_validation import check_response

    records = {
        index: record
        for index, record in enumerate(iter_workload("DOLLY", data_root=data_root))
        if index in pending
    }
    output = external(args.output or args.verify_existing_run)
    verify = args.verify_existing_run is not None
    if verify:
        frozen = json.loads((output / "selection.private.json").read_text())
        terminal = json.loads((output / "terminal.private.json").read_text())
        if (
            frozen["new_indices"] != pending
            or frozen["categories"] != categories
            or frozen["dataset_sha256"] != runner.DATA_SHA
            or terminal["returncode"] != 0
        ):
            raise ValueError(
                "existing run has a different selection or non-successful terminal state"
            )
    else:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        (output / "selection.private.json").write_text(
            json.dumps(
                {
                    "selected_at_unix": time.time(),
                    "categories": categories,
                    "new_indices": pending,
                    "dataset_sha256": runner.DATA_SHA,
                    "reused_indices": sorted(set(selected) & reused),
                },
                indent=2,
            )
        )
    completed = []
    for index in pending:
        fields = records[index].task["input"]
        if fields != {key: rows[index][key] for key in ["instruction", "context"]}:
            raise ValueError("provider inputs differ from original fields")
        body = {
            "model": args.model,
            "messages": [
                {"role": "user", "content": runner.generation_prefix(fields, constants)}
            ],
            "temperature": 0,
            "max_tokens": 2048,
        }
        if verify:
            saved = json.loads((output / f"sample-{index}.private.json").read_text())
            if saved["request"] != body:
                raise ValueError(
                    "saved request differs from the original template and protocol"
                )
            response = saved["response"]
        else:
            count = runner.post(
                endpoint.removesuffix("/v1") + "/tokenize",
                {
                    "model": args.model,
                    "messages": body["messages"],
                    "add_generation_prompt": True,
                },
            )["count"]
            if count + 2048 > args.max_model_len:
                raise ValueError("full prompt exceeds declared context; no truncation")
            response = runner.post(endpoint + "/chat/completions", body)
            (output / f"sample-{index}.private.json").write_text(
                json.dumps({"request": body, "response": response})
            )
            if response["usage"]["prompt_tokens"] != count:
                raise ValueError(
                    "server prompt count differs from full-input preflight"
                )
        if check_response(response, body)["status"] != "RESPONSE_INTEGRITY_VERIFIED":
            raise ValueError("original response integrity failed; do not replace task")
        completed.append(index)
    receipt = {
        "source_id": "DOLLY",
        "status": "ORIGINAL_CATEGORY_INPUTS_AND_RESPONSES_VERIFIED",
        "categories": categories,
        "new_original_indices_verified": completed,
        "prior_original_indices_reused": sorted(set(selected) & reused),
        "dataset_sha256": runner.DATA_SHA,
        "model_calls_this_command": 0 if verify else len(completed),
        "quality_score": "NOT_APPLICABLE_REQUEST_CORPUS",
        "raw_data_published": False,
        "full_dataset": args.full,
    }
    if not verify:
        (output / "terminal.private.json").write_text(
            json.dumps({"returncode": 0, "finished_at_unix": time.time()})
        )
    (output / "public-entry-audit.private.json").write_text(
        json.dumps(receipt, indent=2)
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "verified_original_requests": len(completed),
                "reused_original_requests": len(set(selected) & reused),
                "model_calls_this_command": receipt["model_calls_this_command"],
            }
        )
    )


if __name__ == "__main__":
    main()
