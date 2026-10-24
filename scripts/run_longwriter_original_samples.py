"""Generate from original LongWriter-6k inputs; not LongBench-Write scoring.

Reference assistant messages remain scorer-side. Use only the existing local NPU
endpoint; retain output/budget failures, with no prompt shortening or rewriting.
"""

import argparse
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REVISION = "0db15c0624f19d63e2efe1021595af933cc5b6cc"  # pragma: allowlist secret - public revision
DATA_SHA = "baac20a6fe9fc9d8ad600c8f1e98214f4b9590d810c9c9c875237d57b8342f52"  # pragma: allowlist secret - original public file checksum


def original_messages(row):
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) < 2
        or messages[-1].get("role") != "assistant"
    ):
        raise ValueError("original final assistant reference required")
    if any(
        not isinstance(message, dict)
        or message.get("role") not in {"system", "user", "assistant"}
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ValueError("unsupported original message representation")
    if messages[-2]["role"] != "user":
        raise ValueError("original generation input must end in user request")
    return messages[:-1]


def post(url, body):
    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=1800) as response:
        return json.load(response)


def original_indices(*, start=0, count=3, full=False):
    if full:
        if start != 0 or count != 3:
            raise ValueError("full selection cannot be combined with a range")
        return range(6000)
    if start < 0 or count < 1 or start + count > 6000:
        raise ValueError("range must stay within the original 6000 rows")
    return range(start, start + count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--full", action="store_true", help="All 6000 original corpus rows"
    )
    args = parser.parse_args()
    try:
        indices = original_indices(start=args.start, count=args.count, full=args.full)
    except ValueError as error:
        parser.error(str(error))
    endpoint = args.endpoint.rstrip("/")
    url = urlparse(endpoint)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost"}
        or url.path != "/v1"
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        parser.error("only local /v1 model endpoints permitted")
    path = args.data_root / "long.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("original LongWriter-6k file identity mismatch")
    with path.open(encoding="utf-8") as handle:
        originals = [
            json.loads(line) for line in islice(handle, indices.start, indices.stop)
        ]
    if len(originals) != len(indices):
        raise ValueError("original preselected rows missing; no replacement")
    from vllm_hust_workload_data.bindings import bind_source
    from vllm_hust_workload_data.consumer import resolve_input
    from vllm_hust_workload_data.pipeline import prepared_rows, run
    from vllm_hust_benchmark.workload_response_validation import check_response

    os.umask(0o077)
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    binding = bind_source(
        "LONGWRITER", data_root=args.data_root, files=[path.name], revision=REVISION
    )
    manifest = output / "binding.json"
    manifest.write_text(json.dumps(binding))
    run(
        {
            "sources": [
                {
                    "source_id": "LONGWRITER",
                    "data_root": str(args.data_root.resolve()),
                    "manifest": str(manifest),
                    "limit": indices.stop,
                }
            ]
        },
        output=output / "prepared",
    )
    database = output / "prepared/workload.sqlite"
    references = list(islice(prepared_rows(database), indices.start, indices.stop))
    if len(references) != len(indices):
        raise ValueError("prepared input count mismatch")
    receipt = {
        "source_id": "LONGWRITER",
        "revision": REVISION,
        "dataset_sha256": DATA_SHA,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "STARTED",
        "started_at_unix": time.time(),
        "selection": "ORIGINAL_TRAIN_RANGE_OR_FULL_FIXED_BEFORE_EXECUTION",
        "selected_row_indices": list(indices),
        "all_original_rows_requested": args.full,
        "model": args.model,
        "original_user_messages_unchanged": True,
        "reference_final_assistant_excluded": True,
        "generation_config": {"temperature": 0.5, "max_tokens": 32768},
        "configuration_scope": "declared generation integration budget, not an original API request or official benchmark protocol",
        "scoring": "NO_VALIDATED_SEMANTIC_OR_LENGTH_EVALUATOR",
        "quality_claim": False,
        "full_suite_claim": False,
        "samples": [],
    }
    target = output / "receipt.json"
    try:
        target.write_text(json.dumps(receipt, indent=2))
        for index, reference, original in zip(indices, references, originals):
            messages = original_messages(original)
            record = resolve_input(
                database, reference["record_id"], data_root=args.data_root
            )
            if record.input_record()["task"]["input"] != {
                "messages": messages
            } or record.outcome_record() != {
                "reference_message": original["messages"][-1]
            }:
                raise ValueError("original input/reference separation failed")
            body = {
                "model": args.model,
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 32768,
            }
            capacity = post(
                endpoint.removesuffix("/v1") + "/tokenize",
                {
                    "model": args.model,
                    "messages": messages,
                    "add_generation_prompt": True,
                },
            )
            if capacity["count"] + 32768 > 131072:
                raise ValueError(
                    "complete input and declared output budget do not fit; no truncation"
                )
            response = post(endpoint + "/chat/completions", body)
            (output / f"sample-{index}.private.json").write_text(json.dumps(response))
            receipt["samples"].append(
                {
                    "row_index": index,
                    "record_id": reference["record_id"],
                    "input_fields_roundtrip": True,
                    "full_prompt_tokens": response.get("usage", {}).get(
                        "prompt_tokens"
                    ),
                    "preflight_prompt_tokens": capacity["count"],
                    "completion_tokens": response.get("usage", {}).get(
                        "completion_tokens"
                    ),
                    "finish_reasons": [
                        choice.get("finish_reason")
                        for choice in response.get("choices", [])
                    ],
                    "response_validation": check_response(response, body),
                }
            )
            target.write_text(json.dumps(receipt, indent=2))
        if not all(
            sample["response_validation"]["status"] == "RESPONSE_INTEGRITY_VERIFIED"
            and sample["full_prompt_tokens"] == sample["preflight_prompt_tokens"]
            for sample in receipt["samples"]
        ):
            raise RuntimeError("source generation did not satisfy integrity checks")
        receipt["status"] = (
            "ORIGINAL_MESSAGES_GENERATED_AND_INTEGRITY_VERIFIED_NO_QUALITY_SCORE"
        )
    except BaseException as error:
        receipt.update(status="EXECUTION_ERROR", error_type=type(error).__name__)
        raise
    finally:
        receipt["finished_at_unix"] = time.time()
        target.write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
