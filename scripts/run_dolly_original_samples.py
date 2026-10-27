"""Original Dolly instruction/context integration, not official quality scoring.

Use the publisher's fixed training-prompt prefix before its response slot, with
the original fields unchanged. The context-bearing prefix is an explicit serving
integration choice, not an upstream evaluation protocol. Outputs are private.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DATA_SHA = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"  # pragma: allowlist secret - original public dataset checksum
CONSTS_SHA = "f911820bab540af71a904be77600fc03755153b63d7fe5755bb096e8984041a5"  # pragma: allowlist secret - publisher source checksum
REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"  # pragma: allowlist secret - public revision
PUBLISHER_REVISION = "e2d664ddf06e0723d818dc56d6c522ead4bd881d"  # pragma: allowlist secret - public revision


def generation_prefix(row, constants):
    """Preserve publisher fields verbatim; never read the reference response."""
    if not isinstance(row.get("instruction"), str) or not isinstance(
        row.get("context"), str
    ):
        raise ValueError("original instruction/context strings required")
    template = (
        constants.PROMPT_WITH_INPUT_FORMAT
        if row["context"]
        else constants.PROMPT_NO_INPUT_FORMAT
    )
    if template.count("{response}") != 1:
        raise ValueError("publisher response-slot boundary changed")
    prefix = template.split("{response}", 1)[0]
    return prefix.format(instruction=row["instruction"], input=row["context"])


def post(url, payload):
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--publisher-constants", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.path != "/v1"
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        parser.error("only local /v1 model endpoints permitted")
    data = args.data_root / "databricks-dolly-15k.jsonl"
    if hashlib.sha256(data.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("original dataset identity mismatch")
    if hashlib.sha256(args.publisher_constants.read_bytes()).hexdigest() != CONSTS_SHA:
        raise ValueError("fixed publisher constants identity mismatch")
    spec = importlib.util.spec_from_file_location(
        "dolly_publisher_constants", args.publisher_constants
    )
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    from vllm_hust_workload_data.bindings import bind_source
    from vllm_hust_workload_data.consumer import resolve_input
    from vllm_hust_workload_data.pipeline import prepared_rows, run
    from vllm_hust_benchmark.workload_response_validation import check_response

    os.umask(0o077)
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    binding = bind_source(
        "DOLLY", data_root=args.data_root, files=[data.name], revision=REVISION
    )
    manifest = output / "binding.json"
    manifest.write_text(json.dumps(binding))
    run(
        {
            "sources": [
                {
                    "source_id": "DOLLY",
                    "data_root": str(args.data_root.resolve()),
                    "manifest": str(manifest),
                    "limit": 3,
                }
            ]
        },
        output=output / "prepared",
    )
    database = output / "prepared/workload.sqlite"
    references = list(prepared_rows(database))
    originals = [json.loads(line) for line in data.read_text().splitlines()[:3]]
    if len(references) != len(originals) or len(originals) != 3:
        raise ValueError("three preselected original rows required")
    receipt = {
        "source_id": "DOLLY",
        "revision": REVISION,
        "publisher_revision": PUBLISHER_REVISION,
        "publisher_constants_sha256": CONSTS_SHA,
        "dataset_sha256": DATA_SHA,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "STARTED",
        "started_at_unix": time.time(),
        "selection": "FIRST_3_ORIGINAL_TRAIN_ROWS_BEFORE_EXECUTION",
        "selected_row_indices": [0, 1, 2],
        "model": args.model,
        "rendering": "publisher training prefix before response slot; original instruction/context unchanged; chat transport",
        "generation_config": {"temperature": 0, "max_tokens": 2048},
        "scoring": "NO_PUBLISHER_EVALUATOR_OR_INDEPENDENT_SEMANTIC_ORACLE",
        "quality_claim": False,
        "full_suite_claim": False,
        "samples": [],
    }
    target = output / "receipt.json"
    target.write_text(json.dumps(receipt, indent=2))
    try:
        for index, (reference, original) in enumerate(zip(references, originals)):
            record = resolve_input(
                database, reference["record_id"], data_root=args.data_root
            )
            fields = record.input_record()["task"]["input"]
            if fields != {key: original[key] for key in ("instruction", "context")}:
                raise ValueError(
                    "provider fields differ from original input or contain extra fields"
                )
            prompt = generation_prefix(fields, constants)
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 2048,
            }
            capacity = post(
                endpoint.removesuffix("/v1") + "/tokenize",
                {
                    "model": args.model,
                    "messages": body["messages"],
                    "add_generation_prompt": True,
                },
            )
            if capacity["count"] + 2048 > 131072:
                raise ValueError(
                    "full original input exceeds deployment capacity; no truncation"
                )
            response = post(endpoint + "/chat/completions", body)
            (output / f"sample-{index}.private.json").write_text(json.dumps(response))
            validation = check_response(response, body)
            receipt["samples"].append(
                {
                    "row_index": index,
                    "record_id": reference["record_id"],
                    "category": original["category"],
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
                    "response_validation": validation,
                }
            )
            target.write_text(json.dumps(receipt, indent=2))
        if all(
            row["response_validation"]["status"] == "RESPONSE_INTEGRITY_VERIFIED"
            and row["full_prompt_tokens"] == row["preflight_prompt_tokens"]
            for row in receipt["samples"]
        ):
            receipt["status"] = (
                "ORIGINAL_FIELDS_GENERATED_AND_INTEGRITY_VERIFIED_NO_QUALITY_SCORE"
            )
        else:
            receipt["status"] = "OUTPUT_INTEGRITY_OR_CAPACITY_NOT_VERIFIED"
            raise RuntimeError(
                "generation did not satisfy input/response integrity checks"
            )
    except BaseException as error:
        receipt.update(status="EXECUTION_ERROR", error_type=type(error).__name__)
        raise
    finally:
        receipt["finished_at_unix"] = time.time()
        target.write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
