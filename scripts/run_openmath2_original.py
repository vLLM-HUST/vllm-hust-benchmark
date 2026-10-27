"""Generate from unchanged published corpus problems; not an official quality benchmark.

Default is metadata-only plan. Explicit operator model/budget are required to run.
Stored teacher solutions/expected answers never enter the model request.
"""

import argparse
import json
from pathlib import Path

from run_historical_dialogue_original import has_generated_content
from run_oasst_original_continuation import post, validate_endpoint


def execute(
    records,
    *,
    output,
    endpoint,
    model,
    max_tokens,
    context,
    transport=post,
    expected_count=None,
    final_verify=None,
):
    endpoint = validate_endpoint(endpoint)
    if not model or type(max_tokens) is not int or not 0 < max_tokens < context:
        raise ValueError("explicit model, budget and context required")
    if expected_count is not None:
        records = list(records)
        if not records or len(records) != expected_count:
            raise ValueError("preselected original record count differs")

    def samples():
        seen_files = set()
        previous = None
        previous_index = -1
        for record in records:
            name = record.source["file"]
            index = record.replay["source_row_index"]
            if name != previous:
                if name in seen_files or index != 0:
                    raise ValueError(
                        "original files must appear once, starting at row0"
                    )
                seen_files.add(name)
                previous, previous_index = name, -1
            if (
                record.source["source_id"] != "OPENMATH2"
                or set(record.task["input"]) != {"problem"}
                or not isinstance(record.task["input"]["problem"], str)
                or not record.task["input"]["problem"]
                or type(index) is not int
                or index != previous_index + 1
                or record.task["task_id"] != f"{record.source['file_sha256']}:{index}"
            ):
                raise ValueError(
                    "unique unchanged original OPENMATH2 problems required"
                )
            previous_index = index
            yield {
                "source": record.source,
                "task_id": record.task["task_id"],
                "row_index": record.replay["source_row_index"],
                "request": {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": record.task["input"]["problem"]}
                    ],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "n": 1,
                },
                "original_reference": record.outcome,
                "original_annotations": record.replay["annotations"],
            }

    selected = list(samples()) if expected_count is not None else samples()
    output = Path(output)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    receipt = {
        "source_id": "OPENMATH2",
        "status": "STARTED",
        "model": model,
        "max_tokens": max_tokens,
        "context_capacity": context,
        "profile": "UNCHANGED_PUBLISHED_PROBLEM_AS_USER_INPUT_EXPLICIT_OPERATOR_BUDGET",
        "original_api_request_capture": False,
        "author_evaluation_reproduction": False,
        "quality_scored": False,
        "source_checkbox_eligible": False,
        "generation_attempts": 0,
        "completed_count": 0,
        "stop_count": 0,
        "length_count": 0,
        "boundary": "Published synthetic training corpus generation, not independent task scoring, teacher reproduction or performance. No problem rewriting, answer leakage, retry or replacement.",
    }

    def save(name, value):
        path = output / name
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        path.chmod(0o600)

    save(
        "selection-private.json",
        selected
        if expected_count is not None
        else {
            "selection": "ALL_ORIGINAL_ROWS_IN_EXPLICIT_SELECTED_FILES_STREAMED_NO_OUTCOME_FILTERING",
            "expected_count": None,
        },
    )
    try:
        for number, sample in enumerate(selected):
            attempt = {**sample, "status": "PREPARED"}
            name = f"request-{number:06d}.private.json"
            save(name, attempt)
            try:
                tokenized = transport(
                    endpoint.removesuffix("/v1") + "/tokenize",
                    {
                        "model": model,
                        "messages": sample["request"]["messages"],
                        "add_generation_prompt": True,
                    },
                )
                attempt["tokenize"] = tokenized
                count = tokenized.get("count")
                if type(count) is not int or count <= 0 or count + max_tokens > context:
                    raise ValueError(
                        "original problem plus budget exceeds declared context"
                    )
                receipt["generation_attempts"] += 1
                save("receipt-private.json", receipt)
                response = transport(endpoint + "/chat/completions", sample["request"])
                attempt["response"] = response
                save(name, attempt)
                choices = response.get("choices", [])
                usage = response.get("usage", {})
                if (
                    len(choices) != 1
                    or choices[0].get("finish_reason") not in {"stop", "length"}
                    or not has_generated_content(choices[0])
                    or usage.get("prompt_tokens") != count
                    or type(usage.get("completion_tokens")) is not int
                    or not 0 < usage["completion_tokens"] <= max_tokens
                ):
                    raise ValueError("model response integrity not verified")
                attempt["status"] = (
                    "RESPONSE_INTEGRITY_VERIFIED"
                    if choices[0]["finish_reason"] == "stop"
                    else "BUDGET_EXHAUSTED_QUALITY_UNVERIFIED"
                )
                receipt["completed_count"] += 1
                receipt[choices[0]["finish_reason"] + "_count"] += 1
            except Exception as error:
                attempt.update(
                    status="FAILED_NOT_REPLACED", error_type=type(error).__name__
                )
                raise
            finally:
                save(name, attempt)
        if not receipt["completed_count"] or (
            expected_count is not None and receipt["completed_count"] != expected_count
        ):
            raise ValueError("original completion count differs")
        if final_verify is not None:
            final_verify()
        receipt["status"] = "ORIGINAL_CORPUS_PROBLEMS_GENERATED_NO_INDEPENDENT_SCORE"
    except Exception as error:
        receipt.update(status="FAILED_NOT_COMPLETE", error_type=type(error).__name__)
        raise
    finally:
        save("receipt-private.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan", "run"], default="plan")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Original first count rows per selected file; no outcome-based selection",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="All original rows in the explicitly selected files",
    )
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--context", type=int)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    args = parser.parse_args()
    from vllm_hust_workload_data.registry import source_spec
    from vllm_hust_workload_data import iter_workload
    from vllm_hust_workload_data.assets import verify_file

    spec = source_spec("OPENMATH2")
    files = {row["path"]: row for row in spec["files"]}
    if (
        len(set(args.files)) != len(args.files)
        or any(name not in files for name in args.files)
        or args.count < 3
    ):
        parser.error("distinct original registered shards and count>=3 required")
    if args.mode == "plan":
        print(
            json.dumps(
                {
                    "source_id": "OPENMATH2",
                    "status": "PLAN_ONLY",
                    "files": args.files,
                    "original_rows_per_file": None if args.full else args.count,
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                    "scope": "Corpus workload generation, not independent quality evaluation; no files read or model requests",
                },
                indent=2,
            )
        )
        return
    if any(
        value is None
        for value in [
            args.data_root,
            args.output,
            args.model,
            args.max_tokens,
            args.context,
        ]
    ):
        parser.error(
            "run requires external data/output, model and explicit max-tokens/context"
        )
    data = args.data_root.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(data) or data.is_relative_to(output):
        parser.error("private output must be separate from original data")

    def records():
        for name in args.files:
            yield from iter_workload(
                "OPENMATH2",
                data_root=data,
                files=[name],
                limit=None if args.full else args.count,
            )

    def reverify():
        for name in args.files:
            verify_file(data, files[name])

    receipt = execute(
        records(),
        output=output,
        endpoint=args.endpoint,
        model=args.model,
        max_tokens=args.max_tokens,
        context=args.context,
        expected_count=None if args.full else len(args.files) * args.count,
        final_verify=reverify,
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in [
                    "source_id",
                    "status",
                    "generation_attempts",
                    "quality_scored",
                    "source_checkbox_eligible",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
