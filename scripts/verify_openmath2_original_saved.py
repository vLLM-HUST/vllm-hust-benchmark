"""Verify saved original corpus requests/references/responses without model calls.

This is transport/input integrity, not independent task-quality scoring.
"""

import argparse
import hashlib
import json
from pathlib import Path

from run_historical_dialogue_original import has_generated_content
from run_openmath2_original import corpus_input_field


def verify(saved, records, *, expected_count=None, source_id="OPENMATH2"):
    input_field = corpus_input_field(source_id)
    saved = Path(saved).resolve(strict=True)
    receipt = json.loads((saved / "receipt-private.json").read_text("utf-8"))
    if (
        receipt.get("source_id") != source_id
        or receipt.get("status")
        != "ORIGINAL_CORPUS_PROBLEMS_GENERATED_NO_INDEPENDENT_SCORE"
        or receipt.get("quality_scored") is not False
        or receipt.get("author_evaluation_reproduction") is not False
    ):
        raise ValueError("complete original corpus generation receipt required")
    count = stops = lengths = 0
    digests = []
    for number, record in enumerate(records):
        path = saved / f"request-{number:06d}.private.json"
        raw = path.read_bytes()
        case = json.loads(raw)
        request = {
            "model": receipt["model"],
            "messages": [
                {"role": "user", "content": record.task["input"][input_field]}
            ],
            "temperature": 0,
            "max_tokens": receipt["max_tokens"],
            "n": 1,
        }
        if (
            record.source["source_id"] != source_id
            or set(record.task["input"]) != {input_field}
            or case["source"] != record.source
            or case["task_id"] != record.task["task_id"]
            or case["row_index"] != record.replay["source_row_index"]
            or case["request"] != request
            or case["original_reference"] != record.outcome
            or case["original_annotations"] != record.replay["annotations"]
        ):
            raise ValueError("original input/reference/source identity mismatch")
        response = case["response"]
        choices = response.get("choices", [])
        usage = response.get("usage", {})
        tokens = case["tokenize"].get("count")
        if (
            type(tokens) is not int
            or tokens <= 0
            or tokens + receipt["max_tokens"] > receipt["context_capacity"]
            or len(choices) != 1
            or choices[0].get("finish_reason") not in {"stop", "length"}
            or not has_generated_content(choices[0])
            or usage.get("prompt_tokens") != tokens
            or type(usage.get("completion_tokens")) is not int
            or not 0 < usage["completion_tokens"] <= receipt["max_tokens"]
        ):
            raise ValueError("saved response integrity mismatch")
        expected_status = (
            "RESPONSE_INTEGRITY_VERIFIED"
            if choices[0]["finish_reason"] == "stop"
            else "BUDGET_EXHAUSTED_QUALITY_UNVERIFIED"
        )
        if case["status"] != expected_status:
            raise ValueError("original stop/length state changed")
        count += 1
        stops += int(choices[0]["finish_reason"] == "stop")
        lengths += int(choices[0]["finish_reason"] == "length")
        if expected_count is not None:
            digests.append(
                {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
            )
    case_files = sum(
        1 for path in saved.glob("request-*.private.json") if path.is_file()
    )
    if (
        not count
        or (expected_count is not None and count != expected_count)
        or case_files != count
        or receipt["completed_count"] != count
        or receipt["generation_attempts"] != count
        or receipt["stop_count"] != stops
        or receipt["length_count"] != lengths
    ):
        raise ValueError("original selected cases or completion counters differ")
    return {
        "source_id": source_id,
        "status": "SAVED_ORIGINAL_INPUT_REFERENCE_AND_RESPONSE_INTEGRITY_VERIFIED_NO_MODEL_CALLS",
        "original_cases": count,
        "stop_count": stops,
        "length_count": lengths,
        "model_calls": 0,
        "quality_scored": False,
        "source_checkbox_eligible": False,
        "saved_case_digests": digests,
        "boundary": "Original published problem field values/typed references, source IDs and saved API response integrity only. Not original API capture, task correctness, author training/evaluation reproduction, model replay, whole corpus coverage or performance.",
    }


def main(source_id="OPENMATH2"):
    corpus_input_field(source_id)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.count < 3 or len(set(args.files)) != len(args.files):
        parser.error("distinct explicit original files and count>=3 required")
    if args.output:
        output = args.output.resolve()
        if output.exists() or any(
            output.is_relative_to(root.resolve())
            for root in (args.saved, args.data_root)
        ):
            parser.error(
                "fresh output outside original data and saved evidence required"
            )
    from vllm_hust_workload_data import iter_workload

    def records():
        for name in args.files:
            yield from iter_workload(
                source_id,
                data_root=args.data_root,
                files=[name],
                limit=None if args.full else args.count,
            )

    result = verify(
        args.saved,
        records(),
        expected_count=None if args.full else len(args.files) * args.count,
        source_id=source_id,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "saved_case_digests"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
