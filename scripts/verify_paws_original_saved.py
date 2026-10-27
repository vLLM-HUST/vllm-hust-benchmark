"""Offline PAWS original-pair, saved-wire-output and classification-score audit.

No model/judge call, prediction repair or rewriting of the original run.
"""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_paws_original_assets import PINS
from prepare_xsum_original_assets import verify as verify_file
from run_paws_original import INSTRUCTION, prediction, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {"source_id": "PAWS", "status": "PLAN_ONLY", "new_model_calls": 0}
            )
        )
        return
    output = args.output.resolve()
    for original in (
        args.run.resolve(strict=True),
        args.data_root.resolve(strict=True),
    ):
        if (
            output == original
            or output.is_relative_to(original)
            or original.is_relative_to(output)
        ):
            raise ValueError("Audit output must be separate from original data and run")
    if output.exists():
        raise ValueError("Fresh audit output required")
    receipt = json.loads(
        (args.run / "receipt-private.json").read_text(encoding="utf-8")
    )
    if (
        receipt["status"]
        != "ORIGINAL_SELECTED_PAIRS_CLASSIFIED_AND_SCORED_NOT_SOURCE_DELIVERY"
    ):
        raise ValueError("Incomplete native run; do not replace missing tasks")
    pins = json.loads(PINS.read_text(encoding="utf-8"))["dataset"]
    if (
        receipt["dataset_revision"] != pins["revision"]
        or receipt["max_tokens"] != 16
        or receipt["temperature"] != 0
    ):
        raise ValueError("Original source or declared model profile changed")
    for asset in pins["files"]:
        verify_file(args.data_root / asset["path"], asset)
    import pyarrow.parquet as pq

    records = [
        json.loads(line)
        for line in (args.run / "completed-private.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    indexed = {(record["file_index"], record["position"]): record for record in records}
    if (
        len(indexed) != len(records)
        or len(records) != receipt["completed_tasks"]
        or len(records) != receipt["selected_tasks"]
    ):
        raise ValueError("Duplicate or missing original native task")
    checked = []
    for file_index, asset in enumerate(pins["files"]):
        path = asset["path"]
        original = pq.ParquetFile(args.data_root / path)
        full = receipt["full_all_rows"] or (
            receipt["full_test"] and path.startswith("labeled_final/test-")
        )
        count = original.metadata.num_rows if full else 3
        position = 0
        for batch in original.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                if position >= count:
                    break
                record = indexed.get((file_index, position))
                if (
                    record is None
                    or record["file"] != path
                    or record["id"] != row["id"]
                    or record["label"] != row["label"]
                ):
                    raise ValueError("Original task ID/order/reference mismatch")
                folder = args.run / f"file-{file_index}"
                raw = (folder / f"row-{position}.request-private.json").read_bytes()
                response_raw = (
                    folder / f"row-{position}.response-private.json"
                ).read_bytes()
                if (
                    hashlib.sha256(raw).hexdigest() != record["request_sha256"]
                    or hashlib.sha256(response_raw).hexdigest()
                    != record["response_sha256"]
                ):
                    raise ValueError("Saved native request/response changed")
                body = json.loads(raw)
                message = body["messages"]
                if (
                    len(message) != 1
                    or message[0]["role"] != "user"
                    or not message[0]["content"].startswith(INSTRUCTION)
                ):
                    raise ValueError("Declared zero-shot input adapter changed")
                pair = json.loads(message[0]["content"][len(INSTRUCTION) :])
                if pair != {
                    "sentence1": row["sentence1"],
                    "sentence2": row["sentence2"],
                }:
                    raise ValueError(
                        "Sentence rewritten or reference/ID leaked into model input"
                    )
                expected = {
                    "model": receipt["model"],
                    "messages": message,
                    "max_tokens": 16,
                    "temperature": 0,
                }
                if not receipt["enable_thinking"]:
                    expected["chat_template_kwargs"] = {"enable_thinking": False}
                if body != expected:
                    raise ValueError("Unrecorded native request options")
                response = json.loads(response_raw)
                if (
                    prediction(response) != record["prediction"]
                    or response["usage"] != record["usage"]
                    or response["choices"][0]["finish_reason"]
                    != record["finish_reason"]
                ):
                    raise ValueError("Saved prediction/finish/usage mismatch")
                checked.append(record)
                position += 1
            if position >= count:
                break
        if position != count:
            raise ValueError("Missing original selected rows")
    if len(checked) != len(records) or score(checked) != receipt["scores"]:
        raise ValueError("Original classification score or coverage mismatch")
    for asset in pins["files"]:
        verify_file(args.data_root / asset["path"], asset)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    result = {
        "source_id": "PAWS",
        "status": "ORIGINAL_PAIRS_REFERENCES_SAVED_RESPONSES_AND_SCORES_VERIFIED",
        "tasks": len(checked),
        "scores": receipt["scores"],
        "new_model_calls": 0,
        "paid_judge_calls": 0,
        "original_run_modified": False,
        "source_checkbox_eligible": False,
        "model_adapter": "declared_zero_shot_not_author_chat_prompt_or_paper_reproduction",
    }
    (output / "receipt-public.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
