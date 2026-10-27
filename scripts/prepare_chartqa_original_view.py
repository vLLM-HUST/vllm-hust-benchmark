"""Lossless original HF ChartQA test view for unmodified VLMEvalKit.

This is storage-format preparation, not a new benchmark or a model execution.
Raw source and generated images/questions/answers must remain outside Git.
"""

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path

NAME = "ChartQA_HF_ORIGINAL_TEST"
SHA256 = "efcf31333368c485eb569ec7bd4cc3694bd0e8e1a06c503124b51d27c924491d"  # pragma: allowlist secret
BYTES = 68898419
ROWS = 2500
REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"  # pragma: allowlist secret


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def validate_row(row):
    if not isinstance(row.get("query"), str) or not isinstance(row.get("label"), list):
        raise ValueError("Original question/reference types must be preserved")
    if not row["label"] or not all(isinstance(answer, str) for answer in row["label"]):
        raise ValueError("Original nonempty string answer list required")
    if not isinstance(row.get("image"), dict) or not isinstance(
        row["image"].get("bytes"), bytes
    ):
        raise ValueError(
            "Original image bytes required; no image generation or reconstruction"
        )
    if not isinstance(row.get("human_or_machine"), (int, str)):
        raise ValueError("Original source category required")


def write_view(rows, target):
    with Path(target).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["index", "image", "question", "answer", "category"],
            delimiter="\t",
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            validate_row(row)
            writer.writerow(
                {
                    "index": index,
                    "image": base64.b64encode(row["image"]["bytes"]).decode("ascii"),
                    "question": row["query"],
                    "answer": json.dumps(row["label"], ensure_ascii=False),
                    "category": row["human_or_machine"],
                }
            )


def verify_view(rows, target):
    csv.field_size_limit(32 * 1024 * 1024)
    count = 0
    with Path(target).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for count, row in enumerate(rows, 1):
            validate_row(row)
            saved = next(reader, None)
            if saved is None or saved["index"] != str(count - 1):
                raise ValueError("Original order/count lost")
            if (
                saved["question"] != row["query"]
                or json.loads(saved["answer"]) != row["label"]
                or saved["category"] != str(row["human_or_machine"])
                or base64.b64decode(saved["image"], validate=True)
                != row["image"]["bytes"]
            ):
                raise ValueError(
                    "Original question, answer, category or image bytes changed"
                )
        if next(reader, None) is not None:
            raise ValueError("Unexpected additional questions")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Dedicated external LMUData directory",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    source, output = args.source_file.resolve(), args.output.resolve()
    if source == output or source.is_relative_to(output):
        raise ValueError("Generated view must be separate from original assets")
    if not args.execute and not args.verify_only:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "source_revision": REVISION,
                    "source_sha256": SHA256,
                    "rows": ROWS,
                    "view_name": NAME,
                    "model_calls": 0,
                }
            )
        )
        return
    if source.stat().st_size != BYTES or digest(source) != SHA256:
        raise ValueError("Original test Parquet identity mismatch")
    import pyarrow.parquet as pq

    rows = pq.read_table(source).to_pylist()
    if len(rows) != ROWS:
        raise ValueError("Original test row count mismatch")
    target = output / f"{NAME}.tsv"
    if args.execute:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        write_view(rows, target)
    count = verify_view(rows, target)
    if digest(source) != SHA256:
        raise ValueError("Original asset changed during preparation")
    result = {
        "source_id": "CHARTQA",
        "status": "ALL_ORIGINAL_TEST_VIEW_RELATIONS_VERIFIED_NOT_NATIVE_EXECUTION",
        "source_revision": REVISION,
        "source_sha256": SHA256,
        "view_sha256": digest(target),
        "view_name": NAME,
        "rows": count,
        "image_bytes_unchanged": True,
        "question_answer_order_unchanged": True,
        "published_toolkit_tsv_identity_claim": False,
        "model_calls": 0,
        "native_benchmark_verified": False,
    }
    if args.execute:
        (output / "preparation-public.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
