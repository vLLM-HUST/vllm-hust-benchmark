"""Select original full-test JSONL rows; never decode pickle or execute tasks.

The fixed sample is the first N publisher-file rows, with every original field
unchanged. Output is external, private and is not native execution evidence.
"""

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "workload-preparation/data/catalogs/source_manifests_v1.json"
spec = importlib.util.spec_from_file_location(
    "lcb_original_io", Path(__file__).with_name("prepare_xsum_original_assets.py")
)
IO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(IO)


def prepare(data_root, output, *, sample_count=3, catalog=CATALOG):
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("positive fixed sample count required")
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if (
        output.exists()
        or output.is_relative_to(data_root)
        or data_root.is_relative_to(output)
    ):
        raise ValueError("fresh output separate from original assets required")
    source = json.loads(Path(catalog).read_text("utf-8"))["sources"]["LIVECODEBENCH"]
    items = [f for f in source["files"] if f["path"] == "code_generation/test.jsonl"]
    if len(items) != 1:
        raise ValueError("fixed original full-test source required")
    item = items[0]
    path = IO.bound_path(data_root, "", item)
    IO.verify(path, item)
    records = []
    with path.open("rb") as stream:
        for position in range(sample_count):
            raw = stream.readline(1024**3 + 1)
            if len(raw) > 1024**3:
                raise ValueError(
                    "original row exceeds preparation memory gate; do not truncate"
                )
            if not raw:
                raise ValueError("fixed sample exceeds original source rows")
            row = json.loads(raw)
            if (
                not isinstance(row, dict)
                or not {
                    "question_id",
                    "question_content",
                    "public_test_cases",
                    "private_test_cases",
                    "metadata",
                }
                <= row.keys()
            ):
                raise ValueError("original code-generation fields missing")
            if not isinstance(row["private_test_cases"], str):
                raise ValueError("retain original opaque private-test string")
            records.append((position, row))
    identifiers = [row["question_id"] for _, row in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate original question identifiers")
    output.mkdir(mode=0o700, parents=True)
    task_file = output / "codegen-original-tasks.private.json"
    task_file.write_text(
        json.dumps([row for _, row in records], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "source_id": "LIVECODEBENCH",
        "status": "ORIGINAL_FULL_TEST_TASK_ROWS_PREPARED_NOT_NATIVE_DELIVERY",
        "publisher_revision": item["publisher_revision"],
        "source_sha256": item["sha256"],
        "selection_rule": "first_n_publisher_file_rows_no_replacement",
        "selected_positions": [position for position, _ in records],
        "selected_question_ids": identifiers,
        "selected_field_names": [list(row) for _, row in records],
        "task_file_sha256": IO.sha(task_file),
        "private_tests_decoded": False,
        "original_fields_unchanged": True,
        "model_calls": 0,
        "native_execution_complete": False,
        "source_checkbox_eligible": False,
    }
    (output / "receipt-private.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "OFFLINE_PLAN_ONLY",
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                    "selection_rule": "first_n_publisher_file_rows_no_replacement",
                    "sample_count": args.sample_count,
                }
            )
        )
        return
    print(
        json.dumps(
            prepare(args.data_root, args.output, sample_count=args.sample_count),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
