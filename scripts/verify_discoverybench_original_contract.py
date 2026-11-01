"""Read-only original real-task identity/file preflight, not Agent execution or scoring."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from prepare_discoverybench_original_assets import (
    HF_ANSWER_FILES,
    blob_identity,
    prepare_publisher_index,
)


def original_queries(value):
    if isinstance(value, list):
        for item in value:
            yield from original_queries(item)
    elif isinstance(value, dict) and "qid" in value and "question" in value:
        yield value
    else:
        raise ValueError("unexpected original query container")


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("missing or escaped original file: " + str(relative))
    return path


def check_relations(data, rows):
    """No downloads, task filtering, encoding guesses, aliases or prompt generation."""
    data = Path(data).resolve(strict=True)
    verified = 0
    errors = []
    seen = set()
    for row in rows:
        identity = {key: row[key] for key in ("dataset", "metadataid", "query_id")}
        try:
            dataset = row["dataset"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset) or dataset in {".", ".."}:
                raise ValueError("unsafe original dataset identifier")
            if not all(
                re.fullmatch(r"[0-9]+", row[key]) for key in ("metadataid", "query_id")
            ):
                raise ValueError("invalid original numeric identifier")
            # Match the integer identity used below to locate metadata and queries.
            # Alternate zero padding must not inflate verified task coverage.
            key = (dataset, int(row["metadataid"]), int(row["query_id"]))
            if key in seen:
                raise ValueError("duplicate original task identity")
            seen.add(key)
            relative = (
                Path("test")
                / dataset
                / ("metadata_" + str(int(row["metadataid"])) + ".json")
            )
            metadata_path = inside(data, relative)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            queries = [
                query
                for query in original_queries(metadata["queries"])
                if int(query["qid"]) == int(row["query_id"])
            ]
            if len(queries) != 1 or not isinstance(queries[0]["question"], str):
                raise ValueError("original query relation is not unique")
            if not row["gold_hypo"]:
                raise ValueError("empty original reference")
            if not isinstance(metadata["datasets"], list) or not metadata["datasets"]:
                raise ValueError("empty original dataset references")
            for item in metadata["datasets"]:
                inside(data, metadata_path.parent.relative_to(data) / item["name"])
            verified += 1
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append(
                {**identity, "error_type": type(error).__name__, "error": str(error)}
            )
    return {
        "source_id": "DISCOVERYBENCH",
        "status": "ORIGINAL_RELATIONS_VERIFIED_NOT_EXECUTED"
        if not errors
        else "ORIGINAL_RELATION_FAILURES_RETAINED_NOT_EXECUTED",
        "task_references": len(rows),
        "verified_task_references": verified,
        "error_type_counts": dict(Counter(row["error_type"] for row in errors)),
        "errors": errors,
        "model_calls": 0,
        "original_files_changed": False,
        "source_checkbox_eligible": False,
        "boundary": "Only same-release real-task identity, original UTF8 metadata, exact filenames and nonempty separate references. Not author prompt construction, Agent/code isolation, native execution, scoring or research handoff. Errors retained; no filtering or guessed repair.",
    }


def verify(native_assets, answer_assets):
    native_assets = Path(native_assets).resolve(strict=True)
    answer_assets = Path(answer_assets).resolve(strict=True)
    prepare_publisher_index(native_assets, verify_only=True, native_real=True)
    name = "answer_key/answer_key_real.csv"
    path = inside(answer_assets, name)
    size, identity = HF_ANSWER_FILES[name]
    if path.stat().st_size != size or blob_identity(path) != identity:
        raise ValueError("original real answer key identity mismatch")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return check_relations(native_assets / "discoverybench/real", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-assets", required=True, type=Path)
    parser.add_argument("--answer-assets", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.native_assets, args.answer_assets)
    print(json.dumps(result, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
