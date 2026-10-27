"""Regrade saved native predictions with pinned, unchanged CrossCodeEval scoring.

Never calls a model. Outputs go to a new private directory, not the original run.
The compatible tree-sitter library is supplied explicitly and recorded by hash;
this is not a claim to reproduce the author's unpinned grammar build.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

from prepare_crosscodeeval_original_assets import PINS, verify_git_blob


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_predictions(original, predictions):
    if not original or len(original) != len(predictions):
        raise ValueError("all selected original cases require predictions")
    for row, prediction in zip(original, predictions, strict=True):
        if (
            {key: prediction.get(key) for key in row} != row
            or prediction.get("task_id") != row["metadata"]["task_id"]
            or not isinstance(prediction.get("pred"), str)
        ):
            raise ValueError("original fields, references or task IDs differ")


def selected_originals(archive, language, positions, profile):
    if (
        not positions
        or any(type(value) is not int or value < 0 for value in positions)
        or positions != list(range(positions[0], positions[0] + len(positions)))
    ):
        raise ValueError("fixed contiguous original positions required")
    name = f"{language}/{profile}.jsonl"
    with tarfile.open(archive, "r:xz") as bundle:
        matches = [item for item in bundle.getmembers() if item.name == name]
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError("one original regular task member required")
        selected = []
        with bundle.extractfile(matches[0]) as stream:
            for index, line in enumerate(stream):
                if positions[0] <= index <= positions[-1]:
                    selected.append(json.loads(line))
                if index >= positions[-1]:
                    break
    if len(selected) != len(positions):
        raise ValueError("original sample extends beyond member")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--data-archive", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ts-lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be in 1..32")
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    for item in pins["author_files"]:
        verify_git_blob(args.author_root / item["path"], item)
    verify_git_blob(args.data_archive, pins["dataset"])
    receipt_path = args.run_root / "receipt-private.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("source_id") != "CROSSCODEEVAL"
        or receipt.get("status")
        != "AUTHOR_NATIVE_FOUR_LANGUAGE_NPU_GENERATION_COMPLETE_SCORING_PENDING"
        or [item["language"] for item in receipt["languages"]]
        != pins["native_profile"]["languages"]
        or any(
            receipt.get(key) != value
            for key, value in pins["native_profile"]["author_api_defaults"].items()
        )
    ):
        raise ValueError("completed pinned four-language native generation required")
    prepared = []
    for item in receipt["languages"]:
        language = item["language"]
        root = args.run_root / language
        prediction = root / "prediction.jsonl"
        subset_name = item.get("subset_file", "original-first3.private.jsonl")
        if subset_name not in (
            "original-first3.private.jsonl",
            "original.private.jsonl",
        ):
            raise ValueError("unknown original subset layout")
        subset = root / subset_name
        if (
            sha256(prediction) != item["predictions_sha256"]
            or sha256(subset) != item["original_subset_sha256"]
        ):
            raise ValueError("saved prediction/original subset identity differs")
        original = selected_originals(
            args.data_archive,
            language,
            item["original_positions"],
            pins["native_profile"]["task"],
        )
        if [json.loads(line) for line in subset.read_bytes().splitlines()] != original:
            raise ValueError("saved subset is not the selected original archive tasks")
        validate_predictions(
            original,
            [json.loads(line) for line in prediction.read_bytes().splitlines()],
        )
        prepared.append((item, prediction, subset, len(original)))
    if receipt["model_calls"] != sum(item[3] for item in prepared):
        raise ValueError("native request coverage differs")
    library_sha = sha256(args.ts_lib)
    # Load only after all input checks; never import model-generation code.
    sys.path.insert(0, str(args.author_root.resolve() / "scripts"))
    import eval_metric

    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    results = []
    for item, prediction, subset, count in prepared:
        root = args.output / item["language"]
        root.mkdir(mode=0o700)
        shutil.copyfile(prediction, root / "prediction.jsonl")
        shutil.copyfile(subset, root / "original.private.jsonl")
        native_args = SimpleNamespace(
            output_dir=str(root.resolve()),
            prompt_file=str((root / "original.private.jsonl").resolve()),
            ts_lib=str(args.ts_lib.resolve()),
            language=item["language"],
        )
        # Author pool sizes as cpu_count()-1. Only bound executor resources.
        with patch.object(eval_metric.mp, "cpu_count", return_value=args.workers + 1):
            eval_metric.compute_metric_stmt(native_args)
        result = json.loads((root / "results.json").read_text(encoding="utf-8"))
        if result["total"] != count:
            raise ValueError("native scorer dropped selected tasks")
        results.append(
            {
                "language": item["language"],
                "original_positions": item["original_positions"],
                "author_results": result,
            }
        )
    public = {
        "source_id": "CROSSCODEEVAL",
        "status": "SAVED_NATIVE_PREDICTIONS_ORIGINAL_AUTHOR_GRADING_COMPLETE",
        "generation_receipt_sha256": sha256(receipt_path),
        "publisher_revision": pins["revision"],
        "parser_library_sha256": library_sha,
        "workers": args.workers,
        "original_cases": sum(item[3] for item in prepared),
        "new_model_calls": 0,
        "results": results,
        "publisher_grading_python_unchanged": True,
        "quality_or_performance_acceptance_claim": False,
        "source_checkbox_eligible": False,
    }
    (args.output / "receipt-public.json").write_text(
        json.dumps(public, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(public))


if __name__ == "__main__":
    main()
