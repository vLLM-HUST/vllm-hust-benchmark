"""Reconstruct original RepoBench selection and rescore saved outputs, no inference.

Run in the qualified native environment. Raw inputs and answers stay external.
Partial mode audits only retained completed outputs and never completes a source.
Failed/unknown attempts remain explicit; no replacement, normalization or retry.
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import run_repobench_original_npu as native


def equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"original saved evidence mismatch: {label}")


def audit_saved_tasks(tasks, root, author, tokenizer, *, partial=False):
    completed = []
    unknown = []
    paths = sorted(root.glob("attempt-*.private.json"))
    for expected_index, path in enumerate(paths):
        equal(
            path.name,
            f"attempt-{expected_index:05d}.private.json",
            "contiguous original attempt order",
        )
        attempt = json.loads(path.read_text())
        index = attempt["task_index"]
        equal(index, expected_index, "fixed task index")
        if index >= len(tasks):
            raise ValueError("extra unselected original attempt")
        task = tasks[index]
        for key in ("language", "setting", "filtered_index"):
            equal(attempt[key], task[key], "original task relation")
        if attempt["status"] != "GENERATED_ORIGINAL_NEXT_LINE":
            unknown.append(index)
            if not partial:
                raise ValueError(
                    "failed or unknown original attempt; cannot claim completion"
                )
            continue
        tokens = attempt["token_ids"]
        if not 1 <= len(tokens) <= 128 or any(
            type(token) is not int or token < 0 for token in tokens
        ):
            raise ValueError("native token output/budget mismatch")
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        equal(raw, attempt["raw_output"], "saved generated token decoding")
        pred = author.get_first_line_not_comment(raw, task["language"])
        equal(pred, attempt["pred"], "unchanged author next-line extraction")
        completed.append(
            (
                task,
                {
                    "idx": task["filtered_index"],
                    "level": task["original"]["level"],
                    "pred": pred,
                    "gt": task["original"]["next_line"],
                },
            )
        )
    if not partial:
        equal(len(completed), len(tasks), "all fixed original tasks completed")
    return completed, unknown


def audit_result_files(completed, root, *, partial=False):
    expected = {}
    for task, result in completed:
        key = (task["language"], task["setting"])
        expected.setdefault(key, []).append(result)
    for (language, setting), rows in expected.items():
        path = root / language / (setting + ".jsonl")
        saved = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if partial:
            # The worker can append the next response while the snapshot is read.
            equal(saved[: len(rows)], rows, "retained original result prefix")
        else:
            equal(saved, rows, "exact original gt/pred and source indices")
            equal(
                len({row["idx"] for row in saved}),
                len(saved),
                "no silently deduplicated outputs",
            )


def execute(args):
    native.verify_original(args.source_root, args.archive)
    native.verify_files(args.model_root, native.MODEL_FILES)
    root = args.run_root.resolve()
    receipt = json.loads((root / "receipt-public.json").read_text())
    equal(receipt["source_id"], "REPOBENCH", "source")
    equal(receipt["code_revision"], native.REVISION, "author revision")
    equal(
        receipt["model_revision"], native.MODEL_REVISION, "author model asset revision"
    )
    for key, expected in {
        "prompt_budget": 15800,
        "max_new_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.95,
        "do_sample": True,
    }.items():
        equal(receipt[key], expected, "author generation profile")
    if not args.partial:
        equal(
            receipt["status"],
            "ORIGINAL_GENERATION_AND_AUTHOR_SCORING_COMPLETED_PENDING_SAVED_AUDIT",
            "terminal native state",
        )
        equal(receipt["verified_parameter_device_type"], "npu", "model actually on NPU")
    sys.path.insert(0, str(args.source_root.resolve()))
    author = native.original_module(args.source_root, "run")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_root), local_files_only=True, trust_remote_code=False
    )
    saved_selection = json.loads((root / "selection.private.json").read_text())
    languages = list(dict.fromkeys(task["language"] for task in saved_selection))
    prepared = native.prepare(
        SimpleNamespace(
            data_root=args.data_root,
            output=root,
            languages=languages,
            per_stratum=receipt["per_stratum"],
        ),
        author,
        tokenizer,
    )
    equal(
        json.loads(json.dumps(prepared, default=str)),
        saved_selection,
        "fixed selection, original references and author prompts reconstructed from original Parquet",
    )
    equal(len(saved_selection), receipt["selected_tasks"], "declared fixed sample size")
    completed, unknown = audit_saved_tasks(
        saved_selection, root, author, tokenizer, partial=args.partial
    )
    audit_result_files(completed, root, partial=args.partial)
    scores = {}
    legacy_report_matches = {}
    if not args.partial:
        equal(
            receipt["model_calls"],
            len(saved_selection),
            "one native generation per selected task",
        )
        freeze = getattr(args, "freeze_scoring_output", None)
        if receipt.get("scoring_python_hash_seed") != 0 and freeze is None:
            raise ValueError(
                "Legacy unseeded CodeBLEU reports are hash-sensitive; explicitly freeze scores in a new external directory, retaining original reports"
            )
        if freeze is not None:
            freeze = freeze.resolve()
            if freeze.exists() or any(
                freeze.is_relative_to(p.resolve())
                for p in [root, args.source_root, args.data_root, args.model_root]
            ):
                raise ValueError(
                    "fresh scoring directory outside original source/data/model/run required"
                )
            freeze.mkdir(parents=True, mode=0o700)
        for language in languages:
            report = native.score_original(args.source_root, root / language, language)
            equal(
                native.score_original(args.source_root, root / language, language),
                report,
                "repeatable unchanged author scoring at PYTHONHASHSEED=0",
            )
            legacy_report_matches[language] = (
                report == (root / (language + "-official-scoring.txt")).read_text()
            )
            if receipt.get("scoring_python_hash_seed") == 0:
                equal(
                    legacy_report_matches[language],
                    True,
                    "original fixed-seed author report",
                )
            if freeze is not None:
                (freeze / (language + "-official-scoring.txt")).write_text(report)
            scores[language] = report
        if freeze is not None:
            (freeze / "scoring-profile-public.json").write_text(
                json.dumps(
                    {
                        "source_id": "REPOBENCH",
                        "python_hash_seed": 0,
                        "author_scoring_code_unchanged": True,
                        "unseeded_reports_retained": True,
                        "original_report_matches_fixed_seed": legacy_report_matches,
                        "new_model_calls": 0,
                    },
                    indent=2,
                )
            )
    native.verify_original(args.source_root, args.archive)
    return {
        "source_id": "REPOBENCH",
        "status": "PARTIAL_SAVED_ORIGINAL_TASKS_VERIFIED_NOT_COMPLETE"
        if args.partial
        else "ALL_SELECTED_ORIGINAL_TASKS_AND_AUTHOR_SCORES_VERIFIED",
        "selected_tasks": len(saved_selection),
        "verified_saved_tasks": len(completed),
        "unknown_attempt_indices": unknown,
        "official_reports": scores,
        "scoring_python_hash_seed": None if args.partial else 0,
        "legacy_original_report_matches_fixed_seed": legacy_report_matches,
        "new_model_calls": 0,
        "source_checkbox_eligible": False,
        "note": "Source delivery additionally requires public/research consumer evidence; this is not author CUDA baseline quality reproduction.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "archive", "data-root", "model-root", "run-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument(
        "--freeze-scoring-output",
        type=Path,
        help="Explicit fresh external scoring output for historical unseeded reports; original reports retained, no inference",
    )
    args = parser.parse_args()
    print(json.dumps(execute(args), indent=2))


if __name__ == "__main__":
    main()
