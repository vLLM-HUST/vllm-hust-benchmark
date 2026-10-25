"""Reuse DS-1000 generations for author functional evaluation; execute no code.

Keep full original rows/code_context separate from the model request. Preserve
publisher post-generation processing and budget exhaustion, never repair code.
"""

import argparse
import hashlib
import json
from pathlib import Path

from run_ds1000_original_generation import (
    CODE_REVISION,
    DATA_SHA,
    SOURCE_SHA,
    digest,
    parse_publisher_profile,
    request_body,
    response_summary,
    select_indices,
)

TASK_COUNT = 1000


def prepare(generation_dir, *, rows, profile, output, indices=None):
    generation_dir, output = Path(generation_dir).resolve(), Path(output).resolve()
    if (
        output.exists()
        or output.is_relative_to(generation_dir)
        or generation_dir.is_relative_to(output)
    ):
        raise ValueError("fresh output separate from historical generation required")
    if len(rows) != TASK_COUNT or [
        row["metadata"]["problem_id"] for row in rows
    ] != list(range(TASK_COUNT)):
        raise ValueError(
            "full original dataset and original first-task identities required"
        )
    receipt_raw = (generation_dir / "receipt.json").read_bytes()
    receipt = json.loads(receipt_raw)
    generated = select_indices(
        len(rows), indices=receipt.get("selected_original_row_indices", [])
    )
    selected = (
        generated if indices is None else select_indices(len(rows), indices=indices)
    )
    if set(selected) - set(generated):
        raise ValueError(
            "prepared task selection must be contained in original generation"
        )
    if (
        receipt.get("source_id") != "DS1000"
        or receipt.get("code_revision") != CODE_REVISION
        or receipt.get("publisher_literal_config") != profile
        or receipt.get("status")
        != "ORIGINAL_COMPLETION_PROFILE_NOT_FUNCTIONALLY_SCORED"
    ):
        raise ValueError("original raw completion profile receipt required")
    samples = receipt.get("samples", [])
    if [item.get("original_problem_id") for item in samples] != generated:
        raise ValueError("historical task selection mismatch")
    by_id = {item["original_problem_id"]: item for item in samples}
    answers, identities = [], []
    for i in selected:
        prior = by_id[i]
        row = rows[i]
        raw_path = (generation_dir / f"{i}.private.json").resolve()
        processed_path = (generation_dir / f"{i}.postprocessed.private.json").resolve()
        if any(
            not path.is_relative_to(generation_dir)
            for path in (raw_path, processed_path)
        ):
            raise ValueError("escaped historical generation path")
        raw, processed = raw_path.read_bytes(), processed_path.read_bytes()
        saved = json.loads(raw)
        if saved.get("request") != request_body(row["prompt"], receipt["model"]):
            raise ValueError("original prompt or generation configuration mismatch")
        summary, code = response_summary(
            saved["response"], prior["prompt_tokens"], profile["stop_words"]
        )
        if summary["finish_reason"] not in {"stop", "length"} or any(
            summary[key] != prior.get(key) for key in summary
        ):
            raise ValueError("original terminal/usage/postprocessing metadata mismatch")
        expected = {"id": i, "code": code, "metadata": row["metadata"]}
        if json.loads(processed) != expected:
            raise ValueError("saved publisher postprocessing differs; no silent repair")
        answers.append(expected)
        identities.append(
            {
                "problem_id": i,
                "raw_file_observed_sha256": hashlib.sha256(raw).hexdigest(),
                "processed_file_observed_sha256": hashlib.sha256(processed).hexdigest(),
                "finish_reason": summary["finish_reason"],
            }
        )
    if (generation_dir / "receipt.json").read_bytes() != receipt_raw:
        raise ValueError("generation receipt changed during preparation")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    files = {}
    for name, records in (
        ("answers-private.jsonl", answers),
        ("original-dataset-private.jsonl", rows),
    ):
        path = output / name
        with path.open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "status": "AUTHOR_FUNCTIONAL_EVALUATOR_INPUT_PREPARED_NOT_EXECUTED",
        "source_id": "DS1000",
        "code_revision": CODE_REVISION,
        "original_dataset_sha256_required_by_cli": DATA_SHA,
        "all_original_rows_preserved": len(rows),
        "selected_problem_ids": selected,
        "parent_generation_selected_problem_ids": generated,
        "new_model_calls": 0,
        "generated_code_executed": False,
        "code_context_executed": False,
        "official_functional_score_complete": False,
        "source_checkbox_eligible": False,
        "prior_per_file_hash_binding_available": False,
        "assurance": "Current outputs match original requests and recorded metadata; new hashes do not establish historical byte custody. Original inference marker postprocessing preserved; author test postprocess remains a separate step.",
        "historical_files": identities,
        "prepared_file_sha256": files,
    }
    (output / "receipt-private.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--publisher-inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="Explicit subset of saved original selection; parent selection stays recorded",
    )
    args = parser.parse_args()
    source = args.data_root / "test.jsonl"
    if digest(source) != DATA_SHA or digest(args.publisher_inference) != SOURCE_SHA:
        raise ValueError("fixed original data and publisher source required")
    out, data_root = args.output.resolve(), args.data_root.resolve()
    if (
        out.is_relative_to(data_root)
        or data_root.is_relative_to(out)
        or args.publisher_inference.resolve().is_relative_to(out)
    ):
        raise ValueError("output must be separate from original data/code")
    profile = parse_publisher_profile(args.publisher_inference.read_text("utf-8"))
    rows = [json.loads(line) for line in source.read_text("utf-8").splitlines()]
    if digest(source) != DATA_SHA:
        raise ValueError("original data changed during reading")
    prepare(
        args.generation_dir,
        rows=rows,
        profile=profile,
        output=out,
        indices=args.indices,
    )


if __name__ == "__main__":
    main()
