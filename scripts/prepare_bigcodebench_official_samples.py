"""Prepare official BigCodeBench sanitizer/evaluator inputs, execute no code.

Keep raw original generated solution text. Author sanitize is a separate, pinned
step, not our cleanup. All original dataset fields remain in a private JSONL for
the author's documented BIGCODEBENCH_OVERRIDE_PATH. Nothing is published.
"""

import argparse
import hashlib
import json
from pathlib import Path

from run_bigcodebench_original_generation import (
    CODE_REVISION,
    DATA_SHA,
    UTILITY_SHA,
    format_prompt,
    load_original,
    response_summary,
    select_indices,
)

TASK_COUNT = 1140


def prepare(generation_dir, *, rows, formatter, output):
    generation_dir, output = Path(generation_dir).resolve(), Path(output).resolve()
    if (
        output.exists()
        or output.is_relative_to(generation_dir)
        or generation_dir.is_relative_to(output)
    ):
        raise ValueError("fresh output separate from historical generation required")
    ids = [row["task_id"] for row in rows]
    if len(rows) != TASK_COUNT or ids != [
        f"BigCodeBench/{i}" for i in range(TASK_COUNT)
    ]:
        raise ValueError("complete original ordered task set required")
    receipt_raw = (generation_dir / "receipt.json").read_bytes()
    receipt = json.loads(receipt_raw)
    selected_indices = receipt.get("selected_original_row_indices")
    if not isinstance(selected_indices, list):
        raise ValueError("original selection list required")
    selected_indices = select_indices(TASK_COUNT, indices=selected_indices)
    selected_rows = [rows[i] for i in selected_indices]
    selected_ids = [ids[i] for i in selected_indices]
    if (
        receipt.get("source_id") != "BIGCODEBENCH"
        or receipt.get("code_revision") != CODE_REVISION
        or receipt.get("split") != "instruct"
        or receipt.get("status")
        != "ORIGINAL_GENERATION_PROFILE_ATTEMPTED_NOT_FUNCTIONALLY_SCORED"
        or receipt.get("max_completion_tokens") != 1280
        or receipt.get("temperature") != 0
        or receipt.get("top_p") != 0.95
    ):
        raise ValueError("original Instruct generation profile receipt required")
    prior = receipt.get("samples", [])
    if [item.get("task_id") for item in prior] != selected_ids or [
        item.get("row_index") for item in prior
    ] != selected_indices:
        raise ValueError("historical task selection mismatch")
    solutions, identities = [], []
    for i, row, item in zip(selected_indices, selected_rows, prior):
        path = (generation_dir / f"{i}.private.json").resolve()
        if not path.is_relative_to(generation_dir):
            raise ValueError("escaped private generation path")
        raw = path.read_bytes()
        saved = json.loads(raw)
        messages = [
            {
                "role": "user",
                "content": format_prompt(formatter, row["instruct_prompt"]),
            }
        ]
        if saved.get("task_id") != row["task_id"] or saved.get("messages") != messages:
            raise ValueError("original task or publisher-formatted prompt mismatch")
        summary = response_summary(saved["response"], item["prompt_tokens"])
        if summary["finish_reason"] not in {"stop", "length"} or any(
            summary[key] != item.get(key)
            for key in ("finish_reason", "prompt_tokens", "completion_tokens", "status")
        ):
            raise ValueError("historical terminal/usage metadata mismatch")
        solutions.append(
            {
                "task_id": row["task_id"],
                "solution": saved["response"]["choices"][0]["message"]["content"],
            }
        )
        identities.append(
            {
                "task_id": row["task_id"],
                "observed_now_private_file_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if (generation_dir / "receipt.json").read_bytes() != receipt_raw:
        raise ValueError("generation receipt changed during preparation")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    paths = {}
    for name, records in (
        ("raw-solutions-private.jsonl", solutions),
        ("original-dataset-private.jsonl", rows),
    ):
        path = output / name
        with path.open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "status": "AUTHOR_SANITIZER_INPUT_PREPARED_NOT_EXECUTED",
        "source_id": "BIGCODEBENCH",
        "code_revision": CODE_REVISION,
        "original_parquet_sha256_required_by_cli": DATA_SHA,
        "original_dataset_rows_preserved": len(rows),
        "selected_task_ids": selected_ids,
        "selected_original_row_indices": selected_indices,
        "new_model_calls": 0,
        "generated_code_executed": False,
        "official_sanitizer_executed": False,
        "functional_scoring_executed": False,
        "original_solution_text_modified": False,
        "source_checkbox_eligible": False,
        "prior_per_file_cryptographic_binding_available": False,
        "historical_provenance_assurance": "Current saved files match original prompts and prior task/termination/usage metadata. Newly observed hashes do not retroactively prove historical byte identity.",
        "historical_files": identities,
        "prepared_file_sha256": paths,
        "generation_receipt_observed_sha256": hashlib.sha256(receipt_raw).hexdigest(),
    }
    (output / "receipt-private.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from run_bigcodebench_original_generation import digest
    import pyarrow.parquet as pq

    path = args.data_root / "data/v0.1.4-00000-of-00001.parquet"
    if digest(path) != DATA_SHA:
        raise ValueError("original publisher dataset identity mismatch")
    output = args.output.resolve()
    for root in (args.data_root.resolve(), args.official_code.resolve()):
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("output must be separate from original data/code")
    formatter = load_original(
        "bcb_original_formatter",
        args.official_code / "bigcodebench/provider/utility.py",
        UTILITY_SHA,
    )
    rows = pq.read_table(path).to_pylist()
    if digest(path) != DATA_SHA:
        raise ValueError("original publisher dataset changed during reading")
    prepare(args.generation_dir, rows=rows, formatter=formatter, output=output)


if __name__ == "__main__":
    main()
