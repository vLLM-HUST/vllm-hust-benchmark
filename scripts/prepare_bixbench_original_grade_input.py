"""Join completed native trajectories to verified original questions for grading.

Plan-only by default. No new answers, model calls, dropped failures or scoring.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def grade_row(original, trajectory, load_answer):
    if trajectory["problem_id"] != original["question_id"]:
        raise ValueError("native problem identity mismatch")
    if trajectory["ideal_answer"] != original["ideal"]:
        raise ValueError("native reference differs from original")
    answer = load_answer(trajectory["agent_answer"])
    # Same single-answer extraction as publisher create_eval_df; no synthesis.
    if isinstance(answer, dict) and len(answer) == 1:
        answer = next(iter(answer.values()))
    row = {
        "question_id": original["question_id"],
        "question": original["question"],
        "target": original["ideal"],
        "predicted": answer,
    }
    raw_mode = original.get("eval_mode")
    csv_mode = original.get("evaluation_mode")
    if raw_mode is not None and csv_mode is not None and raw_mode != csv_mode:
        raise ValueError("conflicting original evaluation mode fields")
    mode = raw_mode if raw_mode is not None else csv_mode
    if mode not in {"str_verifier", "range_verifier", "llm_verifier"}:
        raise ValueError("original evaluation mode missing or unsupported")
    # Raw publisher JSONL uses eval_mode; grade_outputs.py expects this CSV key.
    row["evaluation_mode"] = mode
    return row


def prepare(args):
    author, data, native, output = (
        Path(value).resolve()
        for value in (args.author_root, args.data_root, args.native_output, args.output)
    )
    if not 1 <= args.limit <= 205:
        raise ValueError("original prefix between1 and205 required")
    if (
        output.exists()
        or output.is_relative_to(ROOT)
        or any(
            output.is_relative_to(source) or source.is_relative_to(output)
            for source in (author, data, native)
        )
    ):
        raise ValueError("fresh external output required")
    report = {
        "source_id": "BIXBENCH",
        "status": "PLAN_ONLY",
        "limit": args.limit,
        "model_calls": 0,
        "scoring_executed": False,
        "source_checkbox_eligible": False,
    }
    if not args.execute:
        return report
    from prepare_bixbench_original_tasks import CATALOG, verify_asset
    from prepare_jsonschemabench_author_source import verify_source

    verify_source(author, ROOT / "configs/workload-native/bixbench-author-source.json")
    entries = {
        item["path"]: item for item in json.loads(CATALOG.read_text("utf-8"))["files"]
    }
    metadata = verify_asset(data, entries["BixBench.jsonl"])
    original = [json.loads(line) for line in metadata.read_text("utf-8").splitlines()]
    selected = original[: args.limit]
    state = json.loads((native / "state.private.json").read_text("utf-8"))
    expected = [row["question_id"] for row in selected]
    if (
        state["source_id"] != "BIXBENCH"
        or state["completed"] != expected
        or state["status"]
        != "ORIGINAL_SELECTED_AGENT_TRAJECTORIES_SAVED_SCORING_PENDING"
    ):
        raise ValueError(
            "all selected original native trajectories must be saved first"
        )
    sys.path.insert(0, str(author))
    from bixbench.postprocessing_utils import load_answer

    found = {}
    for path in (native / "trajectories").rglob("*.json"):
        if path.is_symlink() or not path.resolve().is_relative_to(native):
            raise ValueError("trajectory escapes private native root")
        value = json.loads(path.read_text("utf-8"))
        identity = value.get("problem_id")
        if identity not in expected or identity in found:
            raise ValueError("unexpected or duplicate native trajectory")
        found[identity] = value
    if set(found) != set(expected):
        raise ValueError("missing native trajectories; do not replace failed questions")
    rows = [grade_row(row, found[row["question_id"]], load_answer) for row in selected]
    output.mkdir(mode=0o700)
    destination = output / "original-grade-input.private.csv"
    fields = ["question_id", "question", "target", "predicted", "evaluation_mode"]
    with destination.open("x", newline="", encoding="utf-8") as stream:
        os.chmod(destination, 0o600)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report.update(
        status="ORIGINAL_NATIVE_GRADING_INPUT_PREPARED_NOT_SCORED",
        rows=len(rows),
        csv_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
    )
    (output / "preparation.private.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("author-root", "data-root", "native-output", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
