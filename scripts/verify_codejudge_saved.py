"""Offline audit of original CodeJudge-Eval requests/responses and declared metrics."""

import argparse
from collections import Counter
import json
from pathlib import Path

from prepare_reused_research_assets import IO, selection
from run_codejudge_original import (
    digest,
    extract_choice,
    labels_for,
    metrics,
    request_body,
)


def verify_record(record, row, name, index, profile):
    if record.get("error_type"):
        raise ValueError("saved transport failure retained; not complete")
    if (
        record["file"] != name
        or record["row_index"] != index
        or record["original_prompt_sha256"] != digest(row["input"])
        or any(record[field] != row[field] for field in ("data_id", "task_id"))
    ):
        raise ValueError("original file/row/prompt/candidate identity differs")
    if record["reference_answer"] != row["answer"] or record["request"] != request_body(
        row, profile
    ):
        raise ValueError("original request/reference/profile differs")
    response = record["response"]
    if response.get("model") != profile["model"] or len(response["choices"]) != 1:
        raise ValueError("declared model or single-response contract differs")
    prediction = extract_choice(
        response["choices"][0]["message"].get("content") or "", labels_for(name)
    )
    if prediction != record["prediction"]:
        raise ValueError("saved extraction differs from declared strict-v1")
    return prediction


def audit(data_root, native_root):
    data_root, native_root = Path(data_root).resolve(), Path(native_root).resolve()
    native = json.loads((native_root / "native-public.json").read_text("utf-8"))
    if (
        native["status"]
        != "ORIGINAL_PUBLISHED_TASKS_EXECUTED_AND_PAPER_METRICS_IMPLEMENTED_SAVED_AUDIT_PENDING"
    ):
        raise ValueError("native run not terminal/scored")
    profile = native["profile"]
    if (
        native["profile_sha256"] != digest(profile)
        or profile["extractor"] != "independent-strict-v1-not-author-regex"
        or profile["temperature"] != 0.0
    ):
        raise ValueError("declared original-task fallback profile changed")
    source, files = selection("CODEJUDGE_EVAL", profile["files"])
    if profile["dataset_revision"] != source["revision"]:
        raise ValueError("original publisher revision differs")
    saved = [
        json.loads(line)
        for line in (native_root / "cases-private.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    expected = []
    for file in files:
        IO.verify(IO.bound_path(data_root, "", file), file)
        rows = json.loads((data_root / file["path"]).read_text("utf-8"))
        if len(rows) != file["rows"]:
            raise ValueError("original task count differs")
        expected.extend(
            (file["path"], index, row)
            for index, row in enumerate(rows if profile["full"] else rows[:3])
        )
    if len(saved) != len(expected) or native["saved_cases"] != len(expected):
        raise ValueError("original fixed selection not fully saved")
    grouped = {file["path"]: [] for file in files}
    finish_reasons = Counter()
    for record, (name, index, row) in zip(saved, expected):
        prediction = verify_record(record, row, name, index, profile)
        grouped[name].append((row["answer"], prediction))
        finish_reasons[record["response"]["choices"][0].get("finish_reason")] += 1
    recomputed = {
        name: metrics(
            [row[0] for row in values], [row[1] for row in values], labels_for(name)
        )
        for name, values in grouped.items()
    }
    if recomputed != native["configuration_metrics"]:
        raise ValueError("saved per-configuration paper metrics inconsistent")
    return {
        "source_id": "CODEJUDGE_EVAL",
        "status": "ORIGINAL_SELECTION_REQUEST_RESPONSE_AND_DECLARED_PAPER_METRICS_VERIFIED_OFFLINE",
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "original_configuration_cases": len(expected),
        "variants": len(files),
        "profile": profile,
        "configuration_metrics": recomputed,
        "finish_reasons": dict(finish_reasons),
        "boundary": "No network, model or code execution. Original published input preserved; independently implemented extraction/metrics, not author regex/score reproduction. Not full-source qualification or full1860-case quality scoring.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if (
        output.exists()
        or output.is_relative_to(Path(args.data_root).resolve())
        or output.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise ValueError("fresh external audit receipt required")
    result = audit(args.data_root, args.native_root)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"configuration_metrics", "profile"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
