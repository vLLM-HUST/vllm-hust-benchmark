"""Generate SQL for original prepared Spider inputs; official scoring is separate.

Spider supplies questions/schema/scoring, not a canonical LLM prompt or token
budget. This disclosed zero-shot integration prompt is NOT an official template.
No reference file is opened, SQL is never executed here, and outputs are not fixed.
"""

import argparse
import hashlib
import json
from pathlib import Path

from run_oasst_original_continuation import post, validate_endpoint


def requests(prepared, *, model, max_tokens):
    if not model or type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
        raise ValueError("explicit model and integration output budget required")
    root = Path(prepared).resolve()
    path = root / "inputs.private.jsonl"
    if path.is_symlink():
        raise ValueError("explicit prepared original inputs required")
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if not records:
        raise ValueError("nonempty fixed original prefix required")
    result = []
    for index, row in enumerate(records):
        if (
            set(row) != {"original_dev_index", "db_id", "question", "schema"}
            or row["original_dev_index"] != index
            or not isinstance(row["question"], str)
            or not row["question"]
            or row["schema"].get("db_id") != row["db_id"]
        ):
            raise ValueError("original question/schema input contract differs")
        prompt = (
            "Given the original database schema and question below, return the SQLite SQL query "
            "with its actual values. Return exactly one line of SQL, without Markdown or explanation.\n"
            "Original schema JSON:\n"
            + json.dumps(row["schema"], ensure_ascii=False)
            + "\nOriginal question:\n"
            + row["question"]
        )
        result.append(
            {
                "original_dev_index": index,
                "request": {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "n": 1,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            }
        )
    return hashlib.sha256(raw).hexdigest(), result


def run(
    *,
    prepared,
    model,
    max_tokens,
    endpoint="http://127.0.0.1:19466/v1",
    output=None,
    execute=False,
    transport=post,
):
    endpoint = validate_endpoint(endpoint)
    input_sha, selected = requests(prepared, model=model, max_tokens=max_tokens)
    report = {
        "source_id": "SPIDER",
        "status": "PLAN_ONLY_NO_MODEL_CALLS",
        "prepared_input_sha256": input_sha,
        "original_dev_indices": [r["original_dev_index"] for r in selected],
        "model": model,
        "max_tokens": max_tokens,
        "prompt_origin": "DISCLOSED_ZERO_SHOT_INTEGRATION_NOT_AUTHOR_TEMPLATE",
        "reference_files_opened": False,
        "sql_executed": False,
        "model_calls": 0,
        "model_call_count_scope": "HTTP_REQUEST_ATTEMPTS_NOT_EXECUTION_PROOF",
        "official_scoring_completed": False,
        "source_checkbox_eligible": False,
    }
    if not execute:
        return report
    if output is None:
        raise ValueError("fresh private output required")
    root, output = Path(prepared).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("fresh output separate from original prepared inputs required")
    output.mkdir(mode=0o700, parents=True)
    report["status"] = "ORIGINAL_SQL_GENERATION_RUNNING"
    receipt_path = output / "receipt-public.json"
    receipt_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    candidates, outcomes = [], []
    for item in selected:
        index = item["original_dev_index"]
        (output / f"{index}.request-private.json").write_text(
            json.dumps(item, ensure_ascii=False), encoding="utf-8"
        )
        outcome = {"original_dev_index": index}
        report["model_calls"] += 1
        try:
            response = transport(endpoint + "/chat/completions", item["request"])
            (output / f"{index}.response-private.json").write_text(
                json.dumps(response, ensure_ascii=False), encoding="utf-8"
            )
            choice = response["choices"][0]
            content = choice["message"].get("content")
            valid = (
                isinstance(content, str)
                and bool(content.strip())
                and len(content.splitlines()) == 1
            )
            outcome.update(
                status="ORIGINAL_SINGLE_LINE_CANDIDATE_RETAINED"
                if valid
                else "OUTPUT_FORMAT_BLOCKED_NO_SQL_REWRITE",
                finish_reason=choice.get("finish_reason"),
            )
            candidates.append(content if valid else None)
        except Exception as error:
            outcome.update(
                status="REQUEST_OR_RESPONSE_FAILURE_RETAINED",
                error_type=type(error).__name__,
            )
            candidates.append(None)
        outcomes.append(outcome)
        report["outcomes"] = outcomes
        receipt_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    unchanged = (
        hashlib.sha256((root / "inputs.private.jsonl").read_bytes()).hexdigest()
        == input_sha
    )
    ready = all(candidate is not None for candidate in candidates) and unchanged
    if ready:
        (output / "predictions.private.txt").write_text(
            "".join(c + "\n" for c in candidates), encoding="utf-8", newline="\n"
        )
    report.update(
        status="ORIGINAL_PREDICTIONS_READY_OFFICIAL_SCORING_PENDING"
        if ready
        else "ORIGINAL_GENERATION_BLOCKED_RETAINED_NO_TASK_REPLACEMENT",
        candidates_ready=ready,
        prepared_inputs_unchanged=unchanged,
    )
    receipt_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    result = run(**vars(parser.parse_args()))
    print(json.dumps(result))
    if result["status"] == "ORIGINAL_GENERATION_BLOCKED_RETAINED_NO_TASK_REPLACEMENT":
        raise SystemExit(2)
