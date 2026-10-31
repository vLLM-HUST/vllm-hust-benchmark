"""Original fixed3 text DST execution or read-only saved verification; never audio.

Generation sampling/budget controls stay omitted as in the author client.
The model and optional thinking binding are explicit, not leaderboard reproduction.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys

from prepare_spokenwoz_original_dst import (
    expected_prompts,
    prepare as prepare_inputs,
    selected_originals,
    verify_author,
)
from run_oasst_original_continuation import post, validate_endpoint


def request_body(prompt, model, thinking=None):
    if not isinstance(model, str) or not model:
        raise ValueError("explicit deployed model binding required")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt["content"]}],
    }
    if thinking is not None:
        if not isinstance(thinking, bool):
            raise ValueError("thinking binding must be explicit boolean")
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    return body


def original_state_score(rows):
    """Independent original state accumulation, rejecting stale-parse reuse."""
    if not rows or len({row["basename"] for row in rows}) != len(rows):
        raise ValueError("complete distinct saved original tasks required")
    initial = {key: "" for key in rows[0]["answer"]}
    running, correct, failures = {}, 0, []
    ordered = sorted(
        rows,
        key=lambda row: (
            row["basename"].split("-")[0] + row["basename"].split("-")[1].zfill(4)
        ),
    )
    for row in ordered:
        text = row["result"].replace("\n", "").replace("\r", "").replace("  ", "")
        try:
            if text[0] == "{" and text[-1] == "}":
                pass
            elif text[0] == " ":
                text = text[1:]
            else:
                text = "{}"
            if text[-2] == ",":
                text = text[:-2] + "}"
            elif "STATE:" in text:
                text = text.split("STATE:")[-1]
            elif text[-2] == " " and text[-3] == ",":
                text = text[:-3] + "}"
            predicted = json.loads(text)
            if not isinstance(predicted, dict):
                raise ValueError("original normalized result is not an object")
        except (ValueError, IndexError):
            failures.append(row["basename"])
            continue
        if row["basename"].split("-")[1] == "0":
            running = deepcopy(initial)
        for key, value in predicted.items():
            if key not in initial or str(value).lower() in [
                "unknown",
                "null",
                "none",
                "not specified",
                "repalce",
            ]:
                continue
            if value == []:
                continue  # Original scorer does not commit this branch to state.
            running[key] = value.lower() if isinstance(value, str) else ""
        correct += running == row["answer"]
    return {
        "correct_turns": correct if not failures else None,
        "total_turns": len(rows),
        "parse_failures": failures,
    }


def verify_saved(data_root, author_root, run_root, model, thinking=None):
    prompts = expected_prompts(selected_originals(data_root), author_root)
    run_root = Path(run_root)
    rows, finishes = [], Counter()
    for index, prompt in enumerate(prompts):
        request = json.loads(
            (run_root / f"{index:03d}-request-private.json").read_text(encoding="utf-8")
        )
        if request != request_body(prompt, model, thinking):
            raise ValueError("saved original request or declared binding differs")
        response = json.loads(
            (run_root / f"{index:03d}-response-private.json").read_text(
                encoding="utf-8"
            )
        )
        if response.get("model") != model or len(response.get("choices", [])) != 1:
            raise ValueError("saved response model/choice contract differs")
        choice = response["choices"][0]
        if choice.get("finish_reason") not in {"stop", "length"} or not isinstance(
            choice["message"].get("content"), str
        ):
            raise ValueError("saved native response is not a legal text terminal")
        finishes[choice["finish_reason"]] += 1
        rows.append({**prompt, "result": choice["message"]["content"]})
    score = original_state_score(rows)
    report = {
        "source_id": "SPOKENWOZ",
        "status": "FIXED_ORIGINAL_DST_SAVED_REQUESTS_VERIFIED_NOT_AUDIO",
        "model": model,
        "original_requests_verified": len(rows),
        "finish_reasons": dict(finishes),
        "independent_score": score,
        "new_model_calls": 0,
        "source_checkbox_eligible": False,
    }
    grade_log = run_root / "score-private.log"
    if grade_log.is_file():
        match = re.search(r"JGA:\s*([0-9.]+)", grade_log.read_text(encoding="utf-8"))
        reported = float(match.group(1)) if match else None
        report["official_reported_jga"] = reported
        report["independent_grade_agrees"] = (
            not score["parse_failures"]
            and reported is not None
            and abs(reported - score["correct_turns"] / len(rows)) < 1e-12
        )
    return report, rows


def execute(
    data_root,
    author_root,
    output,
    endpoint,
    model,
    thinking=None,
    *,
    sender=post,
    prepared_root=None,
):
    endpoint = validate_endpoint(endpoint.rstrip("/"))
    author_root = verify_author(author_root)
    prompts = expected_prompts(selected_originals(data_root), author_root)
    output = Path(output).resolve()
    protected = [
        Path(data_root).resolve(),
        author_root,
        Path(__file__).resolve().parents[1],
    ]
    if any(
        output.is_relative_to(path) or path.is_relative_to(output) for path in protected
    ):
        raise ValueError("fresh external output must not overlap source roots")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    if prepared_root is None:
        prepared_root = output / "preparation"
        prepare_inputs(data_root, author_root, prepared_root)
    prepared_file = Path(prepared_root) / "author/prompt_test_output_cleangpt.jsonl"
    prepared = [
        json.loads(line)
        for line in prepared_file.read_text(encoding="utf-8").splitlines()
    ]
    if prepared != prompts:
        raise ValueError(
            "original prepared prompts/references differ before model calls"
        )
    prompts = prepared
    failures = []
    for index, prompt in enumerate(prompts):
        body = request_body(prompt, model, thinking)
        (output / f"{index:03d}-request-private.json").write_text(
            json.dumps(body), encoding="utf-8"
        )
        try:
            response = sender(endpoint + "/chat/completions", body, timeout=600)
            (output / f"{index:03d}-response-private.json").write_text(
                json.dumps(response), encoding="utf-8"
            )
        except (OSError, ValueError) as error:
            failure = {
                "basename": prompt["basename"],
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            (output / f"{index:03d}-failure-private.json").write_text(
                json.dumps(failure), encoding="utf-8"
            )
    if failures:
        raise RuntimeError(
            "original request failures retained; no filtered score or replacement"
        )
    report, rows = verify_saved(data_root, author_root, output, model, thinking)
    scoring = output / "score-inputs"
    scoring.mkdir()
    for name, content in (
        ("chat_val_results.jsonl", rows),
        ("prompt_test_output_cleangpt.jsonl", prompts),
    ):
        (scoring / name).write_text(
            "".join(json.dumps(row) + "\n" for row in content), encoding="utf-8"
        )
    result = subprocess.run(
        [sys.executable, str(author_root / "eval_JGA.py")],
        cwd=scoring,
        capture_output=True,
        text=True,
        timeout=60,
    )
    (output / "score-private.log").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    report, _ = verify_saved(data_root, author_root, output, model, thinking)
    report.update(
        status="ORIGINAL_DST_EXECUTED_AND_SCORED_NOT_AUDIO",
        official_scorer_exit_code=result.returncode,
        actual_model_calls=len(prompts),
    )
    (output / "qualification-public.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if result.returncode or not report.get("independent_grade_agrees"):
        raise RuntimeError(
            "original scoring failed or independent grade disagrees; results preserved"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--thinking", choices=["default", "true", "false"], default="default"
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:19468/v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        help="Reuse original public preparation; otherwise execute it once",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-saved", type=Path)
    args = parser.parse_args()
    thinking = None if args.thinking == "default" else args.thinking == "true"
    if args.verify_saved:
        report, _ = verify_saved(
            args.data_root, args.author_root, args.verify_saved, args.model, thinking
        )
    elif args.execute:
        if args.output is None:
            parser.error("fresh --output required")
        report = execute(
            args.data_root,
            args.author_root,
            args.output,
            args.endpoint,
            args.model,
            thinking,
            prepared_root=args.prepared_root,
        )
    else:
        prompts = expected_prompts(selected_originals(args.data_root), args.author_root)
        report = {
            "status": "PLAN_ONLY_NO_MODEL_CALLS",
            "original_requests": len(prompts),
            "model": args.model,
            "model_calls": 0,
        }
    print(json.dumps(report, indent=2))
    return int(
        report.get("independent_grade_agrees") is False
        or bool(report.get("independent_score", {}).get("parse_failures"))
    )


if __name__ == "__main__":
    raise SystemExit(main())
