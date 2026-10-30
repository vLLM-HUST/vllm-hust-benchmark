"""Invoke the pinned publisher's open-answer grader; plan-only by default.

Input is the publisher grade_outputs.py CSV contract, not task trajectories.
The publisher overwrites its input CSV; only a new private copy is passed to it.
No model service, downloader, question transformation or paid endpoint is started.
"""

import argparse
import asyncio
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def validate(args):
    endpoint = urlsplit(args.api_base)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
        or endpoint.username
        or endpoint.password
        or endpoint.path.rstrip("/") != "/v1"
        or endpoint.port is None
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("owned loopback OpenAI endpoint required")
    if not args.model.startswith("openai/") or not args.model.removeprefix("openai/"):
        raise ValueError("explicit local OpenAI-compatible judge binding required")
    author, original, output = (
        Path(value).resolve()
        for value in (args.author_root, args.input_file, args.output)
    )
    if output.exists() or any(
        output.is_relative_to(source) or source.is_relative_to(output)
        for source in (author, original)
    ):
        raise ValueError("fresh external scoring output required")
    return author, original, output


async def execute(args, report):
    author, original, output = validate(args)
    from prepare_jsonschemabench_author_source import verify_source

    pins = ROOT / "configs/workload-native/bixbench-author-source.json"
    verify_source(author, pins)
    with original.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"question", "target", "predicted"}.issubset(reader.fieldnames or ()):
            raise ValueError("publisher question/target/predicted CSV fields required")
        rows = list(reader)
    if not rows:
        raise ValueError("empty input cannot establish native scoring")
    # Preserve publisher defaults, including evaluation_mode and matching rules.
    # The input and original author source are never passed as writable outputs.
    output.mkdir(mode=0o700)
    copied = output / "graded.private.csv"
    shutil.copyfile(original, copied)
    os.chmod(copied, 0o600)
    sys.path.insert(0, str(author))
    os.environ["OPENAI_API_KEY"] = (
        "local-owned-no-paid-provider"  # pragma: allowlist secret - noncredential for owned loopback service
    )
    spec = importlib.util.spec_from_file_location(
        "bixbench_pinned_original_grade_outputs", author / "grade_outputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report.update(status="RUNNING_ORIGINAL_OPENANSWER_GRADER", input_rows=len(rows))
    receipt = output / "state.private.json"
    receipt.write_text(json.dumps(report, indent=2))
    try:
        metrics = await module.grade_answers(
            copied,
            module.AnswerMode.openanswer,
            model_name=args.model,
            temperature=1.0,
            api_base=args.api_base,
        )
        verify_source(author, pins)
        (output / "metrics.private.json").write_text(json.dumps(metrics, indent=2))
        report.update(
            status="ORIGINAL_OPENANSWER_SCORING_RETURNED", scoring_returned=True
        )
    except BaseException as error:
        report.update(status="SCORING_FAILED_RETAINED", error_type=type(error).__name__)
        raise
    finally:
        receipt.write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("author-root", "input-file", "output", "model", "api-base"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    validate(args)
    report = {
        "source_id": "BIXBENCH",
        "status": "PLAN_ONLY",
        "entry": "publisher grade_outputs.grade_answers",
        "answer_mode": "openanswer",
        "temperature": 1.0,
        "actual_judge_binding": args.model,
        "author_judge_reproduction": False,
        "source_checkbox_eligible": False,
        "scoring_returned": False,
        "boundary": "A returned local-judge score is not official-model accuracy reproduction or proof that native Agent generation and research consumption completed.",
    }
    if args.execute:
        report = asyncio.run(execute(args, report))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
