"""Run three original MATH-500 tasks against an existing OpenAI-compatible service.

The runner uses the pinned source adapter and preparation consumer. It keeps raw
prompts, references, and model outputs private and records only sample identity,
termination, token counts, and the deterministic upstream PRM800K answer grade.
It does not reproduce the HuggingFaceH4 Inspect AI four-epoch/model-graded recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

SOURCE_ID = "MATH500"
SOURCE_REVISION = "7ecc794703b2877f63226f2477a49b34f9b25163"  # pragma: allowlist secret - public upstream revision
DATA_PATH = "prm800k/math_splits/test.jsonl"
DATA_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"  # pragma: allowlist secret - public source checksum
GRADER_HASHES = {
    "grading/grader.py": "9e8bbb6f504ee0d8068e1eca031d78174f1c925cfec391996fdf45c21bb016a9",  # pragma: allowlist secret - upstream grader checksum
    "grading/math_normalize.py": "13998bf8c35bdc8a76868c84e1704a78567e2552557fda69027cc186568d3a4d",  # pragma: allowlist secret - upstream normalizer checksum
}
PROMPT_PREFIX = (
    "Solve the following math problem efficiently and clearly. The last line "
    "of your response should be of the following format: 'Therefore, the final "
    "answer is: $\\boxed{ANSWER}$. I hope it is correct' (without quotes) "
    "where ANSWER is just the final number or expression that solves the "
    "problem. Think step by step before answering."
)


def extract_boxed_answer(text: str) -> str | None:
    """Return the innermost payload of the last balanced LaTeX \boxed{...}."""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    index = start + len(marker)
    depth = 1
    answer_start = index
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                answer = text[answer_start:index].strip()
                return answer or None
        index += 1
    return None


def build_prompt(problem: str) -> str:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("MATH500 original problem must be nonempty text")
    return f"{PROMPT_PREFIX}\n\n{problem}"


def select_task_records(records, selected_ids: list[str] | None = None):
    if selected_ids is None:
        return list(records)
    if not selected_ids or any(not value.strip() for value in selected_ids):
        raise ValueError("explicit task IDs must be nonempty")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("explicit task IDs must be distinct")
    wanted = set(selected_ids)
    selected = [
        record
        for record in records
        if record.replay["annotations"]["unique_id"] in wanted
    ]
    if {record.replay["annotations"]["unique_id"] for record in selected} != wanted:
        raise ValueError("requested task IDs are not in the fixed first-three sample")
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_grader(grader_root: Path):
    root = grader_root.resolve()
    for relative, expected in GRADER_HASHES.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("pinned OpenAI PRM800K grader file identity mismatch")
    sys.path.insert(0, str(root))
    try:
        from grading.grader import grade_answer
    except ImportError as error:
        raise RuntimeError(
            "official grader dependencies are missing in this isolated Python environment"
        ) from error
    return grade_answer


def request_completion(endpoint: str, payload: dict, *, timeout: int) -> dict:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"model endpoint returned HTTP {error.code}") from None
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(
            f"model endpoint request failed: {type(error).__name__}"
        ) from None
    if not isinstance(result, dict):
        raise ValueError("model endpoint returned a non-object response")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--grader-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--endpoint", required=True, help="Full /v1/completions URL")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--task-ids",
        help="Optional comma-separated subset of the fixed first-three source IDs, for failure recovery only",
    )
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.endpoint.endswith("/v1/completions"):
        raise ValueError(
            "use the original text completion interface, not chat conversion"
        )
    if min(args.max_model_len, args.max_tokens, args.timeout) <= 0:
        raise ValueError("model length, output budget, and timeout must be positive")
    if not args.tokenizer.is_dir():
        raise ValueError("an explicit local tokenizer directory is required")

    from transformers import AutoTokenizer
    from vllm_hust_workload_data import iter_workload
    from vllm_hust_workload_data.bindings import bind_source
    from vllm_hust_workload_data.consumer import resolve_input
    from vllm_hust_workload_data.pipeline import prepared_rows, run

    data_file = args.data_root / DATA_PATH
    if not data_file.is_file() or sha256(data_file) != DATA_SHA256:
        raise ValueError("pinned original MATH500 data file identity mismatch")
    grader = load_official_grader(args.grader_root)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=True
    )
    binding = bind_source(
        SOURCE_ID,
        data_root=args.data_root,
        files=[DATA_PATH],
        revision=SOURCE_REVISION,
    )

    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest_path = args.output / "binding.json"
    manifest_path.write_text(json.dumps(binding, indent=2), encoding="utf-8")
    prep = run(
        {
            "sources": [
                {
                    "source_id": SOURCE_ID,
                    "data_root": str(args.data_root.resolve()),
                    "manifest": str(manifest_path.resolve()),
                    "limit": 3,
                }
            ]
        },
        output=args.output / "prepared",
    )
    database = args.output / "prepared/workload.sqlite"
    refs = list(prepared_rows(database))
    source_rows = list(
        iter_workload(SOURCE_ID, data_root=args.data_root, manifest=binding, limit=3)
    )
    if len(refs) != 3 or len(source_rows) != 3:
        raise ValueError("exactly the first three original MATH500 tasks are required")

    resolved = [
        resolve_input(database, row["record_id"], data_root=args.data_root)
        for row in refs
    ]
    for expected, actual in zip(source_rows, resolved):
        if actual.input_record() != expected.input_record():
            raise ValueError(
                "prepared consumer input differs from the original adapter"
            )
        if actual.outcome_record() != expected.outcome_record():
            raise ValueError(
                "prepared consumer reference differs from the source adapter"
            )
        if set(actual.task["input"]) != {"problem"}:
            raise ValueError("only the original problem may enter the model")
    selected_ids = (
        None
        if args.task_ids is None
        else [value.strip() for value in args.task_ids.split(",")]
    )
    model_records = select_task_records(resolved, selected_ids)

    sample_receipts = []
    receipt = {
        "schema_version": "math500-original-npu-sample/v1",
        "source_id": SOURCE_ID,
        "upstream_repository": "openai/prm800k",
        "upstream_revision": SOURCE_REVISION,
        "file": DATA_PATH,
        "file_sha256": DATA_SHA256,
        "mirror_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "selected_tasks": [
            record.replay["annotations"]["unique_id"] for record in model_records
        ],
        "selection": "first three original held-out MATH test tasks; fixed before model calls",
        "model": args.model,
        "server_protocol": "OpenAI-compatible raw text completions; no chat conversion",
        "prompt_recipe": "pinned HuggingFaceH4/MATH-500 eval.yaml prompt_template; original problem only",
        "scorer": "pinned OpenAI PRM800K deterministic grade_answer",
        "scorer_code_sha256": GRADER_HASHES,
        "scorer_difference": "HuggingFaceH4 eval.yaml uses Inspect AI model_graded_fact with four pass_at_1 epochs; this three-task smoke instead uses the deterministic upstream PRM800K symbolic grader and is not an exact reproduction of that recipe.",
        "runner_sha256": sha256(Path(__file__).resolve()),
        "generation": {
            "temperature": 0,
            "seed": 0,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "retries": 0,
        },
        "reader_validation": {
            "source_records": len(source_rows),
            "prepared_records": len(refs),
            "resolved_records": len(resolved),
            "selected_model_tasks": len(model_records),
            "answer_and_solution_excluded_from_input": True,
            "code_sha256": prep["code_sha256"],
        },
        "samples": sample_receipts,
        "status": "STARTED",
        "claim_boundary": "Three original MATH500 tasks only; not full-suite accuracy, exact HuggingFaceH4 Inspect recipe reproduction, or a performance result.",
        "raw_prompts_references_and_model_outputs_published": False,
    }
    receipt_path = args.output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    try:
        for index, record in enumerate(model_records):
            prompt = build_prompt(record.task["input"]["problem"])
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if prompt_tokens + args.max_tokens > args.max_model_len:
                raise ValueError(
                    "original task plus output budget exceeds model context"
                )
            result = request_completion(
                args.endpoint,
                {
                    "model": args.model,
                    "prompt": prompt,
                    "temperature": 0,
                    "seed": 0,
                    "max_tokens": args.max_tokens,
                },
                timeout=args.timeout,
            )
            choices = result.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("model endpoint did not return exactly one completion")
            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            text = choice.get("text")
            if not isinstance(text, str):
                raise ValueError("model endpoint completion text is missing")
            extracted = extract_boxed_answer(text) if finish_reason == "stop" else None
            outcome = record.outcome_record() or {}
            correct = (
                bool(grader(extracted, outcome["answer"]))
                if extracted is not None and finish_reason == "stop"
                else False
            )
            usage = result.get("usage") or {}
            sample_receipts.append(
                {
                    "unique_id": record.replay["annotations"]["unique_id"],
                    "subject": record.replay["annotations"]["subject"],
                    "level": record.replay["annotations"]["level"],
                    "prompt_tokens_local": prompt_tokens,
                    "prompt_tokens_server": usage.get("prompt_tokens"),
                    "completion_tokens_server": usage.get("completion_tokens"),
                    "finish_reason": finish_reason,
                    "terminal": finish_reason == "stop",
                    "boxed_answer_extracted": extracted is not None,
                    "official_grade_correct": correct,
                }
            )
            receipt["status"] = "RUNNING"
            receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        receipt["status"] = (
            "END_TO_END_EXECUTED_AND_SCORED"
            if len(sample_receipts) == len(model_records)
            and all(item["terminal"] for item in sample_receipts)
            else "INCOMPLETE_GENERATION"
        )
    except Exception as error:
        receipt["status"] = "FAILED_EXECUTION"
        receipt["failure_type"] = type(error).__name__
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        raise
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "task_count": len(sample_receipts),
                "all_terminal": all(item["terminal"] for item in sample_receipts),
                "scored_correct": sum(
                    item["official_grade_correct"] for item in sample_receipts
                ),
                "receipt": str(receipt_path),
            }
        )
    )
    return 0 if receipt["status"] == "END_TO_END_EXECUTED_AND_SCORED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
