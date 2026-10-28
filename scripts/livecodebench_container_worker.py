"""Original LiveCodeBench scoring worker; only inside an offline nonroot container.

No network or model calls. Original private-test pickle decoding and all generated
code execution stay inside the qualified sandbox, never the host process.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
from types import SimpleNamespace

AUTHOR_SHA = "56c7c42aee979def86b69547ecd6d5a877a0e4940af8923bf4325ca0defecbaf"  # pragma: allowlist secret - fixed official source archive identity
AUTHOR_ROOT = "LiveCodeBench-28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"


def loopback_only_network(table):
    interfaces = {
        line.split(":", 1)[0].strip() for line in table.splitlines()[2:] if ":" in line
    }
    return interfaces == {"lo"}


def require_isolated_worker():
    if (
        sys.platform != "linux"
        or os.geteuid() == 0
        or not Path("/.dockerenv").is_file()
        or not loopback_only_network(Path("/proc/net/dev").read_text())
    ):
        raise RuntimeError("offline nonroot Docker worker required; no host fallback")


def author_checkout(archive, destination):
    archive, destination = Path(archive), Path(destination)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != AUTHOR_SHA:
        raise ValueError("fixed original author archive identity mismatch")
    if destination.exists():
        raise ValueError("fresh isolated author directory required")
    destination.mkdir()
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0] != AUTHOR_ROOT
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError("unsafe original archive member")
            if member.isfile() and "lcb_runner" in name.parts:
                target = destination.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.extractfile(member).read())
    root = destination / AUTHOR_ROOT
    sys.path.insert(0, str(root))
    os.chdir(root)
    return root


def outputs_from_saved(saved, *, question_id, messages, model):
    if saved.get("question_id") != question_id:
        raise ValueError("original question relationship mismatch")
    expected = {
        "model": model,
        "messages": messages,
        "n": 10,
        "max_tokens": 2000,
        "temperature": 0.2,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    body = dict(saved["request"])
    thinking = body.pop("chat_template_kwargs", None)
    if thinking not in (None, {"enable_thinking": False}) or body != expected:
        raise ValueError("original prompt or generation budget mismatch")
    choices = saved["response"]["choices"]
    if len(choices) != 10 or sorted(c["index"] for c in choices) != list(range(10)):
        raise ValueError("complete original ten-candidate response required")
    ordered = sorted(choices, key=lambda c: c["index"])
    outputs = [c["message"]["content"] for c in ordered]
    if any(not isinstance(text, str) or not text.strip() for text in outputs):
        raise ValueError("original empty completions cannot be replaced")
    return outputs


def prepare_prompts(args):
    require_isolated_worker()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("fresh isolated output required")
    output.mkdir(mode=0o700, parents=True)
    author_checkout(args.author_archive, Path("/tmp/verified-lcb-author"))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.lm_styles import LanguageModel, LMStyle
    from lcb_runner.prompts.code_generation import format_prompt_generation
    from lcb_runner.runner.oai_runner import OpenAIRunner

    defaults = SimpleNamespace(
        model=args.model,
        use_cache=False,
        n=10,
        temperature=0.2,
        top_p=0.95,
        max_tokens=2000,
        openai_timeout=90,
    )
    model = LanguageModel(args.model, args.model, LMStyle.OpenAIChat, None)
    runner = OpenAIRunner(defaults, model)
    rows = json.loads(Path(args.tasks).read_text("utf-8"))
    requests = []
    for raw in rows:
        task = CodeGenerationProblem(**raw)
        body = {
            key: value
            for key, value in runner.client_kwargs.items()
            if key != "timeout"
        }
        body["messages"] = format_prompt_generation(task, model.model_style)
        requests.append({"question_id": task.question_id, "request": body})
    if not requests or len({row["question_id"] for row in requests}) != len(requests):
        raise ValueError("distinct original task requests required")
    (output / "requests-private.json").write_text(
        json.dumps(requests, ensure_ascii=False), encoding="utf-8"
    )
    receipt = {
        "source_id": "LIVECODEBENCH",
        "status": "ORIGINAL_API_PROMPTS_PREPARED_NO_MODEL_CALLS",
        "selected_question_ids": [row["question_id"] for row in requests],
        "original_prompt_style": "OpenAIChat generic API template",
        "model": args.model,
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "native_scoring_complete": False,
    }
    (output / "receipt-private.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt))


def score(args):
    require_isolated_worker()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("fresh isolated output required")
    output.mkdir(mode=0o700, parents=True)
    author_checkout(args.author_archive, Path("/tmp/verified-lcb-author"))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation import extract_instance_results
    from lcb_runner.lm_styles import LanguageModel, LMStyle
    from lcb_runner.prompts.code_generation import format_prompt_generation
    from lcb_runner.runner.scenario_router import combine_results, get_metrics
    from lcb_runner.utils.scenarios import Scenario

    raw = json.loads(Path(args.tasks).read_text("utf-8"))
    tasks = [CodeGenerationProblem(**row) for row in raw]
    if not tasks or len({task.question_id for task in tasks}) != len(tasks):
        raise ValueError("distinct original tasks required")
    model = LanguageModel(args.model, args.model, LMStyle.OpenAIChat, None)
    results = []
    for index, task in enumerate(tasks):
        saved = json.loads(
            (
                Path(args.generations) / f"{index}.request-response.private.json"
            ).read_text("utf-8")
        )
        results.append(
            outputs_from_saved(
                saved,
                question_id=task.question_id,
                messages=format_prompt_generation(task, model.model_style),
                model=args.model,
            )
        )
    combined = combine_results(Scenario.codegeneration, results, model)
    defaults = SimpleNamespace(
        scenario=Scenario.codegeneration, num_process_evaluate=12, timeout=6
    )
    metrics = get_metrics(Scenario.codegeneration, defaults, tasks, combined)
    graded = extract_instance_results(metrics[1])
    if not (len(tasks) == len(combined) == len(graded) == len(metrics[2])):
        raise ValueError(
            "incomplete original scoring relationships; do not truncate zip"
        )
    evaluation = [
        task.insert_output_evaluation(outputs, code, grades, metadata=meta)
        for task, (outputs, code), grades, meta in zip(
            tasks, combined, graded, metrics[2]
        )
    ]
    (output / "original-metrics-private.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (output / "original-eval-all-private.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    receipt = {
        "source_id": "LIVECODEBENCH",
        "status": "FIXED_ORIGINAL_CODEGEN_SAMPLE_SCORED_OTHER_SCENARIOS_PENDING",
        "selected_question_ids": [task.question_id for task in tasks],
        "candidate_counts": [len(row["graded_list"]) for row in evaluation],
        "original_test_counts": [
            len(task.public_test_cases) + len(task.private_test_cases) for task in tasks
        ],
        "num_process_evaluate": 12,
        "timeout_seconds": 6,
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "all_four_scenarios_complete": False,
    }
    (output / "receipt-private.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-archive", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prepare-prompts", action="store_true")
    args = parser.parse_args()
    prepare_prompts(args) if args.prepare_prompts else score(args)


if __name__ == "__main__":
    main()
