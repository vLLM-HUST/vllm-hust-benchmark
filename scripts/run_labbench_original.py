"""Run pinned LAB-Bench public MCQ tasks through its unchanged official harness.

External publisher archive/source and isolated publisher environment are required.
No task data, images or generated outputs belong in this repository. The local
OpenAI-compatible endpoint must use an NPU; this client does not load a model.
Declared model/budget are a connectivity profile, not the paper's model baseline.
"""

import argparse
import asyncio
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import sys
from urllib.parse import urlparse
from zipfile import ZipFile

REVISION = "998a8e0a40cf116c80e1b0e7a805ebb5fb9fa838"  # pragma: allowlist secret - public publisher Git revision
ARCHIVE_SHA256 = "ad737c75dfd7e12771ad1d5028a4e9b9f9d17a7017451c2639535b7314d5cdb3"  # pragma: allowlist secret - original public archive checksum
CATEGORIES = (
    "TableQA",
    "ProtocolQA",
    "FigQA",
    "LitQA2",
    "SeqQA",
    "DbQA",
    "SuppQA",
    "CloningScenarios",
)


def local_endpoint(value):
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("explicit loopback HTTP /v1 endpoint required")
    return value.rstrip("/")


def verify_source(source, archive, *, expected_sha=ARCHIVE_SHA256, expected_files=553):
    source, archive = Path(source).resolve(), Path(archive).resolve()
    with archive.open("rb") as stream:
        actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("publisher archive identity mismatch; re-audit, do not bypass")
    with ZipFile(archive) as zipped:
        entries = [item for item in zipped.infolist() if not item.is_dir()]
        if len(entries) != expected_files:
            raise ValueError("publisher file count mismatch")
        prefix = entries[0].filename.split("/")[0] + "/"
        names = set()
        for item in entries:
            name = item.filename.removeprefix(prefix)
            relative = PurePosixPath(name)
            if (
                not item.filename.startswith(prefix)
                or relative.is_absolute()
                or ".." in relative.parts
                or name in names
            ):
                raise ValueError("unsafe or duplicate publisher archive path")
            names.add(name)
            target = source / name
            if (
                target.is_symlink()
                or not target.resolve().is_relative_to(source)
                or not target.is_file()
                or target.read_bytes() != zipped.read(item)
            ):
                raise ValueError("original publisher file missing or modified")
        if any(
            str(path.relative_to(source)).replace("\\", "/") not in names
            for path in source.rglob("*.py")
        ):
            raise ValueError("unexpected executable Python file in publisher source")
    return len(entries)


def select_instances(instances, per_subset):
    if per_subset is not None and (type(per_subset) is not int or per_subset <= 0):
        raise ValueError("positive fixed sample count required")
    counts = defaultdict(int)
    selected = []
    for subset, instance in instances:
        if per_subset is None or counts[subset] < per_subset:
            selected.append((subset, instance))
            counts[subset] += 1
    if not selected:
        raise ValueError("no original tasks selected")
    return selected


async def execute(args):
    verified = verify_source(args.source_root, args.archive)
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(args.source_root.resolve()):
        raise ValueError("fresh output outside publisher source required")
    output.mkdir(mode=0o700, parents=True)
    os.environ["PUBLIC_RELEASE"] = "True"
    os.environ["OPENAI_BASE_URL"] = local_endpoint(args.endpoint)
    os.environ["OPENAI_API_KEY"] = (
        "local-npu-no-external-credentials"  # pragma: allowlist secret - dummy loopback-only SDK value, not a credential
    )
    sys.path.insert(0, str(args.source_root.resolve()))
    import labbench

    if (
        Path(labbench.__file__).resolve()
        != args.source_root.resolve() / "labbench/__init__.py"
    ):
        raise ValueError("imported a different publisher harness")
    receipt = {
        "source_id": "LABBENCH",
        "revision": REVISION,
        "status": "STARTED_NOT_DELIVERED",
        "verified_publisher_files": verified,
        "model": args.model,
        "declared_output_budget": args.max_tokens,
        "thinking_override": False
        if getattr(args, "disable_thinking", False)
        else None,
        "publisher_model_baseline_reproduced": False,
        "original_cot_template": True,
        "sample_seed": args.seed,
        "per_subset": args.per_subset,
        "categories": [],
        "source_checkbox_eligible": False,
        "performance_claim": False,
    }

    def save():
        (output / "receipt-private.json").write_text(json.dumps(receipt, indent=2))

    save()
    try:
        for category in args.categories:
            evaluator = labbench.Evaluator(labbench.Eval(category))
            selected = select_instances(evaluator.eval_set.instances, args.per_subset)
            evaluator.eval_set.instances = selected
            profile = {
                "category": category,
                "selected_ids": [str(item.id) for _, item in selected],
                "status": "SELECTED_BEFORE_MODEL_CALLS",
                "model_calls": 0,
            }
            receipt["categories"].append(profile)
            save()
            random.seed(args.seed)
            model_options = {"model": args.model, "max_tokens": args.max_tokens}
            if getattr(args, "disable_thinking", False):
                model_options["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            agent = labbench.OpenAIZeroShotAgent(
                use_cot=True,
                open_answer=False,
                model_kwargs=model_options,
            )
            agent.client = agent.client.with_options(
                timeout=args.timeout, max_retries=0
            )
            original_create = agent.client.chat.completions.create

            async def capture(**kwargs):
                index = profile["model_calls"]
                profile["model_calls"] += 1
                attempt = {"request": kwargs}
                path = output / f"{category}-attempt-{index:05d}.private.json"
                path.write_text(json.dumps(attempt))
                save()
                try:
                    response = await original_create(**kwargs)
                    attempt["response"] = response.model_dump()
                    return response
                except Exception as exc:
                    attempt["error_type"] = type(exc).__name__
                    raise
                finally:
                    path.write_text(json.dumps(attempt))

            agent.client.chat.completions.create = capture
            results = await evaluator.score_agent(agent.run_task, n_threads=1)
            tasks = {}
            for task_id, result in results["results"].items():
                tasks[str(task_id)] = {
                    **result,
                    "instance": result["instance"].model_dump(),
                    "input": result["input"].model_dump(),
                }
            (output / f"{category}-results.private.json").write_text(
                json.dumps({**results, "results": tasks}, default=str)
            )
            (output / f"{category}-agent-buffer.private.json").write_text(
                json.dumps(agent.task_buffer, default=str)
            )
            profile.update(
                status="OFFICIAL_SCORING_COMPLETED_PENDING_SAVED_AUDIT",
                metrics=results["metrics_all"],
            )
            save()
        verify_source(args.source_root, args.archive)
        receipt["status"] = "FIXED_ORIGINAL_TASKS_SCORED_PENDING_DELIVERY_AUDIT"
    except Exception as exc:
        receipt.update(status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__)
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", type=local_endpoint, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Explicit local vLLM model-template profile only; original publisher CoT prompt remains unchanged",
    )
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument(
        "--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES)
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--per-subset", type=int)
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.max_tokens <= 0 or not 1 <= args.timeout <= 3600:
        parser.error("positive generation budget and timeout1..3600 required")
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
