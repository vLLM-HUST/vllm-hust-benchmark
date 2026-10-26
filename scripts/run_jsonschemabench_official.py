"""Fixed original JSONSchemaBench bench/evaluator with a local NPU API backend.

Requires a clean pinned official checkout, privately acquired original snapshot,
and a separate compatible official client environment. No downloads, schema
rewrites, remote model fallback, task replacement or model deployment.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from jsonschemabench_local_engine import create_engine
from prepare_jsonschemabench_original_assets import PINS
from prepare_jsonschemabench_author_source import verify_source
from prepare_xsum_original_assets import bound_path, verify

AUTHOR_REVISION = "8003e8405c4d8d8b327b1eb472c9856297d75493"  # pragma: allowlist secret - fixed public author revision


def validate_source(root, source_layout="git"):
    if source_layout == "publisher-files":
        return verify_source(root)
    if source_layout != "git":
        raise ValueError("unknown author source validation layout")
    root = Path(root).resolve(strict=True)

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    if (
        Path(git("rev-parse", "--show-toplevel")).resolve() != root
        or git("rev-parse", "HEAD") != AUTHOR_REVISION
        or git("status", "--porcelain", "--untracked-files=all")
    ):
        raise ValueError("clean fixed official checkout required; no source fallback")
    return root


def validate_snapshot(root, pins):
    root = Path(root).resolve(strict=True)
    for entry in pins["dataset"]["files"]:
        verify(bound_path(root, "", entry), entry)
    return root


def select_tasks(known, selected=None):
    tasks = (
        [name for name in known if name != "default"]
        if selected is None
        else list(selected)
    )
    if not tasks or len(tasks) != len(set(tasks)) or not set(tasks) <= set(known):
        raise ValueError("distinct official dataset configurations required")
    if "default" in tasks and len(tasks) != 1:
        raise ValueError("default aggregate must run separately; do not double-count")
    return tasks


def origin(value):
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("credential-free loopback HTTP origin required")
    return value.rstrip("/")


def checkpoint_engine(engine, output_root, receipt, save):
    generate = engine.generate

    def wrapped(task, messages, schema):
        index = receipt["attempted_samples"]
        receipt["attempted_samples"] += 1
        save()
        record = {
            "task": task,
            "original_messages": messages,
            "original_schema": schema,
            "status": "STARTED",
        }
        path = output_root / f"attempt-{index:06d}.private.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        try:
            result = generate(task, messages, schema)
            record.update(
                status="GENERATION_RETURNED_NOT_YET_OFFICIALLY_EVALUATED",
                output=asdict(result),
            )
            receipt["returned_samples"] += 1
            return result
        except Exception as exc:
            record.update(
                status="GENERATION_FAILED_NOT_REPLACED", error_type=type(exc).__name__
            )
            raise
        finally:
            path.write_text(json.dumps(record), encoding="utf-8")
            save()

    engine.generate = wrapped
    return engine


def run(args):
    pins = json.loads(PINS.read_text("utf-8"))
    source_layout = getattr(args, "source_layout", "git")
    source = validate_source(args.official_source, source_layout)
    snapshot = validate_snapshot(args.snapshot_root, pins)
    tasks = select_tasks(pins["config_names"], args.tasks)
    if not args.full and (type(args.limit) is not int or args.limit < 3):
        raise ValueError(
            "fixed original first >=3 per configuration or --full required"
        )
    if args.full and args.limit is not None:
        raise ValueError("--full cannot be combined with --limit")
    out = args.output.resolve()
    for root in (source, snapshot):
        if out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError("fresh output must not overlap original source/snapshot")
    os.umask(0o077)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "JSONSCHEMABENCH",
        "status": "STARTED_NOT_NATIVE_COMPLETE",
        "source_revision": AUTHOR_REVISION,
        "source_layout": source_layout,
        "dataset_revision": pins["dataset"]["revision"],
        "dataset_pins_sha256": hashlib.sha256(PINS.read_bytes()).hexdigest(),
        "tasks": tasks,
        "limit_per_task": None if args.full else args.limit,
        "attempted_samples": 0,
        "returned_samples": 0,
        "model": args.model,
        "model_backend": "OPERATOR_DECLARED_NPU_NOT_MEASURED_BY_CLIENT",
        "max_tokens": args.max_tokens,
        "context_capacity": args.context_capacity,
        "source_checkbox_eligible": False,
        "performance_claim": False,
        "source_completion_note": "Actual native profile and research consumer require separate review; no completion from software support.",
    }
    path = out / "receipt-public.json"

    def save():
        path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    save()
    old_directory = Path.cwd()
    old_path = list(sys.path)
    environment_keys = (
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "LITELLM_LOCAL_MODEL_COST_MAP",
        "LITELLM_TELEMETRY",
    )
    old_environment = {key: os.environ.get(key) for key in environment_keys}
    dataset, original_location = None, None
    client = None
    try:
        os.environ.update(
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            LITELLM_LOCAL_MODEL_COST_MAP="True",
            LITELLM_TELEMETRY="False",
        )
        if "core" in sys.modules or "engines" in sys.modules:
            raise RuntimeError("fresh client process required; core already imported")
        sys.path.insert(0, str(source))
        dataset = importlib.import_module("core.dataset")
        original_location = dataset.DATASET_HUGGINGFACE_PATH
        # Location-only override, using untouched original README/configs/files.
        dataset.DATASET_HUGGINGFACE_PATH = str(snapshot)
        counts = {
            task: len(dataset.Dataset(dataset.DatasetConfig(task))) for task in tasks
        }
        expected = {
            task: total if args.full else min(total, args.limit)
            for task, total in counts.items()
        }
        if not all(expected.values()):
            raise ValueError("original selected configuration is empty")
        receipt.update(
            original_test_rows_per_configuration=counts,
            expected_output_counts=expected,
            selection_rule="ORIGINAL_TEST_PREFIX_LIMIT_OR_FULL_NO_SHUFFLE",
        )
        save()
        bench = importlib.import_module("core.bench").bench
        engine_module = importlib.import_module("core.engine")
        types_module = importlib.import_module("core.types")
        from openai import OpenAI

        client = OpenAI(
            base_url=args.origin + "/v1",
            api_key="unused-local-service",  # pragma: allowlist secret - loopback dummy, not a credential
            max_retries=0,
            timeout=300,
        )

        def token_count(messages):
            request = Request(
                args.origin + "/tokenize",
                data=json.dumps(
                    {
                        "model": args.model,
                        "messages": messages,
                        "add_generation_prompt": True,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=60) as response:
                return json.load(response)["count"]

        engine = create_engine(
            engine_module=engine_module,
            types_module=types_module,
            client=client,
            token_count=token_count,
            model=args.model,
            max_tokens=args.max_tokens,
            context_limit=args.context_capacity,
            temperature=0,
        )
        engine = checkpoint_engine(engine, out, receipt, save)
        os.chdir(out)
        outputs = bench(
            engine=engine,
            tasks=tasks,
            limit=None if args.full else args.limit,
            save_outputs=True,
        )
        if (
            len(outputs) != len(tasks)
            or {task: len(group) for task, group in zip(tasks, outputs)} != expected
            or sum(len(group) for group in outputs) != receipt["returned_samples"]
        ):
            raise RuntimeError("official output coverage mismatch")
        validate_source(source, source_layout)
        validate_snapshot(snapshot, pins)
        receipt.update(
            status="ORIGINAL_BENCH_AND_EVALUATOR_RETURNED",
            output_counts={task: len(group) for task, group in zip(tasks, outputs)},
            original_formatter_and_evaluator_used=True,
            schema_rewriting=False,
            failed_tasks_replaced=False,
        )
    except Exception as exc:
        receipt.update(
            status="FAILED_NATIVE_FLOW_NOT_COMPLETE", error_type=type(exc).__name__
        )
        raise
    finally:
        save()
        os.chdir(old_directory)
        if client is not None:
            client.close()
        if dataset is not None and original_location is not None:
            dataset.DATASET_HUGGINGFACE_PATH = original_location
        sys.path[:] = old_path
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument(
        "--source-layout", choices=["git", "publisher-files"], default="git"
    )
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin", type=origin, default="http://127.0.0.1:19466")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--context-capacity", type=int, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--limit", type=int)
    selection.add_argument("--full", action="store_true")
    parser.add_argument("--tasks", nargs="+")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
