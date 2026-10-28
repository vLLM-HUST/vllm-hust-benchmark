"""Fixed original JSONSchemaBench bench/evaluator with a local NPU API backend.

Requires a clean pinned official checkout, privately acquired original snapshot,
and a separate compatible official client environment. No downloads, schema
rewrites, remote model fallback, task replacement or model deployment.
"""

import argparse
from dataclasses import asdict
from enum import Enum
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


def reusable_prefix(root, expected, *, failed_tail_terminal=False):
    """Reuse only a terminal failed run's complete, matching successful prefix."""
    if not failed_tail_terminal:
        raise ValueError(
            "independently confirm failed request terminal; a receipt alone is insufficient"
        )
    root = Path(root).resolve(strict=True)
    receipt_path = root / "receipt-public.json"
    previous = json.loads(receipt_path.read_text(encoding="utf-8"))
    if previous.get("status") != "FAILED_NATIVE_FLOW_NOT_COMPLETE":
        raise ValueError("only a confirmed failed native flow can be resumed")
    for key in (
        "source_id",
        "source_revision",
        "dataset_revision",
        "dataset_pins_sha256",
        "tasks",
        "limit_per_task",
        "model",
        "max_tokens",
        "context_capacity",
    ):
        if previous.get(key) != expected.get(key):
            raise ValueError(f"resume original profile differs: {key}")
    if previous.get("origin", "http://127.0.0.1:19466") != expected.get("origin"):
        raise ValueError("resume local origin differs")
    if previous.get("client_http_timeout_seconds", 300) != expected.get(
        "client_http_timeout_seconds", 300
    ):
        raise ValueError("resume transport timeout differs")
    paths = sorted(root.glob("attempt-*.private.json"))
    attempted = previous["attempted_samples"]
    if [path.name for path in paths] != [
        f"attempt-{index:06d}.private.json" for index in range(attempted)
    ]:
        raise ValueError("resume attempt coverage differs")
    successful = []
    for index, path in enumerate(paths):
        raw = path.read_bytes()
        record = json.loads(raw)
        if record.get("status") == "GENERATION_RETURNED_NOT_YET_OFFICIALLY_EVALUATED":
            if len(successful) != index or not isinstance(record.get("output"), dict):
                raise ValueError(
                    "resume requires one contiguous complete output prefix"
                )
            if (
                record["output"].get("task") != record["task"]
                or record["output"].get("messages") != record["original_messages"]
                or record["output"].get("schema") != record["original_schema"]
            ):
                raise ValueError(
                    "saved output original input/reference relationship differs"
                )
            successful.append(
                {
                    "record": record,
                    "path": str(path),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        elif (
            record.get("status") != "GENERATION_FAILED_NOT_REPLACED"
            or index != len(paths) - 1
        ):
            raise ValueError("unresolved or nonterminal prior attempt, do not retry")
    if (
        len(successful) != previous["returned_samples"]
        or len(successful) != attempted - 1
    ):
        raise ValueError(
            "resume must preserve every prior returned output and one failed tail"
        )
    return successful, hashlib.sha256(receipt_path.read_bytes()).hexdigest()


def checkpoint_engine(
    engine,
    output_root,
    receipt,
    save,
    *,
    prior=(),
    restore=None,
    revalidate_rejections=(),
):
    if len(set(revalidate_rejections)) != len(revalidate_rejections) or any(
        type(i) is not int or i < 0 or i >= len(prior) for i in revalidate_rejections
    ):
        raise ValueError("revalidation indices must select unique saved prior attempts")
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
            if index < len(prior):
                saved = prior[index]
                previous = saved["record"]
                if (
                    previous["task"] != task
                    or previous["original_messages"] != messages
                    or previous["original_schema"] != schema
                    or restore is None
                ):
                    raise ValueError(
                        "saved original task/messages/schema differ; no model fallback"
                    )
                result = restore(previous["output"])
                if asdict(result) != previous["output"]:
                    raise ValueError(
                        "author dataclass restoration changed saved output"
                    )
                provenance = {
                    "path": saved["path"],
                    "sha256": saved["sha256"],
                }
                if index in revalidate_rejections:
                    old_output = previous["output"]
                    if (
                        old_output.get("metadata", {})
                        .get("compile_status", {})
                        .get("code")
                        != 1
                        or old_output.get("generated_tokens") != []
                        or old_output.get("generation") not in (None, "")
                    ):
                        raise ValueError(
                            "only empty prior compiler rejection can be revalidated"
                        )
                    record["revalidated_prior_compile_rejection"] = provenance
                    receipt["new_generation_attempts"] = (
                        receipt.get("new_generation_attempts", 0) + 1
                    )
                    receipt["revalidated_prior_compile_rejections"] = (
                        receipt.get("revalidated_prior_compile_rejections", 0) + 1
                    )
                    save()
                    result = generate(task, messages, schema)
                else:
                    if hasattr(engine, "total_usage") and hasattr(
                        result, "token_usage"
                    ):
                        engine.total_usage += result.token_usage
                    record["reused_from"] = provenance
                    receipt["reused_samples"] = receipt.get("reused_samples", 0) + 1
            else:
                receipt["new_generation_attempts"] = (
                    receipt.get("new_generation_attempts", 0) + 1
                )
                save()
                result = generate(task, messages, schema)
            record.update(
                status="GENERATION_RETURNED_NOT_YET_OFFICIALLY_EVALUATED",
                output=asdict(result),
            )
            receipt["returned_samples"] += 1
            return result
        except Exception as exc:
            record.update(
                status="GENERATION_FAILED_NOT_REPLACED",
                error_type=type(exc).__name__,
                error_message=str(exc),
                error_status_code=getattr(exc, "status_code", None),
            )
            body = getattr(exc, "body", None)
            if isinstance(body, (dict, list, str, int, float, bool)) or body is None:
                record["error_body"] = body
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
        "origin": args.origin,
        "model_backend": "OPERATOR_DECLARED_NPU_NOT_MEASURED_BY_CLIENT",
        "max_tokens": args.max_tokens,
        "context_capacity": args.context_capacity,
        "client_http_timeout_seconds": 300,
        "timing_profile_boundary": "Local HTTP stream transport timeout, not separate author compiler/decoder timing limits; no paper/backend-performance equivalence claim.",
        "source_checkbox_eligible": False,
        "performance_claim": False,
        "source_completion_note": "Actual native profile and research consumer require separate review; no completion from software support.",
    }
    prior = []
    resume_root = getattr(args, "reuse_failed_run", None)
    if resume_root is not None:
        prior, prior_receipt_sha = reusable_prefix(
            resume_root,
            receipt,
            failed_tail_terminal=getattr(args, "confirm_failed_tail_terminal", False),
        )
        receipt.update(
            reused_samples=0,
            reusable_prefix_samples=len(prior),
            previous_receipt_sha256=prior_receipt_sha,
            new_generation_attempts=0,
            reuse_boundary="Observed saved-file hashes, not retroactive historical signatures. Original output IDs, references and metadata retained; not continuous-run performance evidence.",
        )
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

        def restore(saved):
            from dacite import Config, from_dict

            return from_dict(
                data_class=types_module.GenerationOutput,
                data=saved,
                config=Config(cast=[Enum], strict=True),
            )

        engine = checkpoint_engine(
            engine,
            out,
            receipt,
            save,
            prior=prior,
            restore=restore,
            revalidate_rejections=getattr(
                args, "revalidate_prior_compile_rejection_indices", ()
            )
            or (),
        )
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
    parser.add_argument("--reuse-failed-run", type=Path)
    parser.add_argument(
        "--revalidate-prior-compile-rejection-indices",
        type=int,
        nargs="+",
        help="Explicit prior empty compiler rejections only, after declared backend change; never regenerate saved model output",
    )
    parser.add_argument(
        "--confirm-failed-tail-terminal",
        action="store_true",
        help="Operator verified prior request/process terminal independently; not inferred from receipt or timeout.",
    )
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
