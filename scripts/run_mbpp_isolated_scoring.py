"""Run pinned CodeEval on original MBPP inputs in existing Linux isolation.

No model calls, downloads, code repair, host execution fallback or automatic cleanup.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_mbpp_original_assets import PINS as ASSET_PINS
from prepare_mbpp_scoring_inputs import digest, validate_saved_generation
from prepare_xsum_original_assets import bound_path, verify
from run_humaneval_isolated_scoring import copy_packages, validate_runtime

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/mbpp-scoring-pins.json"


def original_inputs(prepared, generation, *, rows, examples, source_identity):
    """Re-use the exact saved-generation verifier; no author modules imported."""
    expected_tasks, provenance = validate_saved_generation(
        generation, rows=rows, examples=examples, source_identity=source_identity
    )
    prepared = Path(prepared).resolve()
    receipt_path, task_file = (
        prepared / "receipt-private.json",
        prepared / "tasks-private.json",
    )
    if receipt_path.is_symlink() or task_file.is_symlink():
        raise ValueError("prepared input symlinks are unsupported")
    receipt = json.loads(receipt_path.read_text("utf-8"))
    if (
        receipt.get("source_id") != "MBPP"
        or receipt.get("status") != "ORIGINAL_TASK_SCORING_INPUTS_PREPARED_NOT_EXECUTED"
        or receipt.get("source_identity") != source_identity
        or receipt.get("completion_postprocessing") != "NONE_EXACT_GENERATED_TEXT"
        or receipt.get("functional_tests_executed") is not False
        or receipt.get("tasks_sha256") != digest(task_file)
        or any(receipt.get(key) != value for key, value in provenance.items())
    ):
        raise ValueError("original prepared input receipt mismatch")
    tasks = json.loads(task_file.read_text("utf-8"))
    if tasks != expected_tasks or receipt.get("task_ids") != [
        item["task_id"] for item in expected_tasks
    ]:
        raise ValueError("original source/generation/prepared task parity mismatch")
    return tasks, task_file, receipt_path


def author_modules(author_root, pins):
    root = Path(author_root).resolve()
    result = {}
    if set(Path(path).name for path in pins["files"]) != {
        "__init__.py",
        "code_eval.py",
        "execute.py",
    }:
        raise ValueError("fixed evaluator module inventory required")
    for relative, expected in pins["files"].items():
        path = root / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or digest(path) != expected
        ):
            raise ValueError("unchanged pinned evaluator source required")
        result[Path(relative).name] = path
    return result


def namespace_runtime(prefix):
    runtime = validate_runtime(prefix)
    if any(runtime.is_relative_to(Path(root)) for root in ("/data", "/root", "/proc")):
        raise ValueError(
            "runtime bind path would expose a forbidden namespace root; use a dedicated accessible prefix such as /opt"
        )
    return runtime


def launch(args):
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError(
            "Linux operator-authorized root launcher required; no host fallback"
        )
    if not 1 <= args.sandbox_uid <= 2**31 - 2:
        raise ValueError("explicit nonzero sandbox host UID required")
    if any(not shutil.which(name) for name in ("setpriv", "unshare", "mount")):
        raise RuntimeError("namespace utilities missing; no host fallback")
    pins = json.loads(PINS.read_text("utf-8"))
    assets = json.loads(ASSET_PINS.read_text("utf-8"))
    root = args.assets.resolve()

    def verify_assets():
        for spec in assets["dataset"]["files"]:
            verify(bound_path(root, "data", spec), spec)

    verify_assets()
    import pyarrow.parquet as pq

    rows = pq.read_table(root / "data/full/test-00000-of-00001.parquet").to_pylist()
    demos = pq.read_table(root / "data/full/prompt-00000-of-00001.parquet").to_pylist()
    if len(demos) != 10 or {item["task_id"] for item in demos} != set(range(1, 11)):
        raise ValueError("complete original prompt split required")
    lookup = {item["task_id"]: item for item in demos}
    identity = {
        "repo_id": assets["dataset"]["repo_id"],
        "revision": assets["dataset"]["revision"],
        "asset_manifest_sha256": digest(ASSET_PINS),
        "files": assets["dataset"]["files"],
    }
    tasks, task_file, preparation = original_inputs(
        args.prepared_dir,
        args.generation,
        rows=rows,
        examples=[lookup[i] for i in (2, 3, 4)],
        source_identity=identity,
    )
    if args.deadline < 8 * len(tasks) + 60:
        raise ValueError(
            "outer deadline must allow unchanged reference and prediction timeouts"
        )
    modules = author_modules(args.author_root, pins)
    runtime = namespace_runtime(args.python_runtime)
    work = args.work_root.resolve()
    for source in (
        args.assets,
        args.generation,
        args.prepared_dir,
        args.author_root,
        args.package_root,
        runtime,
    ):
        source = source.resolve()
        if work.is_relative_to(source) or source.is_relative_to(work):
            raise ValueError("fresh work-root must not overlap input or runtime roots")
    task_sha = digest(task_file)
    work.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chown(work, args.sandbox_uid, args.sandbox_uid)
    bundle, sandbox = work / "assets", work / "root"
    for name in ("inputs", "author/metrics", "packages", "output"):
        (bundle / name).mkdir(parents=True)
    sandbox.mkdir()
    os.chown(sandbox, args.sandbox_uid, args.sandbox_uid)
    os.chown(bundle / "output", args.sandbox_uid, args.sandbox_uid)
    for name, source in modules.items():
        shutil.copyfile(source, bundle / "author/metrics" / name)
    shutil.copyfile(task_file, bundle / "inputs/tasks-private.json")
    shutil.copyfile(preparation, bundle / "inputs/preparation-receipt.json")
    shutil.copyfile(PINS, bundle / "inputs/scoring-pins.json")
    worker = ROOT / "scripts/mbpp_isolated_worker.py"
    namespace = ROOT / "scripts/humaneval_isolated_worker.py"
    shutil.copyfile(worker, bundle / "inputs/worker.py")
    shutil.copyfile(namespace, bundle / "inputs/humaneval_isolated_worker.py")
    copy_packages(args.package_root, bundle / "packages", pins)
    config = {
        "root": str(sandbox),
        "assets": str(bundle),
        "runtime": str(runtime),
        "uid": args.sandbox_uid,
    }
    command = [
        "setpriv",
        f"--reuid={args.sandbox_uid}",
        f"--regid={args.sandbox_uid}",
        "--clear-groups",
        "unshare",
        "--propagation",
        "unchanged",
        "--user",
        "--map-root-user",
        "--mount",
        "--net",
        "--pid",
        "--fork",
        "--kill-child",
        str(runtime / "bin/python3"),
        str(bundle / "inputs/worker.py"),
        "--inside",
        json.dumps(config),
    ]
    try:
        with (work / "scorer-private.log").open("x", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.deadline,
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "LANG": "C.UTF-8",
                },
            )
        if process.returncode:
            raise RuntimeError(
                "isolated scorer failed; preserve private log, no host fallback"
            )
        receipt = json.loads((bundle / "output/receipt-public.json").read_text("utf-8"))
        flags, metric = receipt.get("passed"), receipt.get("metric")
        if (
            receipt.get("status") != "SELECTED_ORIGINAL_TESTS_CODEEVAL_SCORED"
            or receipt.get("source_id") != "MBPP"
            or receipt.get("tasks_sha256") != task_sha
            or receipt.get("task_ids") != [item["task_id"] for item in tasks]
            or receipt.get("evaluator_revision") != pins["evaluator_revision"]
            or receipt.get("original_reference_checks_passed") is not True
            or receipt.get("isolation_checks_passed_before_import_and_execution")
            is not True
            or receipt.get("generated_code_executed") is not True
            or not isinstance(flags, list)
            or len(flags) != len(tasks)
            or any(type(flag) is not bool for flag in flags)
            or not isinstance(metric, dict)
            or set(metric) != {"pass@1"}
            or not isinstance(metric["pass@1"], (int, float))
            or not abs(metric["pass@1"] - sum(flags) / len(flags)) < 1e-12
        ):
            raise RuntimeError("isolated result identity or isolation proof mismatch")
        verify_assets()
        if digest(task_file) != task_sha:
            raise ValueError("original scoring input changed")
        receipt.update(
            worker_sha256=digest(worker),
            namespace_helper_sha256=digest(namespace),
            source_identity=identity,
            source_checkbox_eligible=False,
            full_test_claim=len(tasks) == 500,
        )
        (work / "receipt-public.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )
        return receipt
    except Exception as error:
        (work / "failure-public.json").write_text(
            json.dumps(
                {
                    "status": "INCOMPLETE",
                    "exception_type": type(error).__name__,
                    "source_checkbox_eligible": False,
                }
            ),
            encoding="utf-8",
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "assets",
        "generation",
        "prepared-dir",
        "author-root",
        "package-root",
        "python-runtime",
        "work-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sandbox-uid", type=int, required=True)
    parser.add_argument("--deadline", type=int, default=4200)
    args = parser.parse_args()
    print(json.dumps(launch(args)))


if __name__ == "__main__":
    main()
