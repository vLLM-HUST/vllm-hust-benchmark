"""Score unchanged prepared HumanEval completions in Linux namespaces/chroot.

Requires an operator-authorized root process, unprivileged user namespaces and
a dedicated nonzero sandbox host UID. No inference or network acquisition.
Never falls back to host execution. All raw outputs stay in private work-root.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/humaneval-author-scoring-pins.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_file(root, relative, expected):
    root = Path(root).resolve()
    path = root / relative
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root)
        or digest(path) != expected
    ):
        raise ValueError("source identity mismatch; payload suppressed")
    return path


def original_inputs(author, prepared, pins):
    """Validate without importing author modules or executing candidate code."""
    author, prepared = Path(author).resolve(), Path(prepared).resolve()
    files = set(pins["files"])
    actual = {
        p.relative_to(author).as_posix() for p in (author / "human_eval").glob("*.py")
    }
    if files != actual:
        raise ValueError("original author module inventory mismatch")
    for name, expected in pins["files"].items():
        checked_file(author, name, expected)
    data = checked_file(author, "data/HumanEval.jsonl.gz", pins["data_sha256"])
    with gzip.open(data, "rt", encoding="utf-8") as stream:
        problems = [json.loads(line) for line in stream if line.strip()]
    receipt = json.loads((prepared / "receipt-private.json").read_text("utf-8"))
    samples = prepared / "samples-private.jsonl"
    if (
        receipt.get("status") != "OFFICIAL_SCORER_INPUT_PREPARED_NOT_EXECUTED"
        or receipt.get("source_id") != "HUMANEVAL"
        or receipt.get("source_revision") != pins["revision"]
        or receipt.get("completion_postprocessing") != "NONE_EXACT_ORIGINAL_TEXT"
        or digest(samples) != receipt.get("samples_sha256")
    ):
        raise ValueError("original prepared completion receipt mismatch")
    rows = [
        json.loads(line)
        for line in samples.read_text("utf-8").splitlines()
        if line.strip()
    ]
    ids = [row.get("task_id") for row in rows]
    expected = [row["task_id"] for row in receipt["historical_samples"]]
    selected = [row for row in problems if row["task_id"] in ids]
    if (
        len(ids) < 3
        or len(set(ids)) != len(ids)
        or ids != expected
        or [row["task_id"] for row in selected] != ids
        or any(
            set(row) != {"task_id", "completion"}
            or not isinstance(row["completion"], str)
            or not row["completion"]
            for row in rows
        )
    ):
        raise ValueError("fixed original selection or completion shape mismatch")
    return selected, samples


def validate_runtime(prefix):
    prefix = Path(prefix).resolve()
    if prefix in {
        Path(p).resolve()
        for p in ("/", "/usr", "/usr/local", "/data", "/root", "/home", "/workspace")
    }:
        raise ValueError("explicit dedicated Python runtime prefix required")
    if not (prefix / "bin/python3").is_file() or not (prefix / "lib").is_dir():
        raise ValueError("Python runtime layout unavailable")
    if prefix.is_relative_to(Path("/usr/lib")) or prefix.is_relative_to(
        Path("/usr/lib64")
    ):
        raise ValueError("runtime conflicts with read-only system library mounts")
    return prefix


def copy_packages(package_root, output, pins):
    package_root = Path(package_root).resolve()
    for name, version in pins["client_versions"].items():
        meta = package_root / f"{name}-{version}.dist-info/METADATA"
        if f"\nVersion: {version}\n" not in "\n" + meta.read_text("utf-8"):
            raise ValueError("pinned scoring dependency unavailable")
    for name in ("numpy", "numpy.libs", "tqdm"):
        source = package_root / name
        if source.is_symlink() or any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError(
                "dependency symlinks require separate operator qualification"
            )
        shutil.copytree(
            source, output / name, ignore=shutil.ignore_patterns("__pycache__")
        )


def launch(args):
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError(
            "Linux operator-authorized root launcher required; no host fallback"
        )
    if not 1 <= args.sandbox_uid <= 2**31 - 2 or args.deadline < 10:
        raise ValueError("nonzero sandbox UID and meaningful outer deadline required")
    for executable in ("setpriv", "unshare", "mount"):
        if not shutil.which(executable):
            raise RuntimeError("namespace utility unavailable; no host fallback")
    pins = json.loads(PINS.read_text("utf-8"))
    selected, samples = original_inputs(args.author_root, args.prepared_dir, pins)
    runtime = validate_runtime(args.python_runtime)
    work = args.work_root.resolve()
    for source in (args.author_root, args.prepared_dir, args.package_root, runtime):
        source = source.resolve()
        if work.is_relative_to(source) or source.is_relative_to(work):
            raise ValueError("fresh work-root must not overlap input or runtime roots")
    # Do not change permissions on existing shared directories or previous runs.
    work.mkdir(mode=0o700, exist_ok=False)
    os.chown(work, args.sandbox_uid, args.sandbox_uid)
    assets, root = work / "assets", work / "root"
    for name in ("inputs", "author/human_eval", "packages", "output"):
        (assets / name).mkdir(parents=True)
    root.mkdir()
    os.chown(root, args.sandbox_uid, args.sandbox_uid)
    os.chown(assets / "output", args.sandbox_uid, args.sandbox_uid)
    for name in pins["files"]:
        shutil.copyfile(args.author_root / name, assets / "author" / name)
    shutil.copyfile(samples, assets / "inputs/samples-private.jsonl")
    worker = Path(__file__).with_name("humaneval_isolated_worker.py")
    shutil.copyfile(worker, assets / "inputs/worker.py")
    with gzip.open(
        assets / "inputs/problems.jsonl.gz", "wt", encoding="utf-8"
    ) as stream:
        for problem in selected:
            stream.write(json.dumps(problem, ensure_ascii=False) + "\n")
    copy_packages(args.package_root, assets / "packages", pins)
    config = {
        "root": str(root),
        "assets": str(assets),
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
        str(assets / "inputs/worker.py"),
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
                "isolated scorer failed; inspect private log, never run on host"
            )
        receipt = json.loads((assets / "output/receipt-public.json").read_text("utf-8"))
        if (
            receipt.get("status") != "SELECTED_ORIGINAL_TASKS_OFFICIALLY_SCORED"
            or receipt.get("samples_sha256") != digest(samples)
            or receipt.get("original_task_ids") != [p["task_id"] for p in selected]
            or not receipt.get("isolation_checks_passed_before_import_and_execution")
        ):
            raise RuntimeError(
                "isolated receipt does not match original selected inputs"
            )
        receipt.update(
            author_revision=pins["revision"],
            original_data_sha256=pins["data_sha256"],
            worker_sha256=digest(worker),
            source_checkbox_eligible=False,
            full_suite_claim=len(selected) == 164,
        )
        (work / "receipt-public.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
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
            )
            + "\n",
            encoding="utf-8",
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "author-root",
        "prepared-dir",
        "package-root",
        "python-runtime",
        "work-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sandbox-uid", type=int, required=True)
    parser.add_argument("--deadline", type=int, default=600)
    args = parser.parse_args()
    try:
        receipt = launch(args)
    except Exception as error:
        parser.exit(
            1,
            f"isolated scoring incomplete ({type(error).__name__}); inspect private work-root\n",
        )
    print(json.dumps(receipt, ensure_ascii=True))


if __name__ == "__main__":
    main()
