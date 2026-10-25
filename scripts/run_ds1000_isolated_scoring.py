"""Run selected original DS1000 tasks in private Linux namespaces.

Inputs come from prepare_ds1000_official_samples.py. No inference, downloads,
prompt construction or host-execution fallback. Other library selections require
full author-pinned versions. Real qualification is separate from software support.
Requires an authorized root launcher and nonzero UID.
"""

import argparse
import gzip
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_ds1000_original_generation import DATA_SHA, select_indices
from run_humaneval_isolated_scoring import checked_file, digest, validate_runtime

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/ds1000-author-scoring-pins.json"


def validate_inputs(author, prepared, pins):
    author, prepared = Path(author).resolve(), Path(prepared).resolve()
    for relative, expected in pins["files"].items():
        checked_file(author, relative, expected)
    receipt = json.loads((prepared / "receipt-private.json").read_text("utf-8"))
    selected = select_indices(1000, indices=receipt.get("selected_problem_ids", []))
    if (
        receipt.get("status")
        != "AUTHOR_FUNCTIONAL_EVALUATOR_INPUT_PREPARED_NOT_EXECUTED"
        or receipt.get("source_id") != "DS1000"
        or receipt.get("code_revision") != pins["revision"]
        or receipt.get("original_dataset_sha256_required_by_cli") != DATA_SHA
        or receipt.get("all_original_rows_preserved") != 1000
    ):
        raise ValueError("original fixed preparation receipt required")
    for name in ("answers-private.jsonl", "original-dataset-private.jsonl"):
        checked_file(prepared, name, receipt["prepared_file_sha256"][name])
    with gzip.open(author / "data/ds1000.jsonl.gz", "rt", encoding="utf-8") as stream:
        original = [json.loads(line) for line in stream if line.strip()]
    retained = [
        json.loads(line)
        for line in (prepared / "original-dataset-private.jsonl")
        .read_text("utf-8")
        .splitlines()
        if line.strip()
    ]
    answers = [
        json.loads(line)
        for line in (prepared / "answers-private.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(original) != 1000
        or retained != original
        or [r.get("id") for r in answers] != selected
        or any(not isinstance(r.get("code"), str) for r in answers)
        or [int(r["metadata"]["problem_id"]) for r in original] != list(range(1000))
    ):
        raise ValueError("original full data or fixed task selection mismatch")
    libraries = sorted({original[i]["metadata"]["library"] for i in selected})
    return prepared / "answers-private.jsonl", selected, libraries


def validate_environment(runtime, pins, libraries=("Pandas",)):
    runtime = validate_runtime(runtime)
    site = runtime / "lib/python3.10/site-packages"
    installed = list(metadata.distributions(path=[str(site)]))
    required = dict(pins["package_versions"])
    if set(libraries) - {"Pandas", "Numpy"}:
        required.update(pins["full_package_versions"])
    for name, version in required.items():
        aliases = (
            {name} if name != "tensorflow-cpu" else {"tensorflow-cpu", "tensorflow"}
        )
        matches = [
            dist
            for dist in installed
            if dist.metadata.get("Name", "").lower().replace("_", "-") in aliases
        ]
        if not matches or any(
            dist.version.split("+")[0] != version for dist in matches
        ):
            raise ValueError("qualified author-profile environment required")
    return runtime


def launch(args):
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("authorized Linux root launcher required; no host fallback")
    if not 1 <= args.sandbox_uid <= 2**31 - 2 or args.deadline < 130:
        raise ValueError(
            "nonzero sandbox UID and outer deadline >=130 seconds required"
        )
    for executable in ("setpriv", "unshare", "mount"):
        if not shutil.which(executable):
            raise RuntimeError("namespace utilities unavailable; no host fallback")
    pins = json.loads(PINS.read_text("utf-8"))
    samples, selected, libraries = validate_inputs(
        args.author_root, args.prepared_dir, pins
    )
    runtime = validate_environment(args.python_runtime, pins, libraries)
    work = args.work_root.resolve()
    for source in (args.author_root, args.prepared_dir, runtime):
        source = source.resolve()
        if work.is_relative_to(source) or source.is_relative_to(work):
            raise ValueError("fresh work-root must not overlap input/runtime roots")
    work.mkdir(mode=0o700, exist_ok=False)
    os.chown(work, args.sandbox_uid, args.sandbox_uid)
    root, assets = work / "root", work / "assets"
    root.mkdir()
    os.chown(root, args.sandbox_uid, args.sandbox_uid)
    for name in ("inputs", "author/data", "packages", "output"):
        (assets / name).mkdir(parents=True)
    os.chown(assets / "output", args.sandbox_uid, args.sandbox_uid)
    for relative in pins["files"]:
        shutil.copyfile(args.author_root / relative, assets / "author" / relative)
    shutil.copyfile(samples, assets / "inputs/answers-private.jsonl")
    (assets / "inputs/selection.json").write_text(
        json.dumps({"problem_ids": selected, "libraries": libraries}), encoding="utf-8"
    )
    worker = Path(__file__).with_name("ds1000_isolated_worker.py")
    common = Path(__file__).with_name("native_python_namespace.py")
    shutil.copyfile(worker, assets / "inputs/worker.py")
    shutil.copyfile(common, assets / "inputs/native_python_namespace.py")
    # Copy only the explicitly supplied scoring environment, never host model
    # data. Alias it inside chroot so its original /data parent is not exposed.
    shutil.copytree(
        runtime,
        assets / "runtime",
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    config = {
        "root": str(root),
        "assets": str(assets),
        "runtime": "/opt/python",
        "runtime_source": str(assets / "runtime"),
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
        sys.executable,
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
            raise RuntimeError("isolated scorer failed; inspect private log")
        if "can't start new thread" in (work / "scorer-private.log").read_text(
            "utf-8", errors="replace"
        ):
            raise RuntimeError(
                "environment thread exhaustion; reject apparent task scores"
            )
        receipt = json.loads((assets / "output/receipt-public.json").read_text("utf-8"))
        if (
            receipt.get("status") != "SELECTED_ORIGINAL_DS1000_TASKS_OFFICIALLY_SCORED"
            or receipt.get("samples_sha256") != digest(samples)
            or receipt.get("original_problem_ids") != selected
            or receipt.get("official_summary_count") != len(selected)
            or not receipt.get("isolation_checks_passed_before_author_import")
        ):
            raise RuntimeError("native receipt/input binding mismatch")
        receipt.update(
            author_revision=pins["revision"],
            public_worker_sha256=digest(worker),
            public_namespace_helper_sha256=digest(common),
            profile="original-task-selection",
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
    for name in ("author-root", "prepared-dir", "python-runtime", "work-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sandbox-uid", type=int, required=True)
    parser.add_argument("--deadline", type=int, default=600)
    args = parser.parse_args()
    try:
        receipt = launch(args)
    except Exception as error:
        parser.exit(
            1,
            f"DS1000 isolated scoring incomplete ({type(error).__name__}); inspect private work-root\n",
        )
    print(json.dumps(receipt, ensure_ascii=True))


if __name__ == "__main__":
    main()
