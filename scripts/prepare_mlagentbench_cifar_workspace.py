"""Bind original CIFAR task files and archive, without extracting or executing."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_jsonschemabench_author_source import verify_source
from prepare_mlagentbench_cifar_assets import PROFILE, verify

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PINS = ROOT / "configs/workload-native/mlagentbench-author-source.json"
TASK_PREFIX = "MLAgentBench/benchmarks/cifar10/"


def prepare(args):
    report = {
        "source_id": "MLAGENTBENCH",
        "status": "PLAN_ONLY_NO_DATA_EXTRACTION_OR_EXECUTION",
        "model_calls": 0,
        "native_tasks_run": 0,
        "source_checkbox_eligible": False,
    }
    if args.action == "plan":
        return report
    author = verify_source(args.author_root, SOURCE_PINS)
    profile = json.loads(PROFILE.read_text("utf-8"))
    archive = args.data_root / profile["data"]["path"]
    verify(archive, profile["data"])
    output = args.output.resolve()
    protected = [ROOT, author, args.data_root.resolve()]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError(
            "fresh external workspace separate from original assets required"
        )
    pins = json.loads(SOURCE_PINS.read_text("utf-8"))
    task_files = {
        item["path"][len(TASK_PREFIX) :]: item
        for item in pins["files"]
        if item["path"].startswith(TASK_PREFIX)
    }
    if (
        not {"scripts/prepare.py", "scripts/eval.py", "env/train.py"}
        <= task_files.keys()
    ):
        raise ValueError("original task source profile is incomplete")
    if args.action == "prepare":
        if not output.parent.is_dir():
            raise ValueError("existing external workspace parent required")
        if (
            shutil.disk_usage(output.parent).free
            < profile["data"]["bytes"] + 256 * 1024**2
        ):
            raise ValueError("insufficient workspace capacity and reserve")
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        for relative, item in task_files.items():
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with (
                (author / item["path"]).open("rb") as source,
                target.open("xb") as dest,
            ):
                shutil.copyfileobj(source, dest, length=1024**2)
        data = output / "env/data"
        data.mkdir(exist_ok=False)
        with archive.open("rb") as source, (data / archive.name).open("xb") as dest:
            shutil.copyfileobj(source, dest, length=1024**2)
    for relative, item in task_files.items():
        target = output / relative
        if target.is_symlink() or not target.resolve(strict=True).is_relative_to(
            output
        ):
            raise ValueError("task software must be a contained original file")
        if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("task software differs from pinned original")
    verify(output / "env/data" / archive.name, profile["data"])
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != set(task_files) | {"env/data/" + archive.name}:
        raise ValueError("workspace must contain only original task files and archive")
    report.update(
        status="ORIGINAL_TASK_WORKSPACE_VERIFIED_NOT_EXTRACTED_OR_EXECUTED",
        task_files=len(task_files),
        original_archive_bytes=profile["data"]["bytes"],
        archive_extracted=False,
        scientific_training_executed=False,
        next_step="Run unchanged scripts/prepare.py inside the isolated CPU task container, from scripts/; no network required with verified original archive",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
