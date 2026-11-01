"""Use AppWorld's official data downloader in a fresh external private root.

No task changes, generated prompts, Agent execution, scoring or model calls.
The publisher downloader removes an existing data directory: this wrapper
therefore rejects existing destinations before invoking it. Never publish the
unpacked protected assets or treat a successful download as benchmark delivery.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

VERSION = "0.1.3.post1"
SPLITS = ("train", "dev", "test_normal", "test_challenge")


def audit_relations(root, load_task_ids):
    result = {}
    all_ids = set()
    for split in SPLITS:
        ids = load_task_ids(split)
        if not ids or len(set(ids)) != len(ids) or all_ids.intersection(ids):
            raise ValueError("empty, duplicate or overlapping original task IDs")
        for task in ids:
            folder = (root / "data/tasks" / task).resolve()
            if not folder.is_relative_to((root / "data/tasks").resolve()):
                raise ValueError("task directory escapes original data root")
            if not folder.is_dir() or not any(folder.iterdir()):
                raise ValueError("original task directory missing or empty")
        all_ids.update(ids)
        result[split] = {"original_ids": len(ids), "directory_relations_verified": True}
    return {"splits": result, "unique_tasks": len(all_ids)}


def prepare(output, client, cache, execute=False, verify_only=False):
    output, client, cache = map(
        lambda value: Path(value).resolve(), (output, client, cache)
    )
    command = [
        str(client / "bin/appworld"),
        "download",
        "data",
        "--root",
        str(output / "world"),
    ]
    result = {
        "source_id": "APPWORLD",
        "status": "PLAN_ONLY",
        "command": command,
        "model_calls": 0,
        "benchmark_executed": False,
        "source_checkbox_eligible": False,
    }
    if not execute and not verify_only:
        return result
    repository = Path(__file__).resolve().parents[1]
    if (
        sys.platform != "linux"
        or (output.exists() and not verify_only)
        or output.is_relative_to(repository)
    ):
        raise ValueError(
            "Linux and fresh external destination required; never replace existing data"
        )
    if (
        not output.parent.is_dir()
        or not cache.is_dir()
        or not (client / "bin/python").is_file()
    ):
        raise ValueError(
            "existing parent, private AppWorld cache and isolated client required"
        )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        APPWORLD_ROOT=str(output / "world"),
        APPWORLD_CACHE=str(cache),
        TMPDIR=str(output / "tmp"),
        XDG_CACHE_HOME=str(output / "cache"),
    )
    preflight = "import importlib.metadata as m,sys; assert sys.version_info[:2]==(3,11); assert m.version('appworld')==sys.argv[1]; assert sys.prefix!=sys.base_prefix"
    subprocess.run(
        [str(client / "bin/python"), "-c", preflight, VERSION],
        env=env,
        check=True,
        timeout=30,
    )

    def verify():
        code = "import json; from appworld.task import load_task_ids; print(json.dumps({s:load_task_ids(s) for s in ('train','dev','test_normal','test_challenge')}))"
        completed = subprocess.run(
            [str(client / "bin/python"), "-c", code],
            env=env,
            cwd=output / "world",
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        ids = json.loads(completed.stdout)
        return audit_relations(output / "world", lambda split: ids[split])

    if verify_only:
        result.update(verify())
        result["status"] = (
            "ORIGINAL_SPLIT_DIRECTORY_RELATIONS_VERIFIED_READ_ONLY_NOT_EXECUTED"
        )
        return result
    output.mkdir(mode=0o700)
    for name in ("world", "tmp", "cache"):
        (output / name).mkdir(mode=0o700)
    result["status"] = "PREPARING_OFFICIAL_DATA"
    try:
        with (output / "download-private.log").open("x") as log:
            subprocess.run(
                command,
                env=env,
                cwd=output / "world",
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        files = [path for path in (output / "world/data").rglob("*") if path.is_file()]
        if not files:
            raise ValueError("official CLI returned without original data; not success")
        result.update(
            status="OFFICIAL_DOWNLOAD_FILES_PRESENT_NOT_TASK_VALIDATION",
            file_count=len(files),
            bytes=sum(path.stat().st_size for path in files),
        )
        result.update(verify())
        result["status"] = (
            "OFFICIAL_DOWNLOAD_AND_SPLIT_DIRECTORY_RELATIONS_VERIFIED_NOT_EXECUTED"
        )
    except BaseException as error:
        result.update(
            status="FAILED_NOT_COMPLETE",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        (output / "preparation-public.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Read-only reuse; never download or rewrite the original assets",
    )
    args = parser.parse_args()
    if args.execute and args.verify_only:
        parser.error("choose --execute or --verify-only")
    print(
        json.dumps(
            prepare(
                args.output, args.client, args.cache, args.execute, args.verify_only
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
