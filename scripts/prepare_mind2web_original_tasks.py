"""Prepare fixed complete tasks externally from the original encrypted test ZIP."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("test_task", "test_website", "test_domain")


def select_original_tasks(tasks):
    if not isinstance(tasks, list) or len(tasks) < 3:
        raise ValueError(
            "at least three original tasks required; never replace missing tasks"
        )
    selected = tasks[:3]
    for task in selected:
        if (
            not isinstance(task, dict)
            or not task.get("annotation_id")
            or not isinstance(task.get("actions"), list)
            or not task["actions"]
        ):
            raise ValueError("complete original task required")
        if len(task.get("action_reprs", [])) != len(task["actions"]):
            raise ValueError("original action relationship mismatch")
        ids = [action["action_uid"] for action in task["actions"]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate original action identity")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        parser.error("original tasks must remain outside the checkout")
    if not args.execute:
        print(
            json.dumps(
                {
                    "executed": False,
                    "tasks_per_split": 3,
                    "retain_all_fields_and_actions": True,
                }
            )
        )
        return
    spec = json.loads(
        (ROOT / "configs/workload-native/mind2web-original-protocol.json").read_text(
            "utf-8"
        )
    )
    expected = next(
        item for item in spec["official_assets"] if item["path"] == "test.zip"
    )
    if args.archive.stat().st_size != expected["bytes"]:
        raise ValueError("original archive size mismatch")
    with args.archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
            raise ValueError("original archive identity mismatch")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    report = []
    with zipfile.ZipFile(args.archive) as archive:
        for split in SPLITS:
            names = sorted(
                name
                for name in archive.namelist()
                if PurePosixPath(name).parent.as_posix() == split
                and name.endswith(".json")
            )
            if not names:
                raise ValueError("original split missing")
            raw = archive.read(names[0], pwd=b"mind2web")
            tasks = select_original_tasks(json.loads(raw))
            with (args.output / f"{split}-original-first3.private.json").open(
                "x"
            ) as stream:
                json.dump(tasks, stream)
            report.append(
                {
                    "split": split,
                    "original_member": names[0],
                    "member_sha256": hashlib.sha256(raw).hexdigest(),
                    "task_ids": [task["annotation_id"] for task in tasks],
                    "action_counts": [len(task["actions"]) for task in tasks],
                }
            )
    with (args.output / "preparation-public.json").open("x") as stream:
        json.dump({"splits": report, "native_execution": "NOT_RUN"}, stream, indent=2)
    print(json.dumps({"tasks": 9, "native_execution": "NOT_RUN"}))


if __name__ == "__main__":
    main()
