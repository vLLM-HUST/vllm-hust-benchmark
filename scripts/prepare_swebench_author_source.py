"""Prepare pinned official benchmark/Agent software externally; never execute it.

The original package is shared by SWE-bench and Verified. A supplied official
archive avoids hundreds of Git blob API requests. Each selected file is still
checked against both its pinned Git object and SHA256 before publication.
MLE-bench expanded Git LFS assets remain bound to their original pointer objects.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

from prepare_crosscodeeval_original_assets import git_blob_for_payload, verify_git_blob
from prepare_jsonschemabench_author_source import verify_source
from prepare_xsum_original_assets import bound_path

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/swebench-author-source.json"


def prepare(archive_path, output, pins):
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("author software must remain outside the checkout")
    expected = {item["path"]: item for item in pins["files"]}
    if len(expected) != len(pins["files"]):
        raise ValueError("duplicate pinned original path")
    if any(
        "git_mode" in item and item["git_mode"] not in {"100644", "100755"}
        for item in expected.values()
    ):
        raise ValueError("regular original Git file mode required")
    seen = set()
    prefix = None
    with tarfile.open(archive_path, "r:gz") as archive:
        for count, member in enumerate(archive, 1):
            if count > 10000:
                raise ValueError("archive member budget exceeded")
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or len(path.parts) < 1:
                raise ValueError("unsafe archive member path")
            if prefix is None:
                prefix = path.parts[0]
            if prefix != path.parts[0]:
                raise ValueError("one original repository archive root required")
            name = PurePosixPath(*path.parts[1:]).as_posix()
            if name not in expected:
                continue
            item = expected[name]
            if name in seen or not member.isfile() or member.size != item["bytes"]:
                raise ValueError("duplicate, nonregular or size-mismatched original")
            seen.add(name)
            raw = archive.extractfile(member).read(item["bytes"] + 1)
            if (
                len(raw) != item["bytes"]
                or hashlib.sha256(raw).hexdigest() != item["sha256"]
                or git_blob_for_payload(raw, item) != item["git_blob"]
            ):
                raise ValueError("original publisher file identity mismatch")
            target = bound_path(output, "", item)
            if target.exists():
                verify_git_blob(target, item)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            with partial.open("xb") as handle:
                handle.write(raw)
            os.link(partial, target)
            partial.unlink()
            if "git_mode" in item:
                target.chmod(int(item["git_mode"], 8) & 0o777)
    if seen != set(expected):
        raise ValueError("incomplete original software archive")
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--profile",
        choices=[
            "current",
            "v4.1.0",
            "mini-swe-agent",
            "webarena",
            "mlebench",
            "visualwebarena",
        ],
        default="current",
        help="Original software profiles; SWE-bench default unchanged. WebArena/MLE-bench use existing official archives,not task execution.",
    )
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    args = parser.parse_args()
    profiles = {
        "current": PINS,
        "v4.1.0": PINS.with_name("swebench-v410-author-source.json"),
        "mini-swe-agent": PINS.with_name("mini-swe-agent-author-source.json"),
        "webarena": PINS.with_name("webarena-author-source.json"),
        "mlebench": PINS.with_name("mlebench-author-source.json"),
        "visualwebarena": PINS.with_name("visualwebarena-author-source.json"),
    }
    pins_path = profiles[args.profile]
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    if args.action == "prepare":
        if args.archive is None:
            parser.error("explicit existing official software archive required")
        prepare(args.archive, args.author_root, pins)
    if args.action != "plan":
        verify_source(args.author_root, pins_path)
    print(
        json.dumps(
            {
                "source_ids": pins["shared_source_ids"]
                if "shared_source_ids" in pins
                else [pins["source_id"]],
                "revision": pins["revision"],
                "official_archive_url": f"https://api.github.com/repos/{pins['repository']}/tarball/{pins['revision']}",
                "status": "PLAN_ONLY"
                if args.action == "plan"
                else "ORIGINAL_SOFTWARE_VERIFIED_NOT_EXECUTED",
                "files": len(pins["files"]),
                "bytes": sum(item["bytes"] for item in pins["files"]),
                "model_calls": 0,
                "source_checkbox_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
