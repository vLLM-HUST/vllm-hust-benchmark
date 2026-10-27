"""Private, bounded patch-state telemetry; executable in the grading container."""

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def affected_paths(patch, root):
    paths = set()
    # apply --numstat reports the destination of a rename; reverse parsing
    # includes its source. Neither invocation applies the patch.
    for flags in ([], ["--reverse"]):
        result = subprocess.run(
            ["git", "apply", "--numstat", "-z", *flags, str(patch)],
            cwd=root,
            capture_output=True,
            check=True,
        )
        fields = iter(result.stdout.split(b"\0"))
        for field in fields:
            if not field:
                continue
            name = field.split(b"\t", 2)[2]
            names = (name,) if name else (next(fields), next(fields))
            for raw in names:
                name = os.fsdecode(raw)
                path = Path(name)
                if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
                    raise ValueError("Unsafe patch path")
                paths.add(name)
    return sorted(paths)


def file_state(root, name):
    path = root / name
    try:
        path.parent.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError("Patch parent escapes repository")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {"kind": "missing"}
    value = {"mode": stat.S_IMODE(metadata.st_mode)}
    if stat.S_ISLNK(metadata.st_mode):
        value.update(
            kind="symlink",
            target_sha256=hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest(),
        )
    elif stat.S_ISREG(metadata.st_mode):
        value.update(kind="file", bytes=metadata.st_size, sha256=sha(path))
    else:
        value.update(kind="other")
    return value


def snapshot(root, patch, stage):
    result = {"stage": stage, "patch_sha256": sha(patch), "status": "captured"}
    result["head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    result["files"] = {
        name: file_state(root, name) for name in affected_paths(patch, root)
    }
    return result


def main():
    stage, patch, output = sys.argv[1:]
    try:
        value = snapshot(Path.cwd(), Path(patch), stage)
    except Exception as exc:
        value = {
            "stage": stage,
            "status": "capture_failed",
            "error_type": type(exc).__name__,
        }
    Path(output).write_text(json.dumps(value, sort_keys=True) + "\n")
    return 0 if value["status"] == "captured" else 1


if __name__ == "__main__":
    sys.exit(main())
