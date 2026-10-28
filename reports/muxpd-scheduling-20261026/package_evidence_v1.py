"""Package terminal scheduling evidence without caches; no device/network calls.

plan reads only controller JSON and file metadata. build/verify hash and compress
selected payloads and must be invoked only after all timed experiments finish.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath

ARCHIVE_NAME = "muxpd-scheduling-20261026-evidence-v1.tar.gz"
TOP_RUN_FILES = (
    "controller.json",
    "engine-args.json",
    "workload-v1.json",
    "process.log",
    "npu-before.txt",
    "npu-after.txt",
)
SUPPORT_DIRS = ("bench", "bench-v2", "analysis", "async-prototype", "lifecycle-v1")
EXCLUDED_PARTS = {
    "compiler-cache",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".git",
}


def encoded(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or any(p in EXCLUDED_PARTS for p in path.parts)
    ):
        raise ValueError(f"Unsafe or excluded member name: {name}")


def collect_tree(directory):
    if not directory.exists():
        return []
    if directory.is_symlink():
        raise ValueError(f"Selected directory is a symlink: {directory}")
    result = []
    for current, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_PARTS)
        for name in dirs:
            if (Path(current) / name).is_symlink():
                raise ValueError(
                    f"Selected directory is a symlink: {Path(current) / name}"
                )
        for name in sorted(files):
            path = Path(current) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Only regular files can be packaged: {path}")
            result.append(path)
    return result


def selection(root, support_dirs=SUPPORT_DIRS, extra_files=()):
    root = root.resolve()
    files, runs, excluded = {}, [], []

    def add(path):
        relative = path.relative_to(root).as_posix()
        safe_name(relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Only existing regular files can be selected: {path}")
        files[relative] = path

    run_root = root / "runs"
    if not run_root.is_dir():
        raise ValueError("Campaign has no runs directory")
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir() or run_dir.is_symlink():
            raise ValueError(f"Unexpected entry in runs directory: {run_dir}")
        controller_path = run_dir / "controller.json"
        controller = json.loads(controller_path.read_text())
        terminal = bool(controller.get("finished_at"))
        runs.append(
            {
                "label": run_dir.name,
                "terminal": terminal,
                "controller_passed": controller.get("passed"),
                "exit_code": controller.get("exit_code"),
                "started_at": controller.get("started_at"),
                "finished_at": controller.get("finished_at"),
                "source_commit": controller.get("source_commit"),
                "role": controller.get("role"),
                "owned_members_after_cleanup": controller.get(
                    "owned_members_after_cleanup"
                ),
                "errors": controller.get("errors", []),
            }
        )
        for name in TOP_RUN_FILES:
            path = run_dir / name
            if path.exists():
                add(path)
        for name in ("producer", "results"):
            for path in collect_tree(run_dir / name):
                add(path)
        if (run_dir / "compiler-cache").exists():
            excluded.append(
                (run_dir / "compiler-cache").relative_to(root).as_posix() + "/**"
            )
        unexpected = sorted(
            path.name
            for path in run_dir.iterdir()
            if path.name
            not in {*TOP_RUN_FILES, "producer", "results", "compiler-cache"}
        )
        if unexpected:
            raise ValueError(
                f"Unclassified run paths must be reviewed explicitly: {run_dir.name}: {unexpected}"
            )
    if not runs:
        raise ValueError("No runs selected")
    for relative in support_dirs:
        safe_name(relative)
        for path in collect_tree(root / relative):
            add(path)
    for relative in extra_files:
        safe_name(relative)
        add(root / relative)
    packager = Path(__file__).resolve()
    files["publication-tools/package_evidence_v1.py"] = packager
    return {
        "schema": "muxpd-scheduling-selection-v1",
        "source_root": str(root),
        "runs": runs,
        "excluded_paths": excluded,
        "exclusion_policy": "Only enumerated run artifacts and selected support trees; compiler-cache, Python/test/lint caches, Git internals and symlinks are excluded. No model weights or compiler cache are packaged.",
        "files": [
            {"name": name, "source_path": str(path), "size_bytes": path.stat().st_size}
            for name, path in sorted(files.items())
        ],
    }


def require_terminal(selected):
    if any(not row["terminal"] for row in selected["runs"]):
        raise ValueError(
            "Packaging requires every observed run to be terminal; stop timing before hashing/compression"
        )
    if any(row["owned_members_after_cleanup"] for row in selected["runs"]):
        raise ValueError("A terminal run reports surviving owned processes")


def tar_info(name, size):
    safe_name(name)
    value = tarfile.TarInfo(name)
    value.size, value.mode, value.mtime = size, 0o644, 0
    value.uid = value.gid = 0
    value.uname = value.gname = ""
    return value


class HashingReader:
    def __init__(self, stream):
        self.stream, self.digest, self.count = stream, hashlib.sha256(), 0

    def read(self, count=-1):
        value = self.stream.read(count)
        self.digest.update(value)
        self.count += len(value)
        return value


def build(root, output, support_dirs=SUPPORT_DIRS, extra_files=()):
    selected = selection(root, support_dirs, extra_files)
    require_terminal(selected)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if any((parent / ".git").exists() for parent in (output, *output.parents)):
        raise ValueError("Archive payload output must be outside a Git worktree")
    output.mkdir(parents=True, exist_ok=False)
    members = [
        {**row, "sha256": sha_file(Path(row["source_path"]))}
        for row in selected["files"]
    ]
    content = {
        **selected,
        "files": members,
        "member_count": len(members),
        "original_bytes": sum(row["size_bytes"] for row in members),
        "packager_sha256": sha_file(Path(__file__)),
        "status_scope": "Terminal failures are retained. Packaging success does not mean performance, quality, or remote publication passed.",
    }
    manifest_bytes = encoded(content)
    archive = output / ARCHIVE_NAME
    with (
        archive.open("xb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
        ) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
        ) as package,
    ):
        package.addfile(
            tar_info("MANIFEST.json", len(manifest_bytes)), io.BytesIO(manifest_bytes)
        )
        for row in members:
            with Path(row["source_path"]).open("rb") as stream:
                reader = HashingReader(stream)
                package.addfile(tar_info(row["name"], row["size_bytes"]), reader)
                if (
                    reader.count != row["size_bytes"]
                    or reader.digest.hexdigest() != row["sha256"]
                    or stream.read(1)
                ):
                    raise ValueError(f"Source changed during packaging: {row['name']}")
    result = {
        "schema": "muxpd-scheduling-archive-v1",
        "archive": {
            "name": archive.name,
            "size_bytes": archive.stat().st_size,
            "sha256": sha_file(archive),
        },
        "embedded_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "content_manifest": content,
    }
    manifest = output / "MANIFEST-v1.json"
    manifest.write_bytes(encoded(result))
    (output / "SHA256SUMS-v1").write_text(
        f"{result['archive']['sha256']}  {archive.name}\n{sha_file(manifest)}  {manifest.name}\n"
    )
    verify(archive, manifest)
    return result


def verify(archive, manifest):
    descriptor = json.loads(manifest.read_text())
    expected_archive = descriptor["archive"]
    if (
        archive.stat().st_size != expected_archive["size_bytes"]
        or sha_file(archive) != expected_archive["sha256"]
    ):
        raise ValueError("Archive bytes differ from companion manifest")
    content = descriptor["content_manifest"]
    expected = {row["name"]: row for row in content["files"]}
    if len(expected) != len(content["files"]):
        raise ValueError("Duplicate expected member names")
    observed = set()
    with tarfile.open(archive, "r:gz") as package:
        for member in package:
            safe_name(member.name)
            if member.name in observed or not member.isfile():
                raise ValueError("Duplicate or non-regular archive member")
            observed.add(member.name)
            stream = package.extractfile(member)
            if member.name == "MANIFEST.json":
                value = stream.read()
                if (
                    hashlib.sha256(value).hexdigest()
                    != descriptor["embedded_manifest_sha256"]
                    or json.loads(value) != content
                ):
                    raise ValueError("Embedded and companion manifests disagree")
            else:
                if (
                    member.name not in expected
                    or member.size != expected[member.name]["size_bytes"]
                ):
                    raise ValueError("Unexpected member or length")
                reader = HashingReader(stream)
                while reader.read(1024 * 1024):
                    pass
                if reader.digest.hexdigest() != expected[member.name]["sha256"]:
                    raise ValueError(f"Member checksum mismatch: {member.name}")
    if observed != {*expected, "MANIFEST.json"}:
        raise ValueError("Missing archive members")
    return {
        "passed": True,
        "archive": str(archive),
        "original_member_count": len(expected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "build", "verify"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--support-dir",
        action="append",
        help="Override support-tree list; repeat to select multiple root-relative trees",
    )
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help="Explicit extra root-relative small artifact",
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        if not args.archive or not args.manifest:
            parser.error("verify requires --archive and --manifest")
        result = verify(args.archive, args.manifest)
    else:
        if not args.root or (args.command == "build" and not args.output):
            parser.error("plan/build require --root; build also requires --output")
        support = args.support_dir if args.support_dir is not None else SUPPORT_DIRS
        if args.command == "plan":
            result = selection(args.root, support, args.extra_file)
            result["payload_hashing_or_compression_performed"] = False
        else:
            result = build(args.root, args.output, support, args.extra_file)
            result = {
                "archive": result["archive"],
                "original_member_count": result["content_manifest"]["member_count"],
                "output": str(args.output),
            }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
