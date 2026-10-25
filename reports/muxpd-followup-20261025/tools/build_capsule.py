"""Append-only offline evidence capsules; no model, network or Git mutations.

Freeze explicit terminal runs or explicit support files, then build and independently
reread the archive. Existing output directories are never reused. Failed attempts
are retained separately and are never represented by a successful receipt.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


def now():
    return datetime.now(UTC).isoformat()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def hash_stream(stream):
    sha = hashlib.sha256()
    size = 0
    while block := stream.read(1024 * 1024):
        sha.update(block)
        size += len(block)
    return size, sha.hexdigest()


def identity(path):
    path = Path(path).absolute()
    if path.is_symlink():
        target = os.readlink(path)
        return {
            "path": str(path),
            "type": "symlink",
            "link_target": target,
            "bytes": len(os.fsencode(target)),
            "sha256": digest_bytes(os.fsencode(target)),
        }
    if not path.is_file():
        raise ValueError(f"Not a regular file: {path}")
    before = path.stat()
    with path.open("rb") as stream:
        size, sha = hash_stream(stream)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError(f"Source changed while hashing: {path}")
    return {"path": str(path), "type": "file", "bytes": size, "sha256": sha}


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(p in ("", ".", "..") for p in path.parts):
        raise ValueError(f"Unsafe archive member: {name}")
    return name


def validate_terminal(run):
    run = Path(run).resolve()
    controller = run / "controller.json"
    value = json.loads(controller.read_text())
    if not value.get("finished_at") or type(value.get("exit_code")) is not int:
        raise ValueError(f"Run is not terminal: {run}")
    if type(value.get("passed")) is not bool:
        raise ValueError("Missing explicit terminal outcome")
    if value["passed"] and value["exit_code"] != 0:
        raise ValueError("Contradictory terminal outcome")
    pid = value.get("pid")
    if type(pid) is not int or pid <= 0 or Path(f"/proc/{pid}").exists():
        raise ValueError("Missing process identity or recorded process is still alive")
    group = value.get("process_group")
    if type(group) is not int or group <= 0:
        raise ValueError("Missing terminal process-group identity")
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            line = stat.read_text()
        except FileNotFoundError:
            continue
        fields = line[line.rfind(")") + 2 :].split()
        if int(fields[2]) == group and fields[0] != "Z":
            raise ValueError("Recorded process group still has live members")
    for name in (
        "process.log",
        "engine-args.json",
        "plan.json",
        "npu-before.txt",
        "npu-after.txt",
    ):
        if not (run / name).is_file():
            raise ValueError(f"Missing terminal custody file: {name}")
    if not re.fullmatch(r"[0-9a-f]{40}", value.get("source_commit", "")):
        raise ValueError("Run has no exact source commit")
    for key, filename in (
        ("config_sha256", "engine-args.json"),
        ("plan_file_sha256", "plan.json"),
    ):
        if identity(run / filename)["sha256"] != value[key]:
            raise ValueError(f"Controller {key} mismatch")
    return {
        "run_path": str(run),
        "label": value["label"],
        "controller": identity(controller),
        "started_at": value["started_at"],
        "finished_at": value["finished_at"],
        "exit_code": value["exit_code"],
        "controller_passed": value["passed"],
        "source_repository": value["source"],
        "source_commit": value["source_commit"],
        "source_status": value.get("source_status"),
        "scope": "Terminal diagnostic evidence, including failures; collection success is not experimental acceptance.",
    }


def walk_members(root, prefix):
    rows = []
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = Path(base) / name
            if path.is_symlink():
                files.append(name)
                dirs.remove(name)
        for name in files:
            path = Path(base) / name
            row = identity(path)
            row["member"] = safe_name(
                str(PurePosixPath(prefix) / path.relative_to(root).as_posix())
            )
            rows.append(row)
    return sorted(rows, key=lambda r: r["member"])


def freeze_run(run, capsule_id, output):
    terminal = validate_terminal(run)
    rows = walk_members(Path(terminal["run_path"]), f"runs/{terminal['label']}")
    value = {
        "schema": "muxpd-evidence-capsule-manifest-v1",
        "capsule_id": capsule_id,
        "kind": "terminal-run",
        "frozen_at": now(),
        "terminal": terminal,
        "members": rows,
        "producer": identity(__file__),
    }
    write_new(output, value)
    return value


def freeze_support(selection, capsule_id, output):
    selection = Path(selection)
    rows = []
    for entry in json.loads(selection.read_text())["files"]:
        row = identity(entry["path"])
        if entry.get("sha256") != row["sha256"]:
            raise ValueError(
                f"Support source differs from explicit selection: {entry['path']}"
            )
        row["member"] = safe_name(entry["member"])
        rows.append(row)
    if len({r["member"] for r in rows}) != len(rows):
        raise ValueError("Duplicate support member")
    value = {
        "schema": "muxpd-evidence-capsule-manifest-v1",
        "capsule_id": capsule_id,
        "kind": "explicit-support",
        "frozen_at": now(),
        "selection": identity(selection),
        "members": sorted(rows, key=lambda r: r["member"]),
        "producer": identity(__file__),
    }
    write_new(output, value)
    return value


def tar_member(name, data, mode=0o644):
    info = tarfile.TarInfo(safe_name(name))
    info.mode = mode
    info.mtime = 0
    info.size = len(data)
    return info, io.BytesIO(data)


def verify_archive(archive, manifest, expected_sha=None):
    archive_id = identity(archive)
    if expected_sha is not None and archive_id["sha256"] != expected_sha:
        raise ValueError("Compressed archive SHA mismatch")
    expected = {row["member"]: row for row in manifest["members"]}
    if len(expected) != len(manifest["members"]):
        raise ValueError("Duplicate manifest members")
    seen = set()
    embedded = None
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            safe_name(member.name)
            if member.name in seen:
                raise ValueError("Duplicate archived member")
            seen.add(member.name)
            if member.name == "capsule-manifest.json":
                embedded = json.load(tar.extractfile(member))
                continue
            row = expected.get(member.name)
            if row is None:
                raise ValueError(f"Unexpected member: {member.name}")
            if row["type"] == "symlink":
                actual = {
                    "type": "symlink",
                    "link_target": member.linkname,
                    "bytes": len(os.fsencode(member.linkname)),
                    "sha256": digest_bytes(os.fsencode(member.linkname)),
                }
                if not member.issym():
                    raise ValueError("Symlink type changed")
            else:
                if not member.isfile():
                    raise ValueError("File type changed")
                size, sha = hash_stream(tar.extractfile(member))
                actual = {"type": "file", "bytes": size, "sha256": sha}
            if any(row[k] != value for k, value in actual.items()):
                raise ValueError(f"Archived member differs: {member.name}")
    if seen != set(expected) | {"capsule-manifest.json"} or embedded != manifest:
        raise ValueError("Incomplete archive or embedded manifest mismatch")
    return {
        "verified": True,
        "archive": archive_id,
        "members_verified": len(expected),
        "original_bytes": sum(r["bytes"] for r in expected.values()),
        "scope": "Local archive bytes and complete member inventory verified; no remote upload/download or model acceptance implied.",
    }


def build(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]+", manifest["capsule_id"]):
        raise ValueError("Unsafe capsule id")
    output.mkdir(parents=False, exist_ok=False)
    try:
        if manifest["kind"] == "terminal-run":
            current = validate_terminal(manifest["terminal"]["run_path"])
            if current != manifest["terminal"]:
                raise ValueError("Terminal receipt changed after manifest freeze")
            rows = walk_members(Path(current["run_path"]), f"runs/{current['label']}")
            if rows != manifest["members"]:
                raise ValueError("Run files changed after manifest freeze")
        else:
            for row in manifest["members"]:
                if identity(row["path"]) != {
                    k: v for k, v in row.items() if k != "member"
                }:
                    raise ValueError(f"Support file changed: {row['path']}")
        raw_manifest = manifest_path.read_bytes()
        archive = output / f"{manifest['capsule_id']}.tar.gz"
        with (
            archive.open("xb") as target,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=target, mtime=0, compresslevel=1
            ) as zipped,
            tarfile.open(fileobj=zipped, mode="w|") as tar,
        ):
            info, stream = tar_member("capsule-manifest.json", raw_manifest)
            tar.addfile(info, stream)
            for row in manifest["members"]:
                path = Path(row["path"])
                if row["type"] == "symlink":
                    info = tarfile.TarInfo(row["member"])
                    info.type = tarfile.SYMTYPE
                    info.linkname = row["link_target"]
                    info.mtime = 0
                    tar.addfile(info)
                else:
                    info = tarfile.TarInfo(row["member"])
                    info.size = row["bytes"]
                    info.mode = path.stat().st_mode & 0o777
                    info.mtime = 0
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
        receipt = verify_archive(archive, manifest)
        # Recheck complete originals after compression. No source may drift silently.
        for row in manifest["members"]:
            if identity(row["path"]) != {k: v for k, v in row.items() if k != "member"}:
                raise ValueError(f"Source changed during archive build: {row['path']}")
        if manifest["kind"] == "terminal-run":
            current = manifest["terminal"]
            if (
                walk_members(Path(current["run_path"]), f"runs/{current['label']}")
                != manifest["members"]
            ):
                raise ValueError("Run inventory changed during archive build")
        with (output / "manifest.json").open("xb") as stream:
            stream.write(raw_manifest)
        receipt.update(
            schema="muxpd-evidence-capsule-receipt-v1",
            capsule_id=manifest["capsule_id"],
            manifest=identity(output / "manifest.json"),
            completed_at=now(),
            producer=identity(__file__),
        )
        write_new(output / "receipt.json", receipt)
        with (output / "SHA256SUMS").open("x") as stream:
            for path in (archive, output / "manifest.json", output / "receipt.json"):
                stream.write(f"{identity(path)['sha256']}  {path.name}\n")
        return receipt
    except Exception as exc:
        write_new(
            output / "BUILD-FAILED.json",
            {"failed_at": now(), "error": repr(exc), "manifest": str(manifest_path)},
        )
        raise


def export_source(repo, revision, name, output):
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(
        r"[a-z0-9-]+", name
    ):
        raise ValueError("Exact commit and safe source name required")
    repo, output = Path(repo).resolve(), Path(output)
    output.mkdir(parents=False, exist_ok=False)
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", f"{revision}^{{commit}}"], text=True
    ).strip()
    if actual != revision:
        raise ValueError("Commit mismatch")
    archive = output / f"{name}-{revision}.tar"
    with archive.open("xb") as stream:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                "--format=tar",
                f"--prefix={name}-{revision}/",
                revision,
            ],
            stdout=stream,
            check=True,
        )
    rows = []
    with tarfile.open(archive, "r:") as tar:
        if tar.pax_headers.get("comment") != revision:
            raise ValueError("git archive commit header differs")
        for member in tar:
            if member.isfile():
                size, sha = hash_stream(tar.extractfile(member))
                rows.append(
                    {
                        "member": member.name,
                        "type": "file",
                        "bytes": size,
                        "sha256": sha,
                    }
                )
            elif member.issym():
                data = os.fsencode(member.linkname)
                rows.append(
                    {
                        "member": member.name,
                        "type": "symlink",
                        "link_target": member.linkname,
                        "bytes": len(data),
                        "sha256": digest_bytes(data),
                    }
                )
            elif not member.isdir():
                raise ValueError("Unexpected git archive member type")
    value = {
        "schema": "muxpd-exact-git-source-export-v1",
        "repository_path": str(repo),
        "commit": revision,
        "tree": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", f"{revision}^{{tree}}"], text=True
        ).strip(),
        "archive": identity(archive),
        "members": rows,
        "scope": "Exact committed tree only, no Git history. Runtime dirty changes require separately identified patch/file evidence; installed binary artifacts are not reconstructed by this source archive.",
    }
    write_new(output / "source-manifest.json", value)
    return {"commit": revision, "archive": value["archive"], "members": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    run = subs.add_parser("freeze-run")
    run.add_argument("--run", type=Path, required=True)
    support = subs.add_parser("freeze-support")
    support.add_argument("--selection", type=Path, required=True)
    for sub in (run, support):
        sub.add_argument("--capsule-id", required=True)
        sub.add_argument("--output", type=Path, required=True)
    archive = subs.add_parser("build")
    archive.add_argument("--manifest", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    verify = subs.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--archive-sha256", required=True)
    source = subs.add_parser("export-source")
    source.add_argument("--repo", type=Path, required=True)
    source.add_argument("--revision", required=True)
    source.add_argument("--name", required=True)
    source.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-run":
        result = freeze_run(args.run, args.capsule_id, args.output)
    elif args.command == "freeze-support":
        result = freeze_support(args.selection, args.capsule_id, args.output)
    elif args.command == "build":
        result = build(args.manifest, args.output)
    elif args.command == "verify":
        result = verify_archive(
            args.archive, json.loads(args.manifest.read_text()), args.archive_sha256
        )
    else:
        result = export_source(args.repo, args.revision, args.name, args.output)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("members", "terminal")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
