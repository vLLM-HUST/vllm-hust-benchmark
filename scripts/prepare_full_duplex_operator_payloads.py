"""Unpack the two pinned publisher operator payloads without executing shell code.

Only the audited Makeself 2.5.0 single-gzip layout is supported. Both the outer
release ZIP and its .run member are hash-bound; the embedded payload checksum,
paths, file types and contained library links are checked before extraction. No installer,
license acceptance, global CANN update or neural execution occurs.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile

from verify_full_duplex_model_runtime import load_lock


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4194304), b""):
            result.update(block)
    return result.hexdigest()


def embedded_tar(data):
    if not data.startswith(
        b"#!/bin/bash\n# This script was generated using Makeself 2.5.0\n"
    ):
        raise ValueError("audited Makeself 2.5.0 wrapper required")
    prefix = data[:65536]

    def field(name, pattern):
        values = re.findall(rb"^" + name + rb'="(' + pattern + rb')"$', prefix, re.M)
        if len(values) != 1:
            raise ValueError("unique literal Makeself header field required")
        return values[0]

    skip = int(field(b"skip", rb"[0-9]+"))
    size = int(field(b"filesizes", rb"[0-9]+"))
    expected = field(b"SHA", rb"[a-f0-9]{64}").decode("ascii")
    if not 1 <= skip <= 2048 or size != int(field(b"totalsize", rb"[0-9]+")):
        raise ValueError("audited bounded single-stream header required")
    parts = data.split(b"\n", skip)
    if len(parts) != skip + 1:
        raise ValueError("truncated Makeself header")
    payload = parts[-1]
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("embedded publisher payload checksum differs")
    if not payload.startswith(b"\x1f\x8b"):
        raise ValueError("single gzip payload required")
    return payload


def extract_regular_payload(payload, output):
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 20000 or sum(m.size for m in members) > 512 * 1024**2:
            raise ValueError("operator payload exceeds extraction budget")
        names = set()
        by_name = {str(PurePosixPath(m.name)): m for m in members}
        links = {str(PurePosixPath(m.name)) for m in members if m.issym()}
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or str(path) in names
                or not (member.isfile() or member.isdir() or member.issym())
            ):
                raise ValueError(
                    "unsafe, duplicate or non-regular operator payload member"
                )
            names.add(str(path))
            if any(str(parent) in links for parent in path.parents):
                raise ValueError("operator member descends through a symlink")
            if member.issym():
                target = PurePosixPath(member.linkname)
                resolved = by_name.get(str(path.parent / target))
                if (
                    target.is_absolute()
                    or ".." in target.parts
                    or "\\" in member.linkname
                    or resolved is None
                    or not resolved.isfile()
                ):
                    raise ValueError(
                        "operator symlink must name a contained regular payload file"
                    )
        output.mkdir(parents=True, exist_ok=False)
        files = []
        for member in members:
            target = output / PurePosixPath(member.name)
            if not target.resolve().is_relative_to(output.resolve()):
                raise ValueError("operator member escapes output")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym():
                continue  # Links are created only after every regular file.
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as incoming, target.open("xb") as outgoing:
                content = incoming.read(member.size + 1)
                if len(content) != member.size:
                    raise ValueError("truncated operator member")
                outgoing.write(content)
            target.chmod(member.mode & 0o777)
            files.append(
                {
                    "path": str(target.relative_to(output)),
                    "bytes": member.size,
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        for member in members:
            if member.issym():
                target = output / PurePosixPath(member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(member.linkname)
                files.append(
                    {
                        "path": str(target.relative_to(output)),
                        "symlink_target": member.linkname,
                    }
                )
    return files


def prepare(lock, archive_root, output):
    selected = [
        (archive, member)
        for archive in lock["publisher_archives"]
        for member in archive["members"]
        if member["path"].endswith(".run")
    ]
    if len(selected) != 2:
        raise ValueError("two audited publisher operator payloads required")
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("operator payloads must remain outside repository")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "STARTED",
        "shell_or_installer_executed": False,
        "global_cann_modified": False,
        "neural_execution": False,
        "source_checkbox_eligible": False,
        "payloads": [],
    }
    try:
        for index, (spec, member) in enumerate(selected):
            source = archive_root / spec["name"]
            if source.is_symlink() or digest(source) != spec["sha256"]:
                raise ValueError("publisher release archive differs")
            with zipfile.ZipFile(source) as archive:
                if archive.namelist().count(member["path"]) != 1:
                    raise ValueError("unique pinned run member required")
                info = archive.getinfo(member["path"])
                if info.file_size != member["bytes"] or info.file_size > 128 * 1024**2:
                    raise ValueError("publisher run member exceeds pinned size")
                data = archive.read(info)
            if hashlib.sha256(data).hexdigest() != member["sha256"]:
                raise ValueError("publisher run member differs")
            files = extract_regular_payload(
                embedded_tar(data), output / f"payload-{index}"
            )
            report["payloads"].append(
                {"archive": spec["name"], "member": member["path"], "files": files}
            )
        report["status"] = "OPERATOR_PAYLOADS_EXTRACTED_NO_INSTALLER_EXECUTION"
    except Exception as error:
        report.update(
            status="FAILED", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        (output / "qualification-public.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "payloads": len(report["payloads"]),
                "neural_execution": False,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--publisher-archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    lock = load_lock(args.lock)
    if args.prepare:
        prepare(
            lock,
            args.publisher_archive_root.resolve(strict=True),
            args.output.resolve(),
        )
    else:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY_NO_EXTRACTION",
                    "payloads": 2,
                    "shell_or_installer_executed": False,
                    "neural_execution": False,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
