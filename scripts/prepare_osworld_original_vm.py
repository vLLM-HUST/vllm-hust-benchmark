"""Acquire/inspect/extract the pinned original Ubuntu VM, never run a guest."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import zipfile

from prepare_mlagentbench_cifar_assets import acquire, verify

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/osworld-original-ubuntu-vm.json"
RESERVE = 256 * 1024**2


def qcow_header(header):
    if len(header) < 72 or header[:4] != b"QFI\xfb":
        raise ValueError("original member is not a complete QCOW2 header")
    version = struct.unpack_from(">I", header, 4)[0]
    backing_offset = struct.unpack_from(">Q", header, 8)[0]
    backing_size = struct.unpack_from(">I", header, 16)[0]
    virtual_bytes = struct.unpack_from(">Q", header, 24)[0]
    if version not in {2, 3} or not virtual_bytes or version == 3 and len(header) < 104:
        raise ValueError("supported original QCOW2 version and virtual size required")
    if backing_offset or backing_size:
        raise ValueError("external backing-file dependency requires separate review")
    return {
        "version": version,
        "virtual_bytes": virtual_bytes,
        "external_backing": False,
    }


def members(archive, expected):
    rows, names = [], set()
    for item in archive.infolist():
        original_name = item.orig_filename
        relative = PurePosixPath(item.filename)
        canonical = relative.as_posix().casefold()
        mode = item.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in original_name
            or "\0" in original_name
            or not relative.parts
            or ":" in relative.parts[0]
            or canonical in names
            or kind not in {0, stat.S_IFREG, stat.S_IFDIR}
            or item.flag_bits & 1
        ):
            raise ValueError("unsafe, duplicate, encrypted or special archive member")
        names.add(canonical)
        rows.append(item)
    selected = [x for x in rows if x.filename == expected and not x.is_dir()]
    if len(selected) != 1:
        raise ValueError("exact original VM member required")
    with archive.open(selected[0]) as stream:
        header = qcow_header(stream.read(104))
    return rows, header


def external(path):
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT):
        raise ValueError("keep original VM assets outside the repository")
    return resolved


def run(args):
    profile = json.loads(PROFILE.read_text("utf-8"))
    spec = profile["asset"]
    report = {
        "source_id": "OSWORLD",
        "status": "PLAN_ONLY_NO_DOWNLOAD_OR_VM_EXECUTION",
        "publisher_dataset_revision": profile["revision"],
        "archive_bytes": spec["bytes"],
        "expected_vm_member": profile["expected_vm_member"],
        "model_calls": 0,
        "native_tasks": 0,
        "source_checkbox_eligible": False,
    }
    if args.action == "plan":
        return report
    root = external(args.output)
    if args.action == "verify-image":
        image = profile["extracted_vm"]
        verified = verify(root / image["path"], image)
        with (root / image["path"]).open("rb") as stream:
            header = qcow_header(stream.read(104))
        return dict(
            report,
            status="ORIGINAL_PREPARED_VM_VERIFIED_NOT_VM_EXECUTION",
            image=verified,
            qcow2=header,
            image_identity_provenance=image["identity_provenance"],
            archive_reverified_this_call=False,
        )
    path = root / spec["path"]
    if args.action == "acquire":
        if args.max_bytes is None or args.max_bytes < spec["bytes"]:
            raise ValueError("explicit budget must cover the complete original archive")
        acquire(
            root,
            spec,
            max_bytes=args.max_bytes,
            resume=args.resume,
            source_id="OSWORLD",
        )
    report.update(
        status="ORIGINAL_ARCHIVE_VERIFIED_NOT_VM_EXECUTION", archive=verify(path, spec)
    )
    if args.action in {"acquire", "verify"}:
        return report
    with zipfile.ZipFile(path) as archive:
        entries, header = members(archive, profile["expected_vm_member"])
        total = sum(x.file_size for x in entries if not x.is_dir())
        report.update(
            status="ORIGINAL_ARCHIVE_STRUCTURE_INSPECTED_NOT_VM_EXECUTION",
            members=len(entries),
            unpacked_bytes=total,
            qcow2=header,
            all_members_crc_verified=False,
        )
        if args.action == "inspect":
            return report
        if (
            args.extract_output is None
            or args.max_unpacked_bytes is None
            or args.max_unpacked_bytes < total
        ):
            raise ValueError(
                "fresh extraction output and complete uncompressed byte budget required"
            )
        destination = external(args.extract_output)
        if destination.exists() or destination.is_symlink():
            raise ValueError(
                "fresh extraction output required; retain existing attempts"
            )
        parent = destination.parent
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < total + RESERVE:
            raise ValueError("insufficient extraction capacity and reserve")
        destination.mkdir(parents=True, mode=0o700)
        identities = []
        for item in entries:
            target = destination.joinpath(*PurePosixPath(item.filename).parts)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            with archive.open(item) as source, target.open("xb") as output:
                for chunk in iter(lambda: source.read(4 * 1024**2), b""):
                    size += len(chunk)
                    if size > item.file_size:
                        raise ValueError("original member exceeds declared size")
                    digest.update(chunk)
                    output.write(chunk)
            if size != item.file_size:
                raise ValueError("incomplete original extraction retained")
            identities.append(
                {"path": item.filename, "bytes": size, "sha256": digest.hexdigest()}
            )
        report.update(
            status="ORIGINAL_VM_EXTRACTED_NOT_BOOTED_OR_NATIVE_DELIVERED",
            all_members_crc_verified=True,
            extracted_files=identities,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--action",
        choices=["plan", "acquire", "verify", "inspect", "extract", "verify-image"],
        default="plan",
    )
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extract-output", type=Path)
    parser.add_argument("--max-unpacked-bytes", type=int)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
