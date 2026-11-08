"""Prepare exact original dialogue audio, never invented requests or TTS outputs.

Default plan does not access data or network. The fixed profile preserves the
first three complete publisher validation dialogues, original files and order.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from urllib.request import Request, urlopen
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/dailytalk-original-audio-sample.json"


class OriginalRanges(io.RawIOBase):
    """Fail closed if the host ignores Range or the observed archive changes."""

    def __init__(self, archive, budget):
        self.archive = archive
        self.budget = budget
        self.position = self.transferred = 0
        self.cache = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid archive seek mode")
        target = (0, self.position, self.archive["bytes"])[whence] + offset
        if not 0 <= target <= self.archive["bytes"]:
            raise ValueError("archive seek out of bounds")
        self.position = target
        return target

    def read(self, n=-1):
        remaining = self.archive["bytes"] - self.position
        n = remaining if n < 0 else min(n, remaining)
        if not n:
            return b""
        for start, raw in self.cache:
            offset = self.position - start
            if 0 <= offset and offset + n <= len(raw):
                self.position += n
                return raw[offset : offset + n]
        if n > 8 * 1024**2 or self.transferred + n > self.budget:
            raise ValueError("bounded archive range budget exceeded")
        start, end = self.position, self.position + n - 1
        request = Request(
            self.archive["url"],
            headers={
                "Range": f"bytes={start}-{end}",
                "If-Unmodified-Since": self.archive["last_modified"],
            },
        )
        with urlopen(request, timeout=30) as response:
            if (
                response.status != 206
                or response.headers.get("Content-Range")
                != f"bytes {start}-{end}/{self.archive['bytes']}"
                or response.headers.get("Last-Modified")
                != self.archive["last_modified"]
            ):
                raise ValueError("original archive Range/version not honored")
            raw = response.read(n + 1)
            if len(raw) != n:
                raise ValueError("original archive range length differs")
        self.transferred += n
        self.position += n
        return raw

    def prefetch(self, start, end):
        self.seek(start)
        self.cache.append((start, self.read(end - start)))


def selected_ids(metadata_root, profile):
    source = metadata_root / "preprocessed_data/DailyTalk/val_phone.txt"
    if source.stat().st_size != profile["selection"]["validation_manifest_bytes"]:
        raise ValueError("original validation split size differs")
    raw = source.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != profile["selection"]["validation_manifest_sha256"]
    ):
        raise ValueError("original validation split identity differs")
    groups, identifiers = [], []
    for line in raw.decode("utf-8").splitlines():
        identity = line.split("|")[0]
        match = re.fullmatch(r"\d+_\d+_d(\d+)", identity)
        if not match:
            raise ValueError("original validation utterance identity malformed")
        identifiers.append(identity)
        if match[1] not in groups:
            groups.append(match[1])
    if groups[:3] != profile["selection"]["dialogue_ids"]:
        raise ValueError("fixed first3 complete dialogue selection differs")
    expected = {x for x in identifiers if x.rsplit("_d", 1)[1] in groups[:3]}
    actual = {
        PurePosixPath(x["path"]).stem
        for x in profile["files"]
        if x["path"].endswith(".wav")
    }
    if expected != actual:
        raise ValueError("selected dialogue turns missing or replaced")
    return expected


def verify_bytes(raw, item):
    if (
        len(raw) != item["bytes"]
        or hashlib.sha256(raw).hexdigest() != item["sha256"]
        or zlib.crc32(raw) != item["crc32"]
    ):
        raise ValueError("original selected member bytes differ")


def prepare(args):
    profile = json.loads(PROFILE.read_text("utf-8"))
    report = {
        "source_id": "DAILYTALK",
        "status": "PLAN_ONLY_NO_DOWNLOAD",
        "fixed_dialogue_ids": profile["selection"]["dialogue_ids"],
        "original_selected_files": len(profile["files"]),
        "original_selected_bytes": sum(x["bytes"] for x in profile["files"]),
        "network_bytes": 0,
        "model_calls": 0,
        "entire_archive_digest_verified": False,
        "native_synthesis_or_scoring_executed": False,
        "source_checkbox_eligible": False,
    }
    if args.action == "plan":
        return report
    destination = args.data_root.resolve()
    if destination.is_relative_to(ROOT):
        raise ValueError("external private audio root required")
    selected_ids(args.metadata_root.resolve(), profile)
    sys.path.insert(0, str(ROOT / "scripts"))
    from prepare_xsum_original_assets import bound_path

    missing = []
    for item in profile["files"]:
        path = bound_path(destination, "", item)
        if path.exists():
            if path.stat().st_size != item["bytes"]:
                raise ValueError("original selected member size differs")
            verify_bytes(path.read_bytes(), item)
        else:
            if path.with_name(path.name + ".partial").exists():
                raise ValueError("inspect retained partial before acquisition")
            missing.append(item)
    if missing and args.action == "verify":
        raise ValueError("missing original selected files; no replacement tasks")
    if missing:
        needed = sum(x["bytes"] for x in missing)
        if args.max_extracted_bytes is None or args.max_extracted_bytes < needed:
            raise ValueError(
                f"explicit extracted-file budget at least {needed} bytes required"
            )
        if args.max_network_bytes is None or args.max_network_bytes <= 0:
            raise ValueError("explicit network range budget required")
        parent = destination
        while not parent.exists():
            parent = parent.parent
        if (
            args.reserve_free_bytes < 0
            or shutil.disk_usage(parent).free - needed < args.reserve_free_bytes
        ):
            raise ValueError("insufficient free space for original selected files")
        remote = OriginalRanges(profile["archive"], args.max_network_bytes)
        with zipfile.ZipFile(remote) as archive:
            infos_all = archive.infolist()
            indexed = {x.filename: x for x in infos_all}
            if len(indexed) != len(infos_all):
                raise ValueError("duplicate original archive member names")
            groups = []
            for identity in report["fixed_dialogue_ids"]:
                group = [
                    x
                    for x in missing
                    if x["path"].startswith(f"dailytalk/data/{identity}/")
                ]
                if group:
                    groups.append(group)
            groups.extend(
                [[x] for x in missing if len(PurePosixPath(x["path"]).parts) == 2]
            )
            if sum(map(len, groups)) != len(missing):
                raise ValueError("unexpected fixed profile member layout")
            for group in groups:
                infos = [indexed[x["path"]] for x in group]
                for item, info in zip(group, infos):
                    if (
                        info.file_size != item["bytes"]
                        or info.CRC != item["crc32"]
                        or stat.S_ISLNK(info.external_attr >> 16)
                        or info.flag_bits & 1
                    ):
                        raise ValueError("original ZIP member identity/type differs")
                start = min(x.header_offset for x in infos)
                end = min(
                    profile["archive"]["bytes"],
                    max(
                        x.header_offset
                        + 30
                        + len(x.filename.encode())
                        + 65535
                        + x.compress_size
                        for x in infos
                    ),
                )
                remote.prefetch(start, end)
                for item, info in zip(group, infos):
                    raw = archive.read(info)
                    verify_bytes(raw, item)
                    target = bound_path(destination, "", item)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_name(target.name + ".partial")
                    with partial.open("xb") as stream:
                        stream.write(raw)
                    os.link(partial, target)
                    partial.unlink()
        report["network_bytes"] = remote.transferred
    report.update(
        status="ORIGINAL_SELECTED_AUDIO_VERIFIED_NOT_NATIVE_TTS",
        original_audio_files=sum(x["path"].endswith(".wav") for x in profile["files"]),
        complete_original_dialogue_turns_preserved=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    parser.add_argument("--max-network-bytes", type=int)
    parser.add_argument("--max-extracted-bytes", type=int)
    parser.add_argument("--reserve-free-bytes", type=int, default=256 * 1024**2)
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
