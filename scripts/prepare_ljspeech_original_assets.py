"""Acquire the original LJ Speech corpus externally; never run a model.

Resumption requires the matching private ownership record and an exact HTTP range.
Existing final/partial files are not overwritten. Official TorchAudio's archive
SHA256, not a guessed model/data checksum, establishes the downloaded identity.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/ljspeech-original-assets.json"


def verify(path, spec):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != spec["bytes"]:
        raise ValueError("original regular archive with exact size required")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError("original TorchAudio archive checksum differs")


def download(root, spec, *, max_bytes, resume=False, opener=urlopen):
    root = Path(root).resolve()
    if root.is_relative_to(ROOT):
        raise ValueError("raw speech must remain outside the checkout")
    if max_bytes < spec["bytes"]:
        raise ValueError("explicit budget must cover the complete original archive")
    if spec["path"] != "LJSpeech-1.1.tar.bz2":
        raise ValueError("literal original archive filename required")
    root.mkdir(parents=True, exist_ok=True)
    path = root / spec["path"]
    partial = path.with_name(path.name + ".partial")
    marker = root / "ljspeech-acquisition.private.json"
    identity_fields = ("path", "url", "bytes", "sha256")
    identity = {
        "source_id": "LJSPEECH",
        "data_root": str(root),
        "dataset": {key: spec[key] for key in identity_fields},
    }
    if resume:
        if marker.is_symlink():
            raise ValueError("matching private acquisition record required")
        saved = json.loads(marker.read_text(encoding="utf-8"))
        # Earlier records also retained descriptive provenance. Its wording is
        # not file identity; keep the record unchanged and compare exact pins.
        saved["dataset"] = {key: saved["dataset"][key] for key in identity_fields}
        if saved != identity:
            raise ValueError("matching private acquisition record required")
    else:
        if path.exists() or partial.exists() or marker.exists():
            raise ValueError("inspect existing assets; use explicit matching resume")
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(identity, handle, ensure_ascii=False)
    if path.exists():
        verify(path, spec)
        return path
    if partial.is_symlink():
        raise ValueError("partial archive cannot be a symlink")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > spec["bytes"]:
        raise ValueError("partial exceeds original archive size")
    if offset < spec["bytes"]:
        request = Request(
            spec["url"], headers={"Range": f"bytes={offset}-"} if offset else {}
        )
        with opener(request, timeout=60) as response:
            if offset and (
                response.status != 206
                or response.headers.get("Content-Range")
                != f"bytes {offset}-{spec['bytes'] - 1}/{spec['bytes']}"
            ):
                raise ValueError("server did not honor the exact original resume range")
            with partial.open("ab" if partial.exists() else "xb") as handle:
                size = offset
                first = offset == 0
                while chunk := response.read(4 * 1024**2):
                    if first and not chunk.startswith(b"BZh"):
                        raise ValueError("original bzip2 archive magic required")
                    first = False
                    size += len(chunk)
                    if size > spec["bytes"]:
                        raise ValueError("response exceeds original archive size")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        if size != spec["bytes"]:
            raise ValueError("incomplete original archive; preserve matching partial")
    verify(partial, spec)
    os.link(partial, path)
    partial.unlink()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--action", choices=["plan", "download", "verify"], default="plan"
    )
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    spec = pins["dataset"]
    reused = args.action == "download" and (args.data_root / spec["path"]).is_file()
    if args.action == "download":
        download(args.data_root, spec, max_bytes=args.max_bytes, resume=args.resume)
    elif args.action == "verify":
        verify(args.data_root / spec["path"], spec)
    print(
        json.dumps(
            {
                "source_id": "LJSPEECH",
                "release": pins["release"],
                "action": args.action,
                "status": "PLAN_ONLY"
                if args.action == "plan"
                else "ORIGINAL_ARCHIVE_SHA256_VERIFIED_NOT_NATIVE_EXECUTION",
                "bytes": spec["bytes"],
                "existing_final_archive_reused": reused,
                "original_rows_declared": pins["original_rows"],
                "raw_audio_in_git": False,
                "model_calls": 0,
                "source_checkbox_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
