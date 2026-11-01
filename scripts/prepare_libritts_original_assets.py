"""Acquire the original LibriTTS test-other archive outside the checkout.

Offline plan by default. Explicit download has a byte budget; interrupted files
are retained and can only resume against the same owned acquisition metadata.
No archive extraction, neural inference, synthesized prompt or score is produced.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/libritts-original-assets.json"


def verify(path, spec):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != spec["bytes"]:
        raise ValueError("exact original regular archive required")
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            sha.update(chunk)
            md5.update(chunk)
    if sha.hexdigest() != spec["sha256"] or md5.hexdigest() != spec["md5"]:
        raise ValueError("original LibriTTS publisher/reader checksum mismatch")


def download(root, spec, *, max_bytes, resume=False, opener=urlopen):
    root = Path(root).resolve()
    if root.is_relative_to(ROOT):
        raise ValueError("raw speech must remain outside the checkout")
    if max_bytes < spec["bytes"] or spec["path"] != "test-other.tar.gz":
        raise ValueError("budget must cover the pinned original test-other archive")
    root.mkdir(parents=True, exist_ok=True)
    final = root / spec["path"]
    partial = root / (spec["path"] + ".partial")
    marker = root / "libritts-acquisition.private.json"
    identity = {"source_id": "LIBRITTS", "root": str(root), "dataset": spec}
    if resume:
        if marker.is_symlink() or json.loads(marker.read_text("utf-8")) != identity:
            raise ValueError("matching owned acquisition record required")
    else:
        if any(p.exists() or p.is_symlink() for p in (marker, partial, final)):
            raise ValueError(
                "inspect existing files; explicit matching resume required"
            )
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(identity, handle)
    if final.exists():
        verify(final, spec)
        return final
    if final.is_symlink() or partial.is_symlink():
        raise ValueError("archive paths cannot be symlinks")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > spec["bytes"]:
        raise ValueError("partial exceeds original byte budget")
    if offset < spec["bytes"]:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with opener(Request(spec["url"], headers=headers), timeout=60) as response:
            if offset:
                expected = f"bytes {offset}-{spec['bytes'] - 1}/{spec['bytes']}"
                if (
                    response.status != 206
                    or response.headers.get("Content-Range") != expected
                ):
                    raise ValueError("server did not honor exact original resume range")
            elif response.status != 200:
                raise ValueError("whole original response required")
            with partial.open("ab" if partial.exists() else "xb") as handle:
                size = offset
                while chunk := response.read(4 * 1024**2):
                    if size == 0 and not chunk.startswith(b"\x1f\x8b"):
                        raise ValueError("original gzip magic required")
                    size += len(chunk)
                    if size > spec["bytes"]:
                        raise ValueError("response exceeds original byte budget")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        if size != spec["bytes"]:
            raise ValueError("incomplete original download retained")
    verify(partial, spec)
    os.link(partial, final)
    partial.unlink()
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--action", choices=("plan", "download", "verify"), default="plan"
    )
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    spec = json.loads(PINS.read_text("utf-8"))["dataset"]
    if args.action == "download":
        download(args.data_root, spec, max_bytes=args.max_bytes, resume=args.resume)
    elif args.action == "verify":
        verify(args.data_root / spec["path"], spec)
    print(
        json.dumps(
            {
                "source_id": "LIBRITTS",
                "split": "test-other",
                "action": args.action,
                "status": "PLAN_ONLY"
                if args.action == "plan"
                else "ORIGINAL_ARCHIVE_VERIFIED_NOT_NATIVE_EXECUTION",
                "bytes": spec["bytes"],
                "raw_audio_in_git": False,
                "model_calls": 0,
                "source_checkbox_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
