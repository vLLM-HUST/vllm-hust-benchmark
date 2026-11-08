"""Prepare fixed original publisher FLAC members; default is an offline plan.

Use server storage only. This prepares inputs, not model execution or scoring.
Original archive member hashes are checked; whole archives are not rehashed.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zlib

from verify_wavcaps_original_media import PROFILES, contained, verify

# Exact compressed-member offsets from the original ZIP/ZIP64 local headers.
# Each tuple is (part suffix, first byte, last byte inclusive).
RANGES = {
    "SoundBible": [
        ("zip", 267665225, 268127894),
        ("zip", 552457022, 552497112),
        ("zip", 573938493, 578029680),
    ],
    "BBC_Sound_Effects": [
        ("z02", 1247208470, 1249883774),
        ("z09", 4942868562, 4948489969),
        ("z24", 648468890, 651248012),
    ],
    "AudioSet_SL": [
        ("z05", 1038258841, 1038607604),
        ("z02", 4486733233, 4487153439),
        ("z03", 78321666, 78791430),
    ],
    "FreeSound": [
        ("z46", 465864540, 465901403),
        ("z38", 1679828388, 1682181263),
        ("z08", 2047708322, 2048289775),
    ],
}


def plan(profile):
    component = profile["component"]
    parts = {
        p["path"].rsplit(".", 1)[1]: p
        for p in profile.get("selected_archive_parts", [])
    }
    if component == "SoundBible":
        parts["zip"] = {
            "path": profile["archive_path"],
            "bytes": profile["archive_bytes"],
        }
    result = []
    for item, (part, start, end) in zip(
        profile["files"], RANGES[component], strict=True
    ):
        archive = parts[part]
        if not 0 <= start <= end < archive["bytes"] or end - start >= 8 * 1024**2:
            raise ValueError("publisher range exceeds bound")
        result.append(
            {
                "id": item["id"],
                "part": part,
                "start": start,
                "end": end,
                "part_bytes": archive["bytes"],
                "url": f"https://huggingface.co/datasets/{profile['media_repository']}/resolve/{profile['media_revision']}/{archive['path']}",
                "cache_name": f"{part}-range-{start}-{end}.private.bin",
            }
        )
    return result


def fetch(item, *, opener=urlopen):
    length = item["end"] - item["start"] + 1
    request = Request(
        item["url"],
        headers={
            "Range": f"bytes={item['start']}-{item['end']}",
            "Accept-Encoding": "identity",
        },
    )
    with opener(request, timeout=60) as response:
        if (
            response.status != 206
            or response.headers.get("Content-Range")
            != (f"bytes {item['start']}-{item['end']}/{item['part_bytes']}")
            or response.headers.get("Content-Encoding", "identity") != "identity"
        ):
            raise ValueError(
                "publisher did not return the exact unencoded bounded range"
            )
        raw = response.read(length + 1)
    if len(raw) != length:
        raise ValueError("publisher range length differs")
    return raw


def extract(raw, expected):
    decoder = zlib.decompressobj(-15)
    decoded = decoder.decompress(raw, expected["bytes"] + 1)
    if (
        not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
        or len(decoded) != expected["bytes"]
        or zlib.crc32(decoded) != expected["crc32"]
        or hashlib.sha256(decoded).hexdigest() != expected["sha256"]
    ):
        raise ValueError("original member decompression or identity mismatch")
    return decoded


def prepare(
    profile, *, metadata_root, media_root, cache_root=None, download=False, decode=False
):
    media_root = Path(media_root).resolve()
    media_root.mkdir(parents=True, exist_ok=True)
    for item, expected in zip(plan(profile), profile["files"], strict=True):
        # All output names come from the pinned basename-only media profiles.
        name = expected["path"]
        if Path(name).name != name or "\\" in name or ":" in name:
            raise ValueError("original media output must be a basename")
        output = media_root / name
        if output.exists():
            # Fail rather than overwrite any existing, incompatible user file.
            from verify_wavcaps_original_media import original_bytes

            original_bytes(contained(media_root, name), expected)
            continue
        if cache_root is not None:
            cached = contained(cache_root, item["cache_name"])
            with cached.open("rb") as stream:
                raw = stream.read(item["end"] - item["start"] + 2)
        elif download:
            raw = fetch(item)
        else:
            raise ValueError("explicit cache or download required")
        if len(raw) != item["end"] - item["start"] + 1:
            raise ValueError("compressed original range length differs")
        decoded = extract(raw, expected)
        with output.open("xb") as stream:
            stream.write(decoded)
    return verify(
        profile, metadata_root=metadata_root, media_root=media_root, decode=decode
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=PROFILES, default="SoundBible")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--cache-root", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()
    profile = json.loads(PROFILES[args.component].read_text("utf-8"))
    if not args.download and args.cache_root is None:
        print(
            json.dumps(
                {"component": args.component, "plan": plan(profile), "model_calls": 0}
            )
        )
        return
    if args.metadata_root is None or args.media_root is None:
        parser.error("preparation requires --metadata-root and --media-root")
    result = prepare(
        profile,
        metadata_root=args.metadata_root,
        media_root=args.media_root,
        cache_root=args.cache_root,
        download=args.download,
        decode=args.decode,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
