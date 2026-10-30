"""Verify fixed original media bindings, not model inference or quality scores."""

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import zlib

PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/wavcaps-soundbible-original-fixed3-media.json"
)


def contained(root, name):
    base = Path(root).resolve()
    if not name or Path(name).is_absolute() or PureWindowsPath(name).drive:
        raise ValueError("original media path must be relative")
    path = (base / name.replace("\\", "/")).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError("original media missing or escaping root")
    return path


def original_bytes(path, expected):
    with path.open("rb") as stream:
        raw = stream.read(expected["bytes"] + 1)
    if (
        len(raw) != expected["bytes"]
        or hashlib.sha256(raw).hexdigest() != expected["sha256"]
    ):
        raise ValueError("original asset identity mismatch")
    return raw


def verify(profile, *, metadata_root, media_root, decode=False):
    raw = original_bytes(
        contained(metadata_root, profile["metadata"]["path"]), profile["metadata"]
    )
    original = json.loads(raw)
    rows = original["data"][:3]
    if original["num_captions_per_audio"] != 1 or [r["id"] for r in rows] != [
        r["id"] for r in profile["files"]
    ]:
        raise ValueError("fixed original first3 identities differ; never replace tasks")
    if decode:
        import soundfile
    results = []
    for row, item in zip(rows, profile["files"], strict=True):
        path = contained(media_root, item["path"])
        raw = original_bytes(path, item)
        if (
            zlib.crc32(raw) != item["crc32"]
            or raw[:4] != b"fLaC"
            or len(raw) < 42
            or raw[4] & 127
            or int.from_bytes(raw[5:8], "big") != 34
        ):
            raise ValueError("original FLAC header or ZIP CRC mismatch")
        packed = int.from_bytes(raw[18:26], "big")
        rate, channels, bits, frames = (
            packed >> 44,
            ((packed >> 41) & 7) + 1,
            ((packed >> 36) & 31) + 1,
            packed & ((1 << 36) - 1),
        )
        if (rate, channels, bits, frames) != (
            item["sample_rate"],
            item["channels"],
            item["bits_per_sample"],
            item["total_samples"],
        ):
            raise ValueError("original audio dimensions differ")
        if not rate or abs(frames / rate - row["duration"]) > 1 / rate + 1e-7:
            raise ValueError("original media duration does not match metadata")
        if decode:
            if bits != 16 or raw[26:42] == bytes(16):
                raise ValueError(
                    "this exact PCM16/MD5 qualification requires original nonempty MD5"
                )
            digest, decoded = hashlib.md5(), 0
            with soundfile.SoundFile(path, mode="r") as stream:
                if (
                    stream.format,
                    stream.subtype,
                    stream.samplerate,
                    stream.channels,
                ) != ("FLAC", "PCM_16", rate, channels):
                    raise ValueError("decoder differs from original stream info")
                for block in stream.blocks(
                    blocksize=65536, dtype="int16", always_2d=True
                ):
                    decoded += len(block)
                    digest.update(block.astype("<i2", copy=False).tobytes())
            if decoded != frames or digest.digest() != raw[26:42]:
                raise ValueError(
                    "decoded original PCM checksum or frame count mismatch"
                )
        results.append(
            {
                "original_id": row["id"],
                "bytes": len(raw),
                "sample_rate": rate,
                "channels": channels,
                "frames": frames,
                "full_decode_pcm_md5_verified": decode,
            }
        )
    return {
        "source_id": "WAVCAPS",
        "status": "FIXED3_ORIGINAL_MEDIA_VERIFIED_NOT_MODEL_INFERENCE",
        "component": "SoundBible",
        "files": results,
        "metadata_modified": False,
        "audio_modified": False,
        "full_decode_requested": decode,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Requires an existing soundfile decoder; no automatic install or neural inference",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                json.loads(PROFILE.read_text()),
                metadata_root=args.metadata_root,
                media_root=args.media_root,
                decode=args.decode,
            ),
            indent=2,
        )
    )
