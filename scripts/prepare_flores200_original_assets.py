"""Pinned original FLORES-200 archive and NLLB files, never model execution.

The archive SHA is observed from Meta HTTPS, not a published publisher checksum.
Keep assets external. NLLB license is noncommercial; missing codes are not replaced.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from urllib.request import Request, urlopen

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/flores200-author-pins.json"
)


def digest(path, algorithm="sha256", prefix=b""):
    value = hashlib.new(algorithm)
    value.update(prefix)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def contained(root, part, relative):
    root = root.resolve()
    parent = (root / part).resolve()
    path = (parent / relative).resolve()
    if (
        not parent.is_relative_to(root)
        or Path(relative).is_absolute()
        or not path.is_relative_to(parent)
    ):
        raise ValueError("contained original asset path required")
    return path


def verify(path, spec):
    if path.stat().st_size != spec["bytes"]:
        raise ValueError("pinned asset size mismatch; never overwrite")
    if "sha256" in spec:
        valid = digest(path) == spec["sha256"]
    else:
        valid = (
            digest(path, "sha1", f"blob {spec['bytes']}\0".encode()) == spec["git_blob"]
        )
    if not valid:
        raise ValueError("pinned original payload identity mismatch; never overwrite")


def inspect_archive(path, dataset):
    counts = {split: {} for split in dataset["split_rows"]}
    seen, total = set(), 0
    with tarfile.open(path) as bundle:
        for member in bundle:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError("unsafe original archive member")
            if member.isdir():
                continue
            normalized = str(name)
            if normalized in seen or not normalized.startswith("flores200_dataset/"):
                raise ValueError("duplicate or unexpected archive member")
            seen.add(normalized)
            total += member.size
            if (
                member.size > 4 * 1024 * 1024
                or total > 256 * 1024 * 1024
                or len(seen) > 2048
            ):
                raise ValueError(
                    "archive exceeds inspected safety bounds; no partial completion"
                )
            raw = bundle.extractfile(member).read()
            if len(raw) != member.size:
                raise ValueError("incomplete original archive member")
            raw.decode("utf-8")
            match = re.fullmatch(
                r"flores200_dataset/(dev|devtest)/([a-z]{3}_[A-Z][a-z]{3})\.(dev|devtest)",
                normalized,
            )
            if match:
                split, language, suffix = match.groups()
                if split != suffix or split not in counts:
                    raise ValueError("original split/member mismatch")
                lines = raw.split(b"\n")
                if lines[-1] == b"":
                    lines.pop()
                if len(lines) != dataset["split_rows"][split]:
                    raise ValueError("original physical line count mismatch")
                counts[split][language] = len(lines)
    if any(
        len(languages) != dataset["language_code_count"]
        for languages in counts.values()
    ):
        raise ValueError("original language coverage incomplete")
    language_sets = [set(languages) for languages in counts.values()]
    if any(languages != language_sets[0] for languages in language_sets):
        raise ValueError("original parallel language sets differ")
    return {
        "regular_members": len(seen),
        "splits": {
            split: {"files": len(languages), "rows_each": dataset["split_rows"][split]}
            for split, languages in counts.items()
        },
        "extracted": False,
    }


def jobs(pins, parts, endpoint):
    if pins.get("source_id") != "FLORES200":
        raise ValueError("FLORES200 source profile required")
    result = []
    for part in dict.fromkeys(parts):
        if part == "data":
            result.append((part, pins["dataset"], pins["dataset"]["url"]))
        elif part == "model":
            model = pins["model"]
            result.extend(
                (
                    part,
                    spec,
                    f"{endpoint}/{model['repo_id']}/resolve/{model['revision']}/{spec['path']}",
                )
                for spec in model["files"]
            )
        else:
            raise ValueError("only original data/model preparation supported")
    return result


def fetch(path, spec, url):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.is_symlink():
        raise ValueError("partial asset must not be a symlink")
    size = 0
    with (
        urlopen(
            Request(
                url, headers={"User-Agent": "original-FLORES200-asset-preparation"}
            ),
            timeout=60,
        ) as response,
        partial.open("wb") as stream,
    ):
        for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
            size += len(block)
            if size > spec["bytes"]:
                raise ValueError("response exceeds pinned original size")
            stream.write(block)
    verify(partial, spec)
    # Atomic no-clobber publication, including if another process created path.
    path.hardlink_to(partial)
    partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parts", nargs="+", choices=["data", "model"], default=["data", "model"]
    )
    parser.add_argument(
        "--hf-endpoint",
        choices=["https://huggingface.co", "https://hf-mirror.com"],
        default="https://huggingface.co",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    pins = json.loads(PINS.read_text("utf-8"))
    selected = jobs(pins, args.parts, args.hf_endpoint)
    if args.verify_only:
        for part, spec, _ in selected:
            verify(contained(args.output, part, spec["path"]), spec)
        result = (
            inspect_archive(
                contained(args.output, "data", pins["dataset"]["path"]), pins["dataset"]
            )
            if "data" in args.parts
            else None
        )
        print(
            json.dumps(
                {
                    "status": "ORIGINAL_ASSETS_VERIFIED_READ_ONLY",
                    "files": len(selected),
                    "archive": result,
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    receipt_path = args.output / "preparation.private.json"
    pin_sha = digest(PINS)
    if args.output.exists():
        if not args.resume or not receipt_path.is_file():
            raise ValueError("fresh output or matching explicit resume required")
        prior = json.loads(receipt_path.read_text("utf-8"))
        if prior.get("source_id") != "FLORES200" or prior.get("pins_sha256") != pin_sha:
            raise ValueError("resume source/pin identity mismatch")
    else:
        args.output.mkdir(mode=0o700, parents=True)
    receipt = {
        "source_id": "FLORES200",
        "status": "PREPARING_ORIGINAL_ASSETS",
        "pins_sha256": pin_sha,
        "files": [],
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "publisher_archive_checksum_claim": False,
        "raw_payloads_published": False,
    }
    try:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        for part, spec, url in selected:
            path = contained(args.output, part, spec["path"])
            reused = path.exists()
            if reused:
                verify(path, spec)
            else:
                fetch(path, spec, url)
            receipt["files"].append(
                {"part": part, "path": spec["path"], "verified": True, "reused": reused}
            )
            receipt_path.write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
        if "data" in args.parts:
            receipt["archive"] = inspect_archive(
                contained(args.output, "data", pins["dataset"]["path"]), pins["dataset"]
            )
        receipt["status"] = "PINNED_ASSETS_PREPARED_NOT_NATIVE_EXECUTION"
    except Exception as exc:
        receipt.update(
            status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__, error=str(exc)
        )
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
