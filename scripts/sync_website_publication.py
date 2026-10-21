"""Import public website data once, then publish only from this repository.

bootstrap preserves the current public bytes; it does not certify their upstream
experimental provenance. publish requires a commit containing the exact bytes.
"""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "website-publication" / "data"
MANIFEST = "benchmark-source-manifest.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run(website, mode, revision=None):
    destination = website / "data"
    if mode == "bootstrap":
        if SOURCE.exists():
            raise ValueError("Bootstrap already exists; edit benchmark data instead")
        SOURCE.mkdir(parents=True)
        for path in sorted(destination.rglob("*")):
            if path.is_file() and path.name != MANIFEST:
                target = SOURCE / path.relative_to(destination)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    if not SOURCE.is_dir():
        raise ValueError("No benchmark publication source")
    if mode in ("bootstrap", "pack"):
        for path in sorted(SOURCE.rglob("*")):
            if path.is_file() and path.stat().st_size > 1_000_000:
                compressed = gzip.compress(path.read_bytes(), mtime=0)
                parts = []
                for offset in range(0, len(compressed), 900_000):
                    part = compressed[offset : offset + 900_000]
                    name = f"{path.name}.publication.part-{offset // 900_000:03d}"
                    path.with_name(name).write_bytes(part)
                    parts.append(
                        {"name": name, "sha256": digest(part), "bytes": len(part)}
                    )
                path.with_name(path.name + ".publication.json").write_text(
                    json.dumps({"encoding": "gzip", "parts": parts}, indent=2) + "\n"
                )
                path.unlink()
    commit = None
    if revision:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
            cwd=ROOT,
            text=True,
        ).strip()
    if mode == "publish" and not commit:
        raise ValueError(
            "publish requires --revision; uncommitted sources cannot be published"
        )
    entries = []
    copies = []
    for path in sorted(SOURCE.rglob("*")):
        if not path.is_file():
            continue
        if ".publication.part-" in path.name:
            continue
        source_raw = path.read_bytes()
        packed = path.name.endswith(".publication.json")
        if packed:
            archive = json.loads(source_raw)
            if archive["encoding"] != "gzip":
                raise ValueError("Unsupported publication archive")
            chunks = []
            for part in archive["parts"]:
                if Path(part["name"]).name != part["name"]:
                    raise ValueError("Invalid archive part path")
                part_path = path.with_name(part["name"])
                chunk = part_path.read_bytes()
                if digest(chunk) != part["sha256"] or len(chunk) != part["bytes"]:
                    raise ValueError(f"Archive checksum mismatch: {part['name']}")
                if (
                    commit
                    and subprocess.check_output(
                        [
                            "git",
                            "show",
                            f"{commit}:{part_path.relative_to(ROOT).as_posix()}",
                        ],
                        cwd=ROOT,
                    )
                    != chunk
                ):
                    raise ValueError(f"Uncommitted archive part: {part['name']}")
                chunks.append(chunk)
            raw = gzip.decompress(b"".join(chunks))
        else:
            raw = source_raw
        relative = path.relative_to(SOURCE)
        if packed:
            relative = relative.with_name(
                relative.name.removesuffix(".publication.json")
            )
        repository_path = path.relative_to(ROOT).as_posix()
        if commit:
            committed = subprocess.check_output(
                ["git", "show", f"{commit}:{repository_path}"], cwd=ROOT
            )
            if committed != source_raw:
                raise ValueError(f"Uncommitted source change: {repository_path}")
        target = destination / relative
        if mode == "publish":
            copies.append((target, raw))
        elif not target.is_file() or target.read_bytes() != raw:
            raise ValueError(f"Website differs from benchmark: {relative}")
        entries.append(
            {
                "website_path": "data/" + relative.as_posix(),
                "benchmark_path": repository_path,
                "sha256": digest(raw),
                "source_sha256": digest(source_raw),
                "bytes": len(raw),
            }
        )
    expected = {entry["website_path"] for entry in entries}
    extra = {
        path.relative_to(website).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != MANIFEST
    } - expected
    if extra:
        raise ValueError(f"Unregistered website data: {sorted(extra)}")
    for target, raw in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest = {
        "schema": "benchmark-website-publication/v1",
        "repository": "vLLM-HUST/vllm-hust-benchmark",
        "commit": commit,
        "status": "published-source" if commit else "local-unpublished",
        "scope": "Website data files; original experiment provenance is retained inside each artifact.",
        "files": entries,
    }
    if mode != "check":
        (destination / MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
    print(
        f"{mode}: {len(entries)} files, {sum(e['bytes'] for e in entries)} bytes; {manifest['status']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bootstrap", "pack", "check", "publish"))
    parser.add_argument("--website", required=True, type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()
    run(args.website.resolve(), args.mode, args.revision)
