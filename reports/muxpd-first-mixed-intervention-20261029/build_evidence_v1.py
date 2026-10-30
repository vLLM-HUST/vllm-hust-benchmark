"""CPU-only immutable evidence packaging; no device, service or network calls.

Every actual attempt is retained, including terminal failures. The source manifest
freezes the selection. Build detects newly added/removed evidence and hashes the
exact bytes streamed into a deterministic gzip/tar. Verify needs only the archive.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path, PurePosixPath

EXCLUDED = {"compiler-cache", "__pycache__", ".pytest_cache", ".ruff_cache", ".git"}
HEX = re.compile(r"^[0-9a-f]{64}$")


def encoded(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"symlink source forbidden: {path}")
    if not path.is_file():
        raise ValueError(f"Nonregular source: {path}")
    return path


def safe(name):
    value = PurePosixPath(name)
    if (
        value.is_absolute()
        or ".." in value.parts
        or not value.parts
        or any(x in EXCLUDED for x in value.parts)
    ):
        raise ValueError(f"Unsafe archive member: {name}")


def collect(roots):
    files, excluded, attempts = {}, [], []
    for alias, supplied in sorted(roots.items()):
        safe(alias)
        root = Path(supplied).absolute()
        if any(p.is_symlink() for p in (root, *root.parents)) or not root.is_dir():
            raise ValueError(f"Invalid or symlink root: {root}")
        for run_root in sorted(root.glob("runs-*")):
            if not run_root.is_dir() or run_root.is_symlink():
                raise ValueError(f"Invalid runs root: {run_root}")
            for attempt in sorted(run_root.iterdir()):
                if not attempt.is_dir() or attempt.is_symlink():
                    raise ValueError(f"Unexpected attempt entry: {attempt}")
                path = regular(attempt / "controller.json")
                controller = json.loads(path.read_bytes())
                if (
                    not controller.get("finished_at")
                    or controller.get("terminal_process_proven") is not True
                    or type(controller.get("passed")) is not bool
                ):
                    raise ValueError(f"Attempt not proven terminal: {attempt}")
                if controller.get("owned_members_after_cleanup") != []:
                    raise ValueError(f"Attempt still has owned processes: {attempt}")
                attempts.append(
                    {
                        "name": f"{alias}/{attempt.relative_to(root).as_posix()}",
                        **{
                            k: controller.get(k)
                            for k in (
                                "started_at",
                                "finished_at",
                                "terminal_process_proven",
                                "passed",
                                "exit_code",
                                "owned_members_after_cleanup",
                                "cleanup_required",
                                "forced_termination",
                                "errors",
                            )
                        },
                    }
                )
        for directory, dirs, names in os.walk(root, followlinks=False):
            for name in sorted(dirs):
                path = Path(directory) / name
                if name in EXCLUDED:
                    excluded.append(f"{alias}/{path.relative_to(root).as_posix()}/")
                elif path.is_symlink():
                    raise ValueError(f"symlink directory forbidden: {path}")
            dirs[:] = sorted(name for name in dirs if name not in EXCLUDED)
            for name in sorted(names):
                path = regular(Path(directory) / name)
                member = f"{alias}/{path.relative_to(root).as_posix()}"
                safe(member)
                if member in files:
                    raise ValueError(f"Duplicate selection: {member}")
                files[member] = path
    return files, sorted(excluded), attempts


def make_manifest(roots, extras=None):
    files, excluded, attempts = collect(roots)
    rows = []
    for name, path in sorted(files.items()):
        rows.append(
            {
                "name": name,
                "source_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": digest(path),
                "kind": "root",
            }
        )
    for name, value in sorted((extras or {}).items()):
        safe(name)
        if name in files or name == "MANIFEST.json":
            raise ValueError(f"Duplicate/reserved member: {name}")
        path = regular(value)
        rows.append(
            {
                "name": name,
                "source_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": digest(path),
                "kind": "extra",
            }
        )
    rows.sort(key=lambda row: row["name"])
    return {
        "schema": "muxpd-first-mixed-evidence-source-manifest-v1",
        "roots": {k: str(Path(v).absolute()) for k, v in roots.items()},
        "attempts": attempts,
        "excluded_components": sorted(EXCLUDED),
        "excluded_paths": excluded,
        "files": rows,
        "file_count": len(rows),
        "source_bytes": sum(row["size_bytes"] for row in rows),
        "scope": "All terminal attempts retained without filtering on success. Packaging integrity is independent of quality/performance qualification. Earlier external dependencies are declared separately and are not claimed to be completely included.",
    }


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        chunk = self.stream.read(size)
        self.hash.update(chunk)
        self.count += len(chunk)
        return chunk


def info(name, size):
    safe(name)
    entry = tarfile.TarInfo(name)
    entry.size, entry.mode, entry.mtime = size, 0o644, 0
    entry.uid = entry.gid = 0
    entry.uname = entry.gname = ""
    return entry


def build(manifest_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest_bytes = regular(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    selected, excluded, attempts = collect(manifest["roots"])
    expected = {
        row["name"]: Path(row["source_path"])
        for row in manifest["files"]
        if row["kind"] == "root"
    }
    if (
        selected != expected
        or excluded != manifest["excluded_paths"]
        or attempts != manifest["attempts"]
    ):
        raise ValueError("Source selection/attempts changed since manifest")
    with (
        output.open("xb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
        ) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
        ) as archive,
    ):
        archive.addfile(
            info("MANIFEST.json", len(manifest_bytes)), io.BytesIO(manifest_bytes)
        )
        for index, row in enumerate(manifest["files"]):
            path = regular(row["source_path"])
            if path.stat().st_size != row["size_bytes"]:
                raise ValueError(f"Source changed size: {path}")
            with path.open("rb") as stream:
                reader = HashReader(stream)
                archive.addfile(info(row["name"], row["size_bytes"]), reader)
                if (
                    reader.count != row["size_bytes"]
                    or reader.hash.hexdigest() != row["sha256"]
                ):
                    raise ValueError(f"Source changed bytes: {path}")
            if index % 100 == 0:
                print(
                    json.dumps(
                        {"packaged": index + 1, "total": len(manifest["files"])}
                    ),
                    flush=True,
                )
    return {
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": digest(output),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_bytes": manifest["source_bytes"],
        "file_count": manifest["file_count"],
        "attempts": len(manifest["attempts"]),
    }


def verify(archive_path):
    seen = set()
    expected = None
    total = 0
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for entry in archive:
            safe(entry.name)
            if entry.name in seen or not entry.isfile():
                raise ValueError(f"Duplicate/nonregular member: {entry.name}")
            seen.add(entry.name)
            stream = archive.extractfile(entry)
            if expected is None:
                if entry.name != "MANIFEST.json":
                    raise ValueError("Manifest must be first")
                manifest = json.load(stream)
                expected = {row["name"]: row for row in manifest["files"]}
                if len(expected) != len(manifest["files"]):
                    raise ValueError("Duplicate manifest entries")
            else:
                if entry.name not in expected:
                    raise ValueError(f"Extra archive member: {entry.name}")
                row = expected[entry.name]
                value = hashlib.file_digest(stream, "sha256").hexdigest()
                if entry.size != row["size_bytes"] or value != row["sha256"]:
                    raise ValueError(f"Archive bytes changed: {entry.name}")
                total += entry.size
    if expected is None or seen != set(expected) | {"MANIFEST.json"}:
        raise ValueError("Missing archive members")
    return {
        "passed": True,
        "member_count": len(seen),
        "source_bytes": total,
        "archive_bytes": Path(archive_path).stat().st_size,
        "archive_sha256": digest(archive_path),
    }


def external_index(roots):
    selected, _, _ = collect(roots)
    included = {str(path): name for name, path in selected.items()}
    found = {}
    documents = {}

    def remember(path, sha, document):
        if (
            isinstance(path, str)
            and path.startswith("/")
            and isinstance(sha, str)
            and HEX.fullmatch(sha)
        ):
            row = found.setdefault(
                path,
                {
                    "declared_sha256": set(),
                    "declared_by": set(),
                    "included_member": included.get(path),
                },
            )
            row["declared_sha256"].add(sha)
            row["declared_by"].add(document)

    def scan(value, document):
        if isinstance(value, dict):
            for key in ("path", "source_path", "file"):
                remember(value.get(key), value.get("sha256"), document)
            for key, item in value.items():
                remember(
                    key,
                    item
                    if isinstance(item, str)
                    else item.get("sha256")
                    if isinstance(item, dict)
                    else None,
                    document,
                )
                scan(item, document)
        elif isinstance(value, list):
            for item in value:
                scan(item, document)

    for name, path in selected.items():
        relative = Path(name).parts[1:]
        if path.suffix == ".json" and not any(
            part.startswith("runs-") for part in relative
        ):
            value = json.loads(path.read_bytes())
            documents[name] = digest(path)
            scan(value, name)
    return {
        "schema": "muxpd-first-mixed-external-reference-index-v1",
        "scope": "Declared hashes extracted from retained non-run JSON reference/analysis/manifests. External bytes are not re-read or represented as included. Multiple historical declarations are preserved, not normalized. Original documents remain authoritative.",
        "source_documents": documents,
        "references": {
            path: {
                **row,
                "declared_sha256": sorted(row["declared_sha256"]),
                "declared_by": sorted(row["declared_by"]),
            }
            for path, row in sorted(found.items())
        },
        "reference_count": len(found),
        "external_count": sum(row["included_member"] is None for row in found.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("references", "plan", "build", "verify"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command in ("references", "plan"):
        config = json.loads(args.config.read_bytes())
        result = (
            external_index(config["roots"])
            if args.command == "references"
            else make_manifest(config["roots"], config["extras"])
        )
    elif args.command == "build":
        result = build(args.manifest, args.archive)
    else:
        result = verify(args.archive)
    if args.output:
        with args.output.open("xb") as stream:
            stream.write(encoded(result))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "files",
                    "roots",
                    "references",
                    "source_documents",
                    "attempts",
                    "excluded_paths",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
