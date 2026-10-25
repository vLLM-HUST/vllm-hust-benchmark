"""Plan or pack exact qualification bytes; never include a virtual environment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Unsafe archive member: {name}")


def ci_sources(path: Path) -> dict[str, Path]:
    """Accept only the separately frozen terminal CPU/CI inventory."""
    inventory = json.loads(path.read_text())
    if inventory.get("status") != "terminal":
        raise ValueError("CPU/CI inventory is not terminal")
    result = {"validation/input-inventory.json": path}
    for row in inventory["files"]:
        if row.get("live") is not False or not isinstance(row.get("sha256"), str):
            raise ValueError("CPU/CI source is live or has no digest")
        source = Path(row["path"])
        if (
            not source.is_absolute()
            or not source.is_relative_to("/tmp")
            or source.is_symlink()
        ):
            raise ValueError("CPU/CI source must be an explicit ordinary /tmp file")
        raw = source.read_bytes()
        if len(raw) != row["bytes"] or digest(raw) != row["sha256"]:
            raise ValueError("Frozen CPU/CI source differs")
        member = "validation/cpu-ci/" + source.relative_to("/tmp").as_posix()
        safe_member(member)
        if member in result:
            raise ValueError("Duplicate CPU/CI source")
        result[member] = source
    return result


def mapping(
    final: Path,
    failed: Path,
    cpu: Path,
    producer_root: Path,
    numerics: Path | None = None,
    release_hardware: Path | None = None,
    release_cpu: Path | None = None,
    ci_inventory: Path | None = None,
) -> dict[str, Path]:
    result = {}

    def add(name: str, source: Path, expected: str | None = None):
        safe_member(name)
        if name in result or source.is_symlink() or not source.is_file():
            raise ValueError(f"Duplicate or nonregular source: {name}")
        if expected is not None and digest(source.read_bytes()) != expected:
            raise ValueError(f"Captured SHA differs: {name}")
        result[name] = source.resolve()

    roots = [
        ("qualification/final", final),
        ("qualification/failed-startup", failed),
    ]
    if numerics is not None:
        roots.append(("numerical-study", numerics))
    if (release_hardware is None) != (release_cpu is None):
        raise ValueError("Release hardware and CPU roots must be supplied together")
    if release_hardware is not None:
        roots.append(("qualification/release", release_hardware))
    for prefix, root in roots:
        if not json.loads((root / "qualification.json").read_text()).get(
            "finished_at_utc"
        ):
            raise ValueError(f"Active qualification/study cannot be archived: {root}")
        for source in sorted(root.rglob("*")):
            if "__pycache__" in source.relative_to(root).parts:
                continue
            if source.is_symlink():
                raise ValueError(f"Unexpected symlink: {source}")
            if source.is_file():
                add(f"{prefix}/{source.relative_to(root).as_posix()}", source)
    cpu_receipt = json.loads((cpu / "validation.json").read_text())
    cpu_files = json.loads((cpu / "receipt-files.json").read_text())["files"]
    selected = {item["path"]: item["sha256"] for item in cpu_files}
    selected["receipt-files.json"] = None
    selected.setdefault("validation.json", None)
    for item in cpu_receipt["test_sources"]:
        selected[f"tests/{item['name']}"] = item["sha256"]
    for name, expected in selected.items():
        safe_member(name)
        add(f"cpu/{name}", cpu / name, expected)
    for package, package_record in cpu_receipt["packages"].items():
        for kind in ("wheel", "sdist"):
            artifact = package_record[kind]
            source = Path(artifact["path"])
            add(f"distributions/{package}/{source.name}", source, artifact["sha256"])
    # These are the three external clients/checker named in qualification argv.
    commit = json.loads((final / "qualification.json").read_text())["git_commit"]
    for name in (
        "smoke_serving.py",
        "exercise_decoupled.py",
        "check_decoupled_trace.py",
    ):
        pinned = subprocess.run(
            ["git", "show", f"{commit}:scripts/{name}"],
            cwd=producer_root,
            check=True,
            capture_output=True,
        ).stdout
        add(f"producers/{name}", producer_root / "scripts" / name, digest(pinned))
    for name in ("archive_qualification.py", "extract_qualification_facts.py"):
        add(f"analysis/{name}", Path(__file__).parent / name)
    if numerics is not None:
        add(
            "analysis/extract_numerical_study.py",
            Path(__file__).parent / "extract_numerical_study.py",
        )
    if release_cpu is not None:
        receipt = json.loads((release_cpu / "validation.json").read_text())
        selected = {
            item["path"]: item["sha256"]
            for item in json.loads((release_cpu / "receipt-files.json").read_text())[
                "files"
            ]
        }
        selected["receipt-files.json"] = None
        for item in receipt["test_sources"]:
            selected[f"tests/{item['name']}"] = item["sha256"]
        for name, expected in selected.items():
            safe_member(name)
            add(f"cpu/release/{name}", release_cpu / name, expected)
        tests = json.loads((release_cpu / "installed-manager-tests.json").read_text())
        for item in tests["test_sources"]:
            path = Path(item["path"])
            add(f"cpu/release/manager-tests/{path.name}", path, item["sha256"])
        for package, value in receipt["packages"].items():
            for kind in ("wheel", "sdist"):
                artifact = value[kind]
                path = Path(artifact["path"])
                add(
                    f"distributions/release/{package}/{path.name}",
                    path,
                    artifact["sha256"],
                )
        artifact = receipt["manager_hardware_runtime_bridge"]["hardware_wheel"]
        path = Path(artifact["path"])
        add(f"distributions/hardware-manager/{path.name}", path, artifact["sha256"])
        add(
            "analysis/extract_release_facts.py",
            Path(__file__).parent / "extract_release_facts.py",
        )
    if ci_inventory is not None:
        for member, source in ci_sources(ci_inventory).items():
            add(member, source)
    return result


def make_index(sources: dict[str, Path]) -> dict:
    return {
        "schema": "muxpd-ecpa-qualification-archive-v1",
        "scope": "Exact original bytes including failed startup and optional terminal numerical study; no venv, __pycache__, active runs or performance claim.",
        "files": [
            {
                "member": name,
                "captured_path": str(path),
                "bytes": len(raw),
                "sha256": digest(raw),
            }
            for name, path in sorted(sources.items())
            for raw in [path.read_bytes()]
        ],
    }


def verify(archive: Path, index: dict) -> dict:
    expected = {row["member"]: row for row in index["files"]}
    if len(expected) != len(index["files"]):
        raise ValueError("Duplicate index member")
    seen = set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            safe_member(member.name)
            if (
                not member.isfile()
                or member.name in seen
                or member.name not in expected
            ):
                raise ValueError(f"Unexpected archive member: {member.name}")
            source = stream.extractfile(member)
            if source is None:
                raise ValueError("Missing member content")
            raw = source.read()
            row = expected[member.name]
            if len(raw) != row["bytes"] or digest(raw) != row["sha256"]:
                raise ValueError(f"Archive content mismatch: {member.name}")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Missing archive members")
    raw = archive.read_bytes()
    return {
        "verified": True,
        "files": len(seen),
        "archive": archive.name,
        "bytes": len(raw),
        "sha256": digest(raw),
        "scope": "Every decompressed member matched the index; no source overwritten and no raw request re-executed.",
    }


def pack(archive: Path, index: dict) -> dict:
    with archive.open("xb") as out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT
            ) as stream:
                for row in index["files"]:
                    raw = Path(row["captured_path"]).read_bytes()
                    if len(raw) != row["bytes"] or digest(raw) != row["sha256"]:
                        raise ValueError(
                            f"Source changed after indexing: {row['member']}"
                        )
                    info = tarfile.TarInfo(row["member"])
                    info.size, info.mode, info.mtime = len(raw), 0o644, 0
                    stream.addfile(info, io.BytesIO(raw))
    return verify(archive, index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path)
    parser.add_argument("--failed", type=Path)
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--producer-root", type=Path)
    parser.add_argument(
        "--numerics",
        type=Path,
        help="Optional terminal study, separately named in the archive",
    )
    parser.add_argument("--release-hardware", type=Path)
    parser.add_argument("--release-cpu", type=Path)
    parser.add_argument(
        "--ci-inventory",
        type=Path,
        help="Frozen terminal CPU/CI input list with exact bytes/SHA",
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--pack", action="store_true", help="Otherwise only write the exact-byte index"
    )
    parser.add_argument("--verify", type=Path, metavar="ARCHIVE")
    parser.add_argument(
        "--index", type=Path, help="For source-independent archive verification"
    )
    args = parser.parse_args()
    if args.verify:
        if not args.index or args.pack or args.output_directory:
            parser.error("--verify requires only --index")
        print(
            json.dumps(
                verify(args.verify, json.loads(args.index.read_text())), indent=2
            )
        )
        return
    if (
        not all(
            (
                args.final,
                args.failed,
                args.cpu,
                args.producer_root,
                args.output_directory,
            )
        )
        or args.index
    ):
        parser.error("Provide all four source roots and a fresh --output-directory")
    sources = mapping(
        args.final,
        args.failed,
        args.cpu,
        args.producer_root,
        args.numerics,
        args.release_hardware,
        args.release_cpu,
        args.ci_inventory,
    )
    index = make_index(sources)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    (args.output_directory / "archive-index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    if args.pack:
        receipt = pack(
            args.output_directory / "muxpd-ecpa-qualification-20261025.tar.gz", index
        )
        (args.output_directory / "archive-verification.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
