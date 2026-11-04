"""Package terminal TP2 cells with per-member hashes; verify without extraction."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from tp2_pilot import digest


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts:
        raise ValueError("Unsafe archive member")
    return name


def files_in(path, prefix):
    paths = sorted(path.rglob("*")) if path.is_dir() else [path]
    result = []
    for item in paths:
        if "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        if item.is_symlink():
            raise ValueError(("Symlink is not an evidence file", item))
        if item.is_file():
            name = (
                prefix + "/" + item.relative_to(path).as_posix()
                if path.is_dir()
                else prefix
            )
            result.append((item, safe_name(name)))
    return result


def verify_archive(path, expected):
    seen = set()
    decoder = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=decoder.stdout, mode="r|") as tar:
            for member in tar:
                safe_name(member.name)
                if (
                    not member.isfile()
                    or member.name in seen
                    or member.name not in expected
                ):
                    raise ValueError(("Unexpected member", member.name))
                seen.add(member.name)
                h = hashlib.sha256()
                with tar.extractfile(member) as source:
                    for block in iter(lambda: source.read(1048576), b""):
                        h.update(block)
                if (
                    member.size != expected[member.name]["size"]
                    or h.hexdigest() != expected[member.name]["sha256"]
                ):
                    raise ValueError(("Archive member differs", member.name))
        if decoder.wait() != 0 or seen != set(expected):
            raise ValueError("Incomplete archive")
    finally:
        decoder.stdout.close()
        if decoder.poll() is None:
            decoder.terminate()
        decoder.wait()


def build_archive(output, entries):
    names = [name for _, name in entries]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate archive names")
    members = {
        name: dict(size=path.stat().st_size, sha256=digest(path))
        for path, name in entries
    }
    with output.open("xb") as target:
        compressor = subprocess.Popen(
            ["zstd", "-T2", "-3", "-c"], stdin=subprocess.PIPE, stdout=target
        )
        try:
            with tarfile.open(
                fileobj=compressor.stdin, mode="w|", format=tarfile.PAX_FORMAT
            ) as tar:
                for path, name in entries:
                    info = tar.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o755 if path.name == "traceloom" else 0o644
                    with path.open("rb") as source:
                        tar.addfile(info, source)
            compressor.stdin.close()
            if compressor.wait() != 0:
                raise RuntimeError("Compression failed")
        finally:
            if compressor.poll() is None:
                compressor.terminate()
            compressor.wait()
    verify_archive(output, members)
    return dict(
        name=output.name,
        bytes=output.stat().st_size,
        sha256=digest(output),
        members=members,
        verification="every archived member streamed and hashed without extraction",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--support", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root, analysis, support = (
        a.campaign.resolve(),
        a.analysis.resolve(),
        a.support.resolve(),
    )
    summary = json.loads((analysis / "summary.json").read_text())
    if (
        summary["status"] != "COMPLETE_ABBA_STRUCTURAL_CONTROL"
        or len(summary["workers"]) != 8
    ):
        raise ValueError("Full eight-worker analysis required")
    a.output.mkdir(parents=True, exist_ok=False)
    archives = []
    for cell in summary["protocol"]["cells"]:
        name = cell["name"]
        if json.loads((root / (name + ".exit.json")).read_text())["returncode"] != 0:
            raise ValueError("Nonterminal or failed capture")
        entries = files_in(root / name, f"campaign/{name}")
        entries += files_in(root / (name + ".scripts"), f"campaign/{name}.scripts")
        for suffix in (".launch.json", ".exit.json", ".log"):
            entries += files_in(root / (name + suffix), f"campaign/{name}{suffix}")
        entries += files_in(analysis / name, f"analysis/{name}")
        archives.append(
            build_archive(a.output / f"traceloom-tp2-{name}-20261027.tar.zst", entries)
        )
        print("PACKAGED", name, archives[-1]["bytes"], flush=True)
    entries = files_in(support, "support") + files_in(
        root / "scripts", "campaign/scripts"
    )
    entries += files_in(root / "protocol.json", "campaign/protocol.json")
    for path in sorted(analysis.glob("*.json")):
        entries += files_in(path, "analysis/" + path.name)
    archives.append(
        build_archive(a.output / "traceloom-tp2-support-20261027.tar.zst", entries)
    )
    manifest = dict(
        schema="traceloom-tp2-custody/v1",
        archives=archives,
        exclusions=[
            "model weights",
            "compiler caches",
            "Python bytecode",
            "unrelated repository files",
        ],
        package_script_sha256=digest(__file__),
        summary_sha256=digest(analysis / "summary.json"),
    )
    with (a.output / "MANIFEST.json").open("x") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    with (a.output / "SHA256SUMS").open("x") as f:
        for archive in archives:
            f.write(archive["sha256"] + "  " + archive["name"] + "\n")
        f.write(digest(a.output / "MANIFEST.json") + "  MANIFEST.json\n")
    print("ALL_ARCHIVES_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
