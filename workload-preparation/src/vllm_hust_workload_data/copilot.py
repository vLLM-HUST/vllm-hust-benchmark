"""Publisher-manifest qualification for original Copilot session metadata.

This is asset integrity checking, not trace analysis or request synthesis.
"""

from datetime import date
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def verify_session(row):
    """Check structural containers without flattening or inferring missing links."""
    if (
        not isinstance(row, dict)
        or set(row) != {"session_id", "date", "turns"}
        or not isinstance(row["session_id"], str)
        or not isinstance(row["date"], str)
        or not isinstance(row["turns"], list)
    ):
        raise ValueError("invalid original session structure; payload suppressed")
    for turn in row["turns"]:
        if (
            not isinstance(turn, dict)
            or not isinstance(turn.get("turn_id"), str)
            or not isinstance(turn.get("llm_calls"), list)
            or not isinstance(turn.get("tool_batches"), list)
            or any(not isinstance(call, dict) for call in turn["llm_calls"])
        ):
            raise ValueError("invalid original turn structure; payload suppressed")
        for batch in turn["tool_batches"]:
            if (
                not isinstance(batch, dict)
                or not isinstance(batch.get("function_calls"), list)
                or any(not isinstance(call, dict) for call in batch["function_calls"])
            ):
                raise ValueError("invalid original tool batch; payload suppressed")


def verify_original_directory(data_root, *, expected_days):
    """Read unchanged local publisher shards, with no copies or data writes.

    Expected dates must be explicitly pinned by the release caller. Every
    session is structurally checked and all shard hashes measured, not trusted
    from a previous local inspection receipt.
    """
    root = Path(data_root).resolve()
    days = list(expected_days)
    if not days or len(set(days)) != len(days):
        raise ValueError("distinct pinned publisher days required")
    expected_directories = {f"date={day}" for day in days}
    if (
        not root.is_dir()
        or {p.name for p in root.iterdir()} != expected_directories
        or any(p.is_symlink() or not p.is_dir() for p in root.iterdir())
    ):
        raise ValueError("original directory set differs from pinned release days")
    files, qualifications = [], []
    for day in days:
        directory = root / f"date={day}"
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("publisher manifest must be an original regular file")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("date") != day:
            raise ValueError("publisher manifest directory/date mismatch")
        measured = []
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise ValueError("original source contains a non-regular asset")
            if path.name == "manifest.json":
                continue
            if not path.name.endswith(".jsonl.gz"):
                raise ValueError("unexpected file in original publisher day")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            rows = 0
            with gzip.open(path, "rt", encoding="utf-8") as lines:
                for line in lines:
                    row = json.loads(line)
                    verify_session(row)
                    if row["date"] != day:
                        raise ValueError(
                            "original session date differs from publisher day"
                        )
                    rows += 1
            measured.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                    "rows": rows,
                }
            )
        qualifications.append(qualify_day_manifest(manifest, measured))
        files.extend(measured)
    return {"days": qualifications, "files": files, "model_calls": 0}


def extract_verified_archive(archive, output, *, expected_sha256, expected_bytes):
    """Safely expand a pinned publisher archive; never execute or alter sessions.

    A fresh directory is mandatory. Failures remain available for inspection;
    this routine does not remove or overwrite partial work. Only manifest and
    original gzip JSONL members are accepted. No decompressed copy is written.
    """
    archive, output = Path(archive), Path(output).resolve()
    if output.exists():
        raise ValueError("fresh original-member output required")
    if archive.stat().st_size != expected_bytes:
        raise ValueError("publisher archive size mismatch")
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
            raise ValueError("publisher archive checksum mismatch")
    output.mkdir(parents=True, mode=0o700)
    measured, manifests, names = [], [], set()
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.name
                or not (output / member.name).resolve().is_relative_to(output)
            ):
                raise ValueError("unsafe publisher member path")
            if member.isdir():
                continue
            if (
                not member.isfile()
                or len(relative.parts) != 2
                or not relative.parts[0].startswith("date=")
                or not (
                    relative.name == "manifest.json"
                    or relative.name.endswith(".jsonl.gz")
                )
                or member.name in names
            ):
                raise ValueError("unexpected or duplicate publisher member")
            names.add(member.name)
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            stream = bundle.extractfile(member)
            with stream, target.open("xb") as destination:
                while chunk := stream.read(1024 * 1024):
                    destination.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != member.size:
                raise ValueError("truncated publisher member")
            info = {"path": member.name, "bytes": size, "sha256": digest.hexdigest()}
            if relative.name == "manifest.json":
                manifest = json.loads(target.read_text(encoding="utf-8"))
                if relative.parts[0] != f"date={manifest.get('date')}":
                    raise ValueError("publisher manifest directory/date mismatch")
                manifests.append(manifest)
            else:
                rows = 0
                with gzip.open(target, "rt", encoding="utf-8") as lines:
                    for line in lines:
                        row = json.loads(line)
                        verify_session(row)
                        if relative.parts[0] != f"date={row['date']}":
                            raise ValueError("session directory/date mismatch")
                        rows += 1
                measured.append({**info, "rows": rows})
    if len(manifests) != 1:
        raise ValueError("one original day manifest required per release archive")
    qualification = qualify_day_manifest(manifests[0], measured)
    return {"qualification": qualification, "files": measured, "model_calls": 0}


def qualify_day_manifest(manifest: dict, members: list[dict]) -> dict:
    """Require every original shard, exact byte counts and original session counts.

    ``members`` contains independently measured path/bytes/rows metadata, never
    generated sessions. Archive compression bytes differ from inner shard bytes.
    """
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise ValueError("publisher day manifest must declare complete")
    day = manifest.get("date")
    try:
        if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("invalid publisher date") from None
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("publisher shard list is required")
    for field in ("n_sessions", "n_shards", "total_bytes"):
        if type(manifest.get(field)) is not int or manifest[field] < 0:
            raise ValueError("invalid publisher totals")
    expected = {}
    for shard in shards:
        if not isinstance(shard, dict) or any(
            type(shard.get(field)) is not int or shard[field] < 0
            for field in ("shard_idx", "size_bytes", "n_sessions")
        ):
            raise ValueError("invalid publisher shard metadata")
        path = f"date={day}/shard-{shard['shard_idx']:04d}.jsonl.gz"
        if path in expected:
            raise ValueError("duplicate publisher shard")
        expected[path] = (shard["size_bytes"], shard["n_sessions"])
    if (
        len(expected) != manifest["n_shards"]
        or sum(v[0] for v in expected.values()) != manifest["total_bytes"]
        or sum(v[1] for v in expected.values()) != manifest["n_sessions"]
    ):
        raise ValueError("publisher manifest totals disagree")
    observed = {}
    for member in members:
        if not isinstance(member, dict) or not isinstance(member.get("path"), str):
            raise ValueError("invalid measured member")
        path = member["path"]
        if path in observed or any(
            type(member.get(field)) is not int or member[field] < 0
            for field in ("bytes", "rows")
        ):
            raise ValueError("duplicate or invalid measured member")
        observed[path] = (member["bytes"], member["rows"])
    if observed != expected:
        raise ValueError("original shards do not match publisher manifest")
    return {
        "date": day,
        "shards": len(expected),
        "sessions": manifest["n_sessions"],
        "compressed_shard_bytes": manifest["total_bytes"],
    }
