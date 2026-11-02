"""Read immutable experiment evidence without extracting or executing its files."""

import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(content: bytes):
    return json.loads(content, object_pairs_hook=unique_object)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(path.parts)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and path.as_posix() == name
    )


def load_capsule(root: Path) -> dict[str, bytes]:
    expected = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        require(match is not None, "Checksum entry is malformed or unsafe")
        sha, name = match.groups()
        require(name not in (".", "..", "SHA256SUMS"), "Checksum filename is unsafe")
        require(name not in expected, f"Checksum duplicate entry: {name}")
        expected[name] = sha
    actual = {
        p.name
        for p in root.iterdir()
        if (p.is_file() or p.is_symlink()) and p.name != "SHA256SUMS"
    }
    require(set(expected) == actual, "File inventory differs from SHA256SUMS")
    files = {}
    for name, sha in expected.items():
        path = root / name
        require(path.is_file() and not path.is_symlink(), f"Nonregular file: {name}")
        files[name] = path.read_bytes()
        require(digest(files[name]) == sha, f"File digest mismatch: {name}")
    require("members.json" in files, "Member inventory is missing")
    inventory = read_json(files["members.json"])
    require(isinstance(inventory, dict), "Member inventory must be an object")
    parts = sorted(name for name in files if name.startswith("raw.tar.gz.part-"))
    if parts:
        require(
            "raw.tar.gz" not in files, "Archive parts mixed with a complete archive"
        )
        require(
            parts == [f"raw.tar.gz.part-{i:03d}" for i in range(len(parts))],
            "Archive part sequence is incomplete",
        )
        packed = b"".join(files[name] for name in parts)
    else:
        require("raw.tar.gz" in files, "Archive parts or complete archive are missing")
        packed = files["raw.tar.gz"]
    raw = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(packed), mode="r:gz") as archive:
            for member in archive:
                name = member.name
                require(member.isfile(), f"Nonregular member: {name}")
                require(safe_member(name), f"Unsafe member: {name}")
                require(name not in raw, f"Duplicate member: {name}")
                raw[name] = archive.extractfile(member).read()
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"Unreadable archive: {exc}") from exc
    require(set(raw) == set(inventory), "Member inventory differs from archive")
    for name, content in raw.items():
        record = inventory[name]
        require(
            isinstance(record, dict), f"Member inventory entry is malformed: {name}"
        )
        require(len(content) == record.get("bytes"), f"Member length mismatch: {name}")
        require(
            digest(content) == record.get("sha256"), f"Member digest mismatch: {name}"
        )
    return raw
