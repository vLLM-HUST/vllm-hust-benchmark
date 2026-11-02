"""Synthetic integrity failures must not become accepted experiment evidence."""

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports/pyramidkv-standalone-effects-20261103"
)


def module():
    spec = importlib.util.spec_from_file_location(
        "effect_evidence", REPORT / "evidence.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def checksums(root):
    entries = [
        p for p in sorted(root.iterdir()) if p.name != "SHA256SUMS" and p.is_file()
    ]
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in entries
        )
    )


def capsule(root, entries=None, *, parts=False):
    entries = entries or [("run/response.json", b'{"output":"7"}\n', tarfile.REGTYPE)]
    content = io.BytesIO()
    inventory = {}
    with tarfile.open(fileobj=content, mode="w:gz") as archive:
        for name, data, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                info.linkname = "/tmp/never-extract-this"
            archive.addfile(info, io.BytesIO(data))
            inventory[name] = {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    blob = content.getvalue()
    if parts:
        middle = len(blob) // 2
        (root / "raw.tar.gz.part-000").write_bytes(blob[:middle])
        (root / "raw.tar.gz.part-001").write_bytes(blob[middle:])
    else:
        (root / "raw.tar.gz").write_bytes(blob)
    (root / "members.json").write_text(json.dumps(inventory))
    checksums(root)


@pytest.mark.parametrize("parts", [False, True])
def test_capsule_returns_original_bytes_without_extracting(tmp_path, parts):
    capsule(tmp_path, parts=parts)
    assert module().load_capsule(tmp_path) == {"run/response.json": b'{"output":"7"}\n'}
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "run/../escape", "run//file", "run\\file"]
)
def test_capsule_rejects_unsafe_member_even_with_matching_manifest(tmp_path, name):
    capsule(tmp_path, [(name, b"x", tarfile.REGTYPE)])
    with pytest.raises(ValueError, match="Unsafe member"):
        module().load_capsule(tmp_path)


def test_capsule_rejects_duplicate_member_even_with_matching_manifest(tmp_path):
    capsule(tmp_path, [("same", b"x", tarfile.REGTYPE)] * 2)
    with pytest.raises(ValueError, match="Duplicate member"):
        module().load_capsule(tmp_path)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.DIRTYPE, tarfile.LNKTYPE])
def test_capsule_rejects_nonregular_members(tmp_path, kind):
    capsule(tmp_path, [("entry", b"", kind)])
    with pytest.raises(ValueError, match="Nonregular member"):
        module().load_capsule(tmp_path)


@pytest.mark.parametrize("change", ["missing", "extra", "length", "digest"])
def test_capsule_rejects_member_inventory_corruption_after_outer_rehash(
    tmp_path, change
):
    capsule(tmp_path)
    path = tmp_path / "members.json"
    members = json.loads(path.read_text())
    if change == "missing":
        members.clear()
    elif change == "extra":
        members["absent"] = {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    elif change == "length":
        members["run/response.json"]["bytes"] += 1
    else:
        members["run/response.json"]["sha256"] = "0" * 64
    path.write_text(json.dumps(members))
    checksums(tmp_path)
    with pytest.raises(ValueError, match="Member (inventory|length|digest)"):
        module().load_capsule(tmp_path)


def test_capsule_rejects_changed_outer_analysis(tmp_path):
    capsule(tmp_path)
    (tmp_path / "analysis.json").write_text('{"throughput": 1}')
    checksums(tmp_path)
    (tmp_path / "analysis.json").write_text('{"throughput": 999}')
    with pytest.raises(ValueError, match="File digest mismatch: analysis.json"):
        module().load_capsule(tmp_path)


@pytest.mark.parametrize("change", ["duplicate", "omitted", "unsafe", "gap"])
def test_capsule_requires_complete_unique_safe_outer_inventory(tmp_path, change):
    capsule(tmp_path, parts=True)
    path = tmp_path / "SHA256SUMS"
    original = path.read_text()
    if change == "duplicate":
        path.write_text(original + original.splitlines()[0] + "\n")
    elif change == "omitted":
        path.write_text("\n".join(original.splitlines()[1:]) + "\n")
    elif change == "unsafe":
        path.write_text(original + "0" * 64 + "  ../outside\n")
    else:
        (tmp_path / "raw.tar.gz.part-001").rename(tmp_path / "raw.tar.gz.part-002")
        checksums(tmp_path)
    with pytest.raises(ValueError, match="(Checksum|File inventory|Archive part)"):
        module().load_capsule(tmp_path)


def test_capsule_rejects_duplicate_json_inventory_keys(tmp_path):
    capsule(tmp_path)
    path = tmp_path / "members.json"
    original = path.read_text()
    path.write_text(original[:-1] + "," + original[1:])
    checksums(tmp_path)
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        module().load_capsule(tmp_path)


@pytest.mark.parametrize("optimized", [False, True])
def test_real_loader_integrity_checks_survive_optimized_python(tmp_path, optimized):
    capsule(tmp_path)
    (tmp_path / "members.json").write_text("{}")
    command = [
        sys.executable,
        *(["-O"] if optimized else []),
        "-c",
        (
            "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from evidence import load_capsule; load_capsule(Path(sys.argv[2]))"
        ),
        str(REPORT),
        str(tmp_path),
    ]
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, env=env
    )
    assert result.returncode != 0
    assert "File digest mismatch: members.json" in result.stderr
