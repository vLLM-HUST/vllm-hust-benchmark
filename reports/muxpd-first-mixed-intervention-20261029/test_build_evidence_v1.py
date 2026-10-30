"""Integrity tests for terminal evidence retention and reproducible archives."""

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("build_evidence_v1.py")
    assert path.is_file(), "Reproducible terminal evidence packager is not implemented"
    spec = importlib.util.spec_from_file_location("evidence_package", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def fixture(tmp_path, passed=False):
    root = tmp_path / "campaign"
    run = root / "runs-v1" / "attempt-01"
    run.mkdir(parents=True)
    controller = {
        "finished_at": "2026-10-29T00:00:00+00:00",
        "terminal_process_proven": True,
        "passed": passed,
        "exit_code": 0 if passed else 1,
        "owned_members_after_cleanup": [],
    }
    (run / "controller.json").write_text(json.dumps(controller))
    (run / "raw.bin").write_bytes(bytes(range(256)))
    (run / "compiler-cache").mkdir()
    (run / "compiler-cache" / "excluded.bin").write_bytes(b"cache")
    return root, run


def test_failure_retained_and_cache_excluded(tmp_path):
    m = module()
    root, _run = fixture(tmp_path)
    manifest = m.make_manifest({"campaign": root})
    assert manifest["attempts"][0]["passed"] is False
    assert manifest["attempts"][0]["exit_code"] == 1
    assert {r["name"] for r in manifest["files"]} == {
        "campaign/runs-v1/attempt-01/controller.json",
        "campaign/runs-v1/attempt-01/raw.bin",
    }
    assert manifest["excluded_paths"] == ["campaign/runs-v1/attempt-01/compiler-cache/"]


def test_nonterminal_and_live_owned_rejected(tmp_path):
    m = module()
    root, run = fixture(tmp_path)
    c = json.loads((run / "controller.json").read_text())
    c["terminal_process_proven"] = False
    (run / "controller.json").write_text(json.dumps(c))
    with pytest.raises(ValueError, match="terminal"):
        m.make_manifest({"campaign": root})
    c["terminal_process_proven"] = True
    c["owned_members_after_cleanup"] = [123]
    (run / "controller.json").write_text(json.dumps(c))
    with pytest.raises(ValueError, match="owned"):
        m.make_manifest({"campaign": root})


def test_symlink_rejected(tmp_path):
    m = module()
    root, run = fixture(tmp_path)
    (run / "alias").symlink_to(run / "raw.bin")
    with pytest.raises(ValueError, match="symlink"):
        m.make_manifest({"campaign": root})


def test_reproducible_archive_verified_and_tampering_rejected(tmp_path):
    m = module()
    root, run = fixture(tmp_path)
    manifest = m.make_manifest({"campaign": root})
    path = tmp_path / "manifest.json"
    path.write_bytes(m.encoded(manifest))
    a = tmp_path / "a.tar.gz"
    b = tmp_path / "b.tar.gz"
    m.build(path, a)
    m.build(path, b)
    assert a.read_bytes() == b.read_bytes()
    assert m.verify(a)["member_count"] == 3
    with pytest.raises(FileExistsError):
        m.build(path, a)
    (run / "raw.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        m.build(path, tmp_path / "changed.tar.gz")


def test_added_or_removed_evidence_never_silently_selected(tmp_path):
    m = module()
    root, run = fixture(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_bytes(m.encoded(m.make_manifest({"campaign": root})))
    (run / "extra.log").write_text("must retain")
    with pytest.raises(ValueError, match="selection"):
        m.build(path, tmp_path / "extra.tar.gz")
    (run / "extra.log").unlink()
    (run / "raw.bin").unlink()
    with pytest.raises(ValueError, match="selection"):
        m.build(path, tmp_path / "missing.tar.gz")


@pytest.mark.parametrize("alteration", ["changed", "missing"])
def test_received_archive_rejects_altered_or_missing_payload(tmp_path, alteration):
    m = module()
    root, _run = fixture(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(m.encoded(m.make_manifest({"campaign": root})))
    original = tmp_path / "original.tar.gz"
    received = tmp_path / "received.tar.gz"
    m.build(manifest, original)
    with (
        tarfile.open(original, "r:gz") as source,
        tarfile.open(received, "w:gz") as target,
    ):
        for member in source:
            payload = source.extractfile(member).read()
            if member.name.endswith("raw.bin"):
                if alteration == "missing":
                    continue
                payload = b"x" * len(payload)
            target.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="changed|Missing"):
        m.verify(received)
