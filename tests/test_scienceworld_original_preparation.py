"""Preparation is not a substitute for native agent task validation."""

import hashlib
import importlib.util
import base64
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "scienceworld_preparation",
    Path(__file__).parents[1] / "scripts/prepare_scienceworld_original_assets.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_plan_does_not_write(tmp_path):
    output = tmp_path / "absent"
    result = MODULE.prepare(output)
    assert not output.exists()
    assert result["model_calls"] == 0
    assert not result["source_checkbox_eligible"]
    assert len(result["files"]) == 15
    assert any(item["path"] == "requirements-dev.txt" for item in result["files"])


def test_author_react_plan_is_distinct_from_new_environment(tmp_path):
    result = MODULE.prepare(tmp_path / "absent", profile="author-react")
    assert result["revision"] == MODULE.REACT_REVISION
    assert result["revision"] != MODULE.REVISION
    assert len(result["files"]) == 12
    assert any(
        item["path"] == "goldpaths-all.zip"
        and item["original_path"] == "data_utils/goldpaths-all.zip"
        for item in result["files"]
    )
    assert not (tmp_path / "absent").exists()


def test_author_react_reuses_verified_bytes_without_network(tmp_path, monkeypatch):
    data = b"fixture author bytes"
    original = tmp_path / "original.zip"
    original.write_bytes(data)
    item = {
        "path": "goldpaths-all.zip",
        "size": len(data),
        "sha": hashlib.sha1(
            f"blob {len(data)}".encode() + bytes([0]) + data
        ).hexdigest(),
    }
    monkeypatch.setattr(MODULE, "REACT_FILES", [item])
    monkeypatch.setattr(
        MODULE, "urlopen", lambda *a, **k: pytest.fail("reuse must not download")
    )
    result = MODULE.prepare(
        tmp_path / "fresh",
        profile="author-react",
        execute=True,
        reuse_root=tmp_path,
        gold_archive=original,
    )
    assert result["source_checkbox_eligible"] is False
    assert (
        (tmp_path / "fresh/goldpaths-all.zip").read_bytes()
        == original.read_bytes()
        == data
    )


def test_original_identity_and_corruption(tmp_path):
    path = tmp_path / "original"
    data = b"original\n"
    path.write_bytes(data)
    item = {
        "size": len(data),
        "sha": hashlib.sha1(b"blob 9" + bytes([0]) + data).hexdigest(),
    }
    assert MODULE.verify(path, item) == hashlib.sha256(data).hexdigest()
    path.write_bytes(b"modified\n")
    with pytest.raises(ValueError, match="identity"):
        MODULE.verify(path, item)


def test_execute_preserves_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="fresh output"):
        MODULE.prepare(tmp_path, execute=True)


def test_reject_repository_destination():
    with pytest.raises(ValueError, match="outside repository"):
        MODULE.prepare(Path(__file__).parent / "forbidden", verify_only=True)


def test_execute_preserves_author_bytes_and_verifies_reuse(tmp_path, monkeypatch):
    data = b"unchanged author source\n"
    sha = hashlib.sha1(f"blob {len(data)}".encode() + bytes([0]) + data).hexdigest()
    item = {"path": "nested/runtime", "sha": sha, "size": len(data)}
    monkeypatch.setattr(MODULE, "FILES", [item])

    def fetch(request, timeout):
        assert request.full_url.endswith(sha)
        assert timeout == 90
        return io.BytesIO(
            json.dumps(
                {
                    "sha": sha,
                    "encoding": "base64",
                    "content": base64.b64encode(data).decode(),
                }
            ).encode()
        )

    monkeypatch.setattr(MODULE, "urlopen", fetch)
    output = tmp_path / "fresh"
    result = MODULE.prepare(output, execute=True)
    assert (output / item["path"]).read_bytes() == data
    assert not result["source_checkbox_eligible"]
    monkeypatch.setattr(
        MODULE, "urlopen", lambda *a, **k: pytest.fail("verification must be offline")
    )
    assert MODULE.prepare(output, verify_only=True)["files"] == result["files"]


def test_verify_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"data")
    link = tmp_path / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable on this host")
    with pytest.raises(ValueError, match="symlink"):
        MODULE.verify(link, {"size": 4, "sha": "irrelevant"})
