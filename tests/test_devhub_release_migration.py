import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "release_migration",
    Path(__file__).resolve().parents[1] / "scripts/migrate_devhub_releases.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source(tmp_path):
    tag = "szyn-swebench-qwen35-test-evidence"
    directory = tmp_path / "cache" / tag
    directory.mkdir(parents=True)
    path = directory / "status.json"
    path.write_bytes(b'{"valid": false}\n')
    return {
        "tag_name": tag,
        "name": "INVALIDATED: test attempt",
        "id": 1,
        "html_url": "https://example.invalid/original",
        "body": "Invalid attempt.",
        "assets": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "digest": MODULE.digest(path),
                "browser_download_url": "https://example.invalid/status.json",
            }
        ],
    }


def test_invalidated_status_survives(tmp_path, monkeypatch):
    record = source(tmp_path)
    monkeypatch.setattr(
        MODULE, "releases", lambda repo: [record] if repo == MODULE.SOURCE else []
    )
    output = tmp_path / "manifest.json"
    MODULE.migrate(tmp_path / "cache", output, False)
    manifest = json.loads(output.read_text())
    assert manifest["releases"][0]["original_title"].startswith("INVALIDATED:")
    assert manifest["releases"][0]["remote_verified"] is False


def test_corrupt_download_is_not_published(tmp_path, monkeypatch):
    record = source(tmp_path)
    record["assets"][0]["digest"] = "sha256:" + "0" * 64
    monkeypatch.setattr(
        MODULE, "releases", lambda repo: [record] if repo == MODULE.SOURCE else []
    )
    monkeypatch.setattr(MODULE, "gh", lambda *args: b"")
    with pytest.raises(ValueError, match="Source asset mismatch"):
        MODULE.migrate(tmp_path / "cache", tmp_path / "manifest.json", True)


def test_error_diagnostics_do_not_expose_signed_urls(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"https://example.invalid/?sig=private"
        ),
    )
    with pytest.raises(RuntimeError) as error:
        MODULE.gh("release", "download")
    assert "sig=" not in str(error.value)


def test_existing_custody_snapshot_is_not_rewritten(tmp_path, monkeypatch):
    record = source(tmp_path)
    monkeypatch.setattr(
        MODULE, "releases", lambda repo: [record] if repo == MODULE.SOURCE else []
    )
    output = tmp_path / "manifest.json"
    MODULE.migrate(tmp_path / "cache", output, False)
    original = output.read_bytes()
    record["body"] = "Changed upstream notes"
    with pytest.raises(ValueError, match="rewriting history"):
        MODULE.migrate(tmp_path / "cache", output, False)
    assert output.read_bytes() == original
