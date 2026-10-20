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
    (tmp_path / "cache").chmod(0o700)
    path = directory / "status.json"
    path.write_bytes(b'{"valid": false}\n')
    return {
        "tag_name": tag,
        "draft": True,
        "prerelease": False,
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
    assert manifest["releases"][0]["destination_release"] is None
    assert manifest["releases"][0]["original_tag"] == record["tag_name"]


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


def test_draft_and_prerelease_flags_are_preserved():
    assert MODULE.visibility_flags({"draft": True, "prerelease": True}) == [
        "--draft",
        "--prerelease",
    ]
    assert MODULE.visibility_flags({"draft": False, "prerelease": False}) == []


def test_missing_visibility_fails_closed():
    with pytest.raises(ValueError, match="Missing release visibility"):
        MODULE.visibility_flags({})


def test_draft_cannot_be_silently_published():
    with pytest.raises(ValueError, match="visibility mismatch"):
        MODULE.check_visibility(
            {"draft": True, "prerelease": False},
            {"draft": False, "prerelease": False},
        )


def test_corrected_release_visibility_receipt():
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (
            root / "archive/dev-hub-releases-20261020/visibility-verification.json"
        ).read_text()
    )
    rows = receipt["releases"]
    assert len(rows) == 3
    assert sum(r["assets_verified"] for r in rows) == 78
    assert sum(r["destination_draft"] for r in rows) == 2
    for row in rows:
        assert row["source_draft"] == row["destination_draft"]
        assert row["source_prerelease"] == row["destination_prerelease"]


def test_final_receipt_uses_stable_destination_api_ids():
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (root / "archive/dev-hub-releases-20261020/final-verification.json").read_text()
    )
    assert len(receipt["releases"]) == 3
    assert sum(len(row["assets"]) for row in receipt["releases"]) == 78
    for row in receipt["releases"]:
        assert row["destination_api_url"].endswith(str(row["destination_release_id"]))
        assert len(row["assets"]) == row["assets_verified"]
        assert all(
            asset["api_url"].startswith(
                "https://api.github.com/repos/vLLM-HUST/vllm-hust-benchmark/releases/assets/"
            )
            for asset in row["assets"]
        )


def test_draft_cache_must_be_private(tmp_path, monkeypatch):
    record = source(tmp_path)
    (tmp_path / "cache").chmod(0o755)
    monkeypatch.setattr(MODULE, "releases", lambda repo: [record])
    with pytest.raises(ValueError, match="private cache"):
        MODULE.migrate(tmp_path / "cache", tmp_path / "manifest.json", True)
