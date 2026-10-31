"""Synthetic asset integrity fixtures; never scientific data or programs."""

import hashlib
import importlib.util
import json
from pathlib import Path
import zlib

import pytest

SPEC = importlib.util.spec_from_file_location(
    "science_original_assets",
    Path(__file__).resolve().parents[1]
    / "scripts/verify_scienceagentbench_original_assets.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def identity(path, raw):
    return {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "original_crc32": zlib.crc32(raw),
    }


def test_plan_does_not_read_raw_assets(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(entry, "verify", lambda *a: pytest.fail("raw read"))
    assert (
        entry.main(
            [
                "--metadata-root",
                str(tmp_path / "missing"),
                "--asset-root",
                str(tmp_path / "absent"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "PLAN_ONLY"
    assert not list(tmp_path.iterdir())


def test_modified_same_size_asset_rejected(tmp_path):
    raw = b"synthetic-data"
    spec = identity("fixture.bin", raw)
    (tmp_path / "fixture.bin").write_bytes(raw)
    entry.verify_file(tmp_path, spec)
    (tmp_path / "fixture.bin").write_bytes(b"X" * len(raw))
    with pytest.raises(ValueError, match="SHA256"):
        entry.verify_file(tmp_path, spec)


def test_no_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="contained"):
        entry.verify_file(tmp_path, identity("../other", b""))


def test_whole_dataset_directory_not_only_registered_files(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    assets = tmp_path / "assets"
    metadata.mkdir()
    (metadata / "fixture.bin").write_bytes(b"metadata")
    dataset = assets / "benchmark/datasets/fixture"
    dataset.mkdir(parents=True)
    path = dataset / "original.csv"
    path.write_bytes(b"dataset")
    profile = {
        "metadata": identity("fixture.bin", b"metadata"),
        "files": [identity("benchmark/datasets/fixture/original.csv", b"dataset")],
        "complete_dataset_roots": ["benchmark/datasets/fixture"],
        "instance_ids": [1],
    }
    file = tmp_path / "profile.json"
    file.write_text(json.dumps(profile))
    monkeypatch.setattr(entry, "PROFILE", file)
    assert entry.verify(metadata, assets)["verified_asset_files"] == 1
    (dataset / "extra.csv").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="extra"):
        entry.verify(metadata, assets)


def test_profile_covers_actual_original_first3_not_whole102():
    profile = json.loads(entry.PROFILE.read_text())
    assert profile["instance_ids"] == [1, 2, 3]
    assert len(profile["files"]) == 14
    assert sum(x["bytes"] for x in profile["files"]) == 1006008
    assert len(profile["complete_dataset_roots"]) == 3
