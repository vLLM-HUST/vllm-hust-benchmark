"""Tiny archive fixtures for original-archive getter, not scientific validation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "ml_arxiv_assets", ROOT / "scripts/prepare_mlagentbench_arxiv_assets.py"
)
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def fixture(tmp_path, monkeypatch, name="arxiv/raw/tiny.txt"):
    archive = tmp_path / "arxiv.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(name, "tiny fixture only")
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "data": {
                    "path": archive.name,
                    "bytes": archive.stat().st_size,
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                },
                "zip_members": 1,
                "zip_uncompressed_bytes": len("tiny fixture only"),
                "digest_origin": "fixture, not real original data",
            }
        )
    )
    monkeypatch.setattr(MODULE, "PROFILE", profile)
    return SimpleNamespace(action="verify", data_root=tmp_path)


def test_plan_does_not_touch_assets():
    assert MODULE.prepare(SimpleNamespace(action="plan"))["native_tasks_run"] == 0


def test_full_archive_identity_and_crc_checked_without_extraction(
    tmp_path, monkeypatch
):
    args = fixture(tmp_path, monkeypatch)
    result = MODULE.prepare(args)
    assert result["all_original_zip_crc_passed"] is True
    assert result["source_checkbox_eligible"] is False
    assert not (tmp_path / "arxiv").exists()


def test_wrong_original_archive_identity_is_rejected(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    with (tmp_path / "arxiv.zip").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        MODULE.prepare(args)


@pytest.mark.parametrize("name", ["other/raw/file", "arxiv/../escape"])
def test_foreign_or_escaping_zip_members_rejected(tmp_path, monkeypatch, name):
    args = fixture(tmp_path, monkeypatch, name)
    with pytest.raises(ValueError, match="contract"):
        MODULE.prepare(args)


def test_real_profile_labels_observed_digest_not_publisher_checksum():
    profile = json.loads(
        (
            ROOT
            / "configs/workload-native/mlagentbench-ogbn-arxiv-original-assets.json"
        ).read_text()
    )
    assert "not a publisher-supplied" in profile["digest_origin"]
    assert profile["archive_is_full_original_not_sampled_graph"] is True
    assert profile["data_acquisition_does_not_complete_native_benchmark"] is True
