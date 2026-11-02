"""Asset acquisition cannot substitute for original reranking execution."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_msmarco_original_assets",
    ROOT / "scripts/prepare_msmarco_original_assets.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(raw=b"original fixture\n"):
    return {
        "name": "qrels.dev.tsv",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_profile_preserves_original_pools_and_distinguishes_snapshot_hashes():
    profile = json.loads(MODULE.PROFILE.read_text("utf-8"))
    assert profile["observed_original_structure"]["candidate_rows"] == 6668967
    assert profile["observed_original_structure"]["archive_member"] == "top1000.dev"
    assert "not a publisher-published" in profile["checksum_kind"]
    assert "entire published candidate pool" in profile["protocol"]
    assert len(profile["files"]) == 2


def test_existing_valid_asset_reused_without_network(tmp_path, monkeypatch):
    raw = b"original fixture\n"
    (tmp_path / "qrels.dev.tsv").write_bytes(raw)
    monkeypatch.setattr(MODULE, "urlopen", lambda *a, **kw: pytest.fail("network used"))
    MODULE.acquire_file(tmp_path, fixture(raw))


def test_invalid_existing_asset_is_not_overwritten(tmp_path):
    target = tmp_path / "qrels.dev.tsv"
    target.write_bytes(b"wrong original!!\n")
    with pytest.raises(ValueError):
        MODULE.acquire_file(tmp_path, fixture())
    assert target.read_bytes() == b"wrong original!!\n"


class Response(io.BytesIO):
    status = 200

    def geturl(self):
        return fixture()["url"]


def test_download_published_only_after_verification(tmp_path, monkeypatch):
    raw = b"original fixture\n"
    monkeypatch.setattr(MODULE, "urlopen", lambda *a, **kw: Response(raw))
    MODULE.acquire_file(tmp_path, fixture(raw))
    assert (tmp_path / "qrels.dev.tsv").read_bytes() == raw
    assert not (tmp_path / "qrels.dev.tsv.part").exists()


def test_bad_snapshot_keeps_partial_and_never_publishes(tmp_path, monkeypatch):
    raw = b"wrong fixture!!!\n"
    monkeypatch.setattr(MODULE, "urlopen", lambda *a, **kw: Response(raw))
    with pytest.raises(ValueError):
        MODULE.acquire_file(tmp_path, fixture())
    assert not (tmp_path / "qrels.dev.tsv").exists()
    assert (tmp_path / "qrels.dev.tsv.part").read_bytes() == raw


def test_partial_never_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "qrels.dev.tsv.part"
    path.write_bytes(b"previous failure")
    monkeypatch.setattr(MODULE, "urlopen", lambda *a, **kw: pytest.fail("network used"))
    with pytest.raises(FileExistsError):
        MODULE.acquire_file(tmp_path, fixture())
    assert path.read_bytes() == b"previous failure"


def test_default_is_offline_plan(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["prepare_msmarco_original_assets.py"])
    monkeypatch.setattr(MODULE, "urlopen", lambda *a, **kw: pytest.fail("network used"))
    MODULE.main()
    assert (
        json.loads(capsys.readouterr().out)["status"] == "OFFLINE_PLAN_NO_DATA_ACQUIRED"
    )


def test_download_requires_explicit_license_before_creating_directory(
    tmp_path, monkeypatch
):
    target = tmp_path / "not-created"
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_msmarco_original_assets.py", "--download", "--data-dir", str(target)],
    )
    with pytest.raises(SystemExit) as error:
        MODULE.main()
    assert error.value.code == 2
    assert not target.exists()
