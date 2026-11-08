"""Original asset getter controls using temporary fixtures, no downloads."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_mlagentbench_imdb_assets as module


def fixture_profile(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    files = {
        "assets/test.parquet": {
            "bytes": 8,
            "sha256": hashlib.sha256(b"original").hexdigest(),
        }
    }
    profile.write_text(json.dumps({"files": files}))
    monkeypatch.setattr(module, "PROFILE", profile)
    return SimpleNamespace(
        output=tmp_path / "external", action="plan", max_bytes=None, resume=False
    )


def test_default_plan_creates_nothing(tmp_path, monkeypatch):
    args = fixture_profile(tmp_path, monkeypatch)
    result = module.run(args)
    assert result["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert result["selected_bytes"] == 8
    assert not args.output.exists()


def test_verify_reuses_existing_file_without_download(tmp_path, monkeypatch):
    args = fixture_profile(tmp_path, monkeypatch)
    args.action = "verify"
    p = args.output / "assets/test.parquet"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"original")
    monkeypatch.setattr(
        module, "acquire", lambda *a, **k: pytest.fail("unexpected download")
    )
    assert len(module.run(args)["verified"]) == 1
    p.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum differs"):
        module.run(args)


def test_acquire_requires_full_explicit_budget(tmp_path, monkeypatch):
    args = fixture_profile(tmp_path, monkeypatch)
    args.action = "acquire"
    args.max_bytes = 7
    with pytest.raises(ValueError, match="budget"):
        module.run(args)
    assert not args.output.exists()
