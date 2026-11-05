"""Asset identity and path controls, not native benchmark evidence."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_mlagentbench_imdb_offline as module


def test_asset_digest_and_containment(tmp_path):
    (tmp_path / "data").write_bytes(b"original")
    profile = {"files": {"data": {"sha256": hashlib.sha256(b"original").hexdigest()}}}
    assert module.verify_assets(tmp_path, profile) == tmp_path.resolve()
    (tmp_path / "data").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest differs"):
        module.verify_assets(tmp_path, profile)


def test_parent_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match="relative"):
        module.verify_assets(tmp_path, {"files": {"../outside": {"sha256": "unused"}}})


def test_original_binding_requires_fresh_separate_cache(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    port = tmp_path / "agent"
    (port / "MLAgentBench/benchmarks/imdb/env").mkdir(parents=True)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"files": {}}))
    monkeypatch.setattr(module, "PROFILE", profile)
    monkeypatch.setattr(module, "verify_port", lambda root: None)
    args = SimpleNamespace(
        assets_root=assets,
        port_root=port,
        cache_root=assets / "nested-cache",
        container_asset_root=tmp_path / "mount",
    )
    with pytest.raises(ValueError, match="separate"):
        module.bind(args)
    assert not args.cache_root.exists()


def test_cache_label_does_not_claim_a_publisher_commit():
    assert "not-publisher-git-revision" in module.LOCAL_CACHE_LABEL
