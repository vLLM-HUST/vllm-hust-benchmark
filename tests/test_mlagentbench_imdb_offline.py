"""Asset identity and path controls, not native benchmark evidence."""

import hashlib
import json
import os
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
        container_cache_root=tmp_path / "container-hf",
    )
    with pytest.raises(ValueError, match="separate"):
        module.bind(args)
    assert not args.cache_root.exists()


def test_cache_label_does_not_claim_a_publisher_commit():
    assert "not-publisher-git-revision" in module.LOCAL_CACHE_LABEL


@pytest.mark.parametrize(
    "missing", ["dataset-card/README.md", "assets/unsupervised-00000-of-00001.parquet"]
)
def test_incomplete_original_interface_rejected_before_creating_cache(
    tmp_path, monkeypatch, missing
):
    assets = tmp_path / "assets"
    assets.mkdir()
    port = tmp_path / "agent"
    (port / "MLAgentBench/benchmarks/imdb/env").mkdir(parents=True)
    names = {
        f"assets/{split}-00000-of-00001.parquet"
        for split in ["train", "test", "unsupervised"]
    } | {"dataset-card/README.md"}
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"files": {n: {} for n in names - {missing}}}))
    monkeypatch.setattr(module, "PROFILE", profile)
    monkeypatch.setattr(module, "verify_assets", lambda root, spec: root.resolve())
    monkeypatch.setattr(module, "verify_port", lambda root: None)
    args = SimpleNamespace(
        assets_root=assets,
        port_root=port,
        cache_root=tmp_path / "cache",
        container_asset_root=tmp_path / "container-assets",
        container_cache_root=tmp_path / "container-hf",
    )
    with pytest.raises(ValueError, match="all three original splits"):
        module.bind(args)
    assert not args.cache_root.exists()


@pytest.mark.skipif(
    os.name == "nt", reason="Linux native read-only symbolic-link layout"
)
def test_full_original_dataset_layout_is_bound_without_rewriting_metadata(
    tmp_path, monkeypatch
):
    assets = tmp_path / "assets"
    names = [
        f"assets/{split}-00000-of-00001.parquet"
        for split in ["train", "test", "unsupervised"]
    ] + ["dataset-card/README.md", "distilbert/config.json"]
    files = {}
    for name in names:
        p = assets / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fixture")
        files[name] = {"sha256": hashlib.sha256(b"fixture").hexdigest()}
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"files": files}))
    port = tmp_path / "agent"
    env = port / "MLAgentBench/benchmarks/imdb/env"
    env.mkdir(parents=True)
    original = env / "train.py"
    original.write_bytes(b"original source fixture")
    monkeypatch.setattr(module, "PROFILE", profile)
    monkeypatch.setattr(module, "verify_port", lambda p: None)
    cache = tmp_path / "cache"
    args = SimpleNamespace(
        assets_root=assets,
        port_root=port,
        cache_root=cache,
        container_asset_root=tmp_path / "container-assets",
        container_cache_root=tmp_path / "container-hf",
    )
    result = module.bind(args)
    assert result["original_dataset_card_and_three_split_layout_bound"] is True
    assert (env / "imdb").readlink() == args.container_cache_root / "dataset"
    assert (
        cache / "dataset/README.md"
    ).readlink() == args.container_asset_root / "dataset-card/README.md"
    assert {p.name for p in (cache / "dataset/plain_text").iterdir()} == {
        Path(n).name for n in names if n.endswith(".parquet")
    }
    assert original.read_bytes() == b"original source fixture"
