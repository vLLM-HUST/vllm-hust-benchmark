import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_followup_inventory_and_bidkv_original_identity():
    snapshot = json.loads(
        (ROOT / "archive/mod-followup-20261023/public-inventory.json").read_text()
    )
    assert len(snapshot) == 66
    assert "vllm-hust-benchmark" not in snapshot
    assert sum(len(row["paths"]) for row in snapshot.values()) == 4169
    assert sum(len(row["assets"]) for row in snapshot.values()) == 27
    assert not any(
        a["draft"] for row in snapshot.values() for a in row["assets"].values()
    )
    path = (
        ROOT / "archive/mod-git-20261022/website-source-vllm-hust-bidkv/manifest.json"
    )
    raw = path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == (
            "2d9ba8f088e5b43b8b7ede5a2018264f405dcc575e887fd42de9c665134580e3"  # pragma: allowlist secret
        )
    )
    manifest = json.loads(raw)
    assert len(manifest["files"]) == 14
    for entry in manifest["files"]:
        assert snapshot["vllm-hust-bidkv"]["paths"][entry["path"]] == entry["git_blob"]
        original = (path.parent / "snapshot" / entry["path"]).read_bytes()
        assert len(original) == entry["bytes"]
        assert hashlib.sha256(original).hexdigest() == entry["sha256"]


def test_split_installation_assets_preserve_historical_identity():
    snapshot = json.loads(
        (ROOT / "archive/mod-followup-20261023/public-inventory.json").read_text()
    )
    historical = json.loads(
        (ROOT / "archive/mod-git-20261022/split-history-migration.json").read_text()
    )
    consumers = next(
        row
        for row in historical.values()
        if isinstance(row, dict) and "releases" in row
    )
    assets = snapshot["vllm-ascend-split-batch-hust"]["assets"]
    assert len(assets) == 12
    for release in consumers["releases"]:
        for asset in release["assets"]:
            current = assets[str(asset["id"])]
            assert current["name"] == asset["name"]
            assert current["bytes"] == asset["size"]
            assert current["digest"] == asset["digest"]
