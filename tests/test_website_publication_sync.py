import importlib.util
import json
import random
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_registries_are_current_in_website_publication():
    publication = ROOT / "website-publication/data"
    registry = ROOT / "leaderboard-data/official-targets.json"
    checksum = ROOT / "leaderboard-data/official-targets.sha256"
    assert (publication / "official_targets.json").read_bytes() == registry.read_bytes()
    assert (
        publication / "official_targets.sha256"
    ).read_bytes() == checksum.read_bytes()

    canonical_index = json.loads(
        (
            ROOT
            / "leaderboard-data/dataset-validation/dataset_validation_index_v1.json"
        ).read_text()
    )
    website_index = json.loads(
        (publication / "dataset_validation_index_v1.json").read_text()
    )
    assert (
        website_index["default_scenario_id"] == canonical_index["default_scenario_id"]
    )


def module(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/sync_website_publication.py"
    spec = importlib.util.spec_from_file_location("publication", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    result.ROOT = tmp_path / "benchmark"
    result.SOURCE = result.ROOT / "website-publication/data"
    return result


def test_bootstrap_checks_bytes_and_rejects_silent_website_edits(tmp_path):
    sync = module(tmp_path)
    website = tmp_path / "website"
    (website / "data").mkdir(parents=True)
    (website / "data/example.json").write_text('{"value": 4}\n')
    sync.run(website, "bootstrap")
    sync.run(website, "check")
    manifest = json.loads((website / "data" / sync.MANIFEST).read_text())
    assert manifest["status"] == "local-unpublished"
    assert manifest["commit"] is None
    with pytest.raises(ValueError, match="already exists"):
        sync.run(website, "bootstrap")
    with pytest.raises(ValueError, match="requires --revision"):
        sync.run(website, "publish")
    (website / "data/example.json").write_text('{"value": 5}\n')
    with pytest.raises(ValueError, match="differs"):
        sync.run(website, "check")


def test_unknown_website_data_is_rejected(tmp_path):
    sync = module(tmp_path)
    website = tmp_path / "website"
    (website / "data").mkdir(parents=True)
    (website / "data/example.json").write_text("{}\n")
    sync.run(website, "bootstrap")
    (website / "data/extra.json").write_text("{}\n")
    with pytest.raises(ValueError, match="Unregistered"):
        sync.run(website, "check")


@pytest.mark.parametrize("mutation", ["missing", "commit", "status", "hash", "files"])
def test_check_rejects_missing_or_incorrect_source_manifest(tmp_path, mutation):
    sync = module(tmp_path)
    website = tmp_path / "website"
    (website / "data").mkdir(parents=True)
    (website / "data/example.json").write_text("{}\n")
    sync.run(website, "bootstrap")
    path = website / "data" / sync.MANIFEST
    manifest = json.loads(path.read_text())
    if mutation == "missing":
        path.unlink()
    else:
        if mutation == "commit":
            manifest["commit"] = "a" * 40
        elif mutation == "status":
            manifest["status"] = "published-source"
        elif mutation == "hash":
            manifest["files"][0]["sha256"] = "0" * 64
        else:
            manifest["files"] = []
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        sync.run(website, "check")


def test_large_source_is_lossless_and_archive_tampering_is_rejected(tmp_path):
    sync = module(tmp_path)
    website = tmp_path / "website"
    (website / "data").mkdir(parents=True)
    raw = random.Random(42).randbytes(1_200_000)
    (website / "data/large.bin").write_bytes(raw)
    sync.run(website, "bootstrap")
    sync.run(website, "check")
    assert not (sync.SOURCE / "large.bin").exists()
    assert all(p.stat().st_size <= 1_000_000 for p in sync.SOURCE.iterdir())
    parts = sorted(sync.SOURCE.glob("*.publication.part-*"))
    assert len(parts) == 2
    parts[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        sync.run(website, "check")


def test_check_verifies_published_revision_without_rewriting_manifest(
    tmp_path, monkeypatch
):
    sync = module(tmp_path)
    website = tmp_path / "website"
    (website / "data").mkdir(parents=True)
    raw = b"{}\n"
    (website / "data/example.json").write_bytes(raw)
    sync.run(website, "bootstrap")
    revision = "b" * 40
    monkeypatch.setattr(
        sync.subprocess,
        "check_output",
        lambda args, **kwargs: revision if args[1] == "rev-parse" else raw,
    )
    sync.run(website, "publish", revision)
    sync.run(website, "check", revision)
    path = website / "data" / sync.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["commit"] = "c" * 40
    altered = json.dumps(manifest)
    path.write_text(altered)
    with pytest.raises(ValueError, match="manifest"):
        sync.run(website, "check", revision)
    assert path.read_text() == altered
