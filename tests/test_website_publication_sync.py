import importlib.util
import json
import random
import subprocess
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


def commit_source(sync):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=sync.ROOT).strip()

    git("init", "-q")
    git("config", "core.autocrlf", "false")
    git("add", "website-publication/data")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "Seal test publication",
    )
    return git("rev-parse", "HEAD").decode("ascii")


@pytest.mark.parametrize("packed", [False, True])
def test_publish_rejects_incomplete_checkouts_before_writing(tmp_path, packed):
    sync = module(tmp_path)
    website = tmp_path / "website"
    (website / "data").mkdir(parents=True)
    (website / "data/kept.json").write_bytes(b"{}\n")
    if packed:
        (website / "data/missing.bin").write_bytes(b"x" * 1_000_001)
    else:
        (website / "data/missing.json").write_bytes(b"{}\n")
    sync.run(website, "bootstrap")
    revision = commit_source(sync)

    # Exercise a fresh deployment staging directory, where the old file is
    # absent from both sides and cannot be found by comparing website bytes.
    fresh = tmp_path / "fresh"
    if packed:
        vanished = sync.SOURCE.glob("missing.bin.publication*")
    else:
        vanished = [sync.SOURCE / "missing.json"]
    for path in vanished:
        path.unlink()
    with pytest.raises(ValueError, match=r"missing=.*missing\."):
        sync.run(fresh, "publish", revision)
    assert not (fresh / "data").exists()


def test_publish_rejects_plain_and_packed_sources_for_same_website_path(tmp_path):
    sync = module(tmp_path)
    website = tmp_path / "website"
    (website / "data").mkdir(parents=True)
    (website / "data/example.bin").write_bytes(b"x" * 1_000_001)
    sync.run(website, "bootstrap")
    (sync.SOURCE / "example.bin").write_bytes(b"{}\n")
    revision = commit_source(sync)
    with pytest.raises(
        ValueError, match=r"Duplicate website data path: data/example\.bin"
    ):
        sync.run(tmp_path / "fresh", "publish", revision)


def test_check_verifies_published_revision_without_rewriting_manifest(tmp_path):
    sync = module(tmp_path)
    website = tmp_path / "website"
    (website / "data").mkdir(parents=True)
    raw = b"{}\n"
    (website / "data/example.json").write_bytes(raw)
    sync.run(website, "bootstrap")
    revision = commit_source(sync)
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
