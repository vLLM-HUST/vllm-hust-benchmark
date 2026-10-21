import importlib.util
import json
import random
from pathlib import Path

import pytest


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
