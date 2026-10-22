import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location(
        "data_entrypoints", SCRIPTS / "data_entrypoints.py"
    )
    index = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(index)
finally:
    sys.path.pop(0)


def source_snapshot():
    return {
        "captured_at": "2026-10-22T00:00:00+00:00",
        "website_commit": "a" * 40,
        "website_publication_manifest": {"commit": "b" * 40, "files": []},
        "website_publication_manifest_url": index.MANIFEST_URL,
        "production_manifest_error": None,
        "website_references": [
            {"target": "./data/x.json?v=1", "consumers": ["assets/x.js"]},
            {"target": "./data/missing.json", "consumers": ["assets/x.js"]},
            {"target": "./data/${file}", "consumers": ["assets/x.js"]},
        ],
        "releases": [
            {
                "tag": "evidence",
                "assets": [
                    {"name": "raw.tar.gz", "url": "https://example.org/raw.tar.gz"}
                ],
            }
        ],
    }


def test_reference_extraction():
    text = """fetch('./data/x.json?v=1'); fetch(`./data/${file}`);
    {"path": "reports/m01/raw.json", "url": "https://github.com/vLLM-HUST/test"}"""
    assert index.references(text) == [
        "./data/${file}",
        "./data/x.json?v=1",
        "https://github.com/vLLM-HUST/test",
        "reports/m01/raw.json",
    ]
    escaped = json.dumps(
        {"path": "reports/原始记录.json", "url": "https://example.org/raw"}
    ).replace("/", "\\/")
    assert index.references(escaped) == [
        "https://example.org/raw",
        "reports/原始记录.json",
    ]


def test_binding_content_changes_and_reference_status(tmp_path):
    path = "website-publication/data/x.json"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text('{"path":"reports/m01/raw.json"}')
    paths = [path, "reports/m01/raw.json"]
    sources = source_snapshot()
    first = index.build(tmp_path, paths, sources)
    assert first == index.build(tmp_path, list(reversed(paths)), sources)
    refs = {r["target"]: r for r in first["references"]}
    assert refs["./data/x.json?v=1"]["benchmark_path"] == path
    assert refs["./data/missing.json"]["resolution"] == "unmapped-frontend-reference"
    assert refs["./data/${file}"]["resolution"] == "directory-or-dynamic"
    assert refs["reports/m01/raw.json"]["resolution"] == "tracked-evidence-path"
    assert first["counts"]["assets"] == 1
    file.write_text("{}")
    assert first != index.build(tmp_path, paths, sources)


def test_packed_publication_hashes_and_safe_paths(tmp_path):
    part = "website-publication/data/x.json.publication.part-000"
    descriptor = "website-publication/data/x.json.publication.json"
    (tmp_path / part).parent.mkdir(parents=True)
    raw = b'{"url":"https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence"}'
    chunk = gzip.compress(raw, mtime=0)
    (tmp_path / part).write_bytes(chunk)
    meta = {
        "encoding": "gzip",
        "parts": [
            {"name": Path(part).name, "sha256": index.sha(chunk), "bytes": len(chunk)}
        ],
    }
    (tmp_path / descriptor).write_text(json.dumps(meta))
    result = index.build(tmp_path, [part, descriptor], source_snapshot())
    binding = result["publication_bindings"][0]
    assert binding["website_path"] == "data/x.json"
    assert binding["sha256"] == index.sha(raw)
    assert len(result["references"]) == 4
    (tmp_path / part).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        index.build(tmp_path, [part, descriptor], source_snapshot())
    meta["parts"][0]["name"] = "../escape"
    (tmp_path / descriptor).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Unsafe"):
        index.build(tmp_path, [part, descriptor], source_snapshot())


def test_committed_snapshot_frontend_coverage():
    sources = json.loads((index.ROOT / index.SOURCES).read_text())
    catalog = index.build(index.ROOT, index.tracked_paths(index.ROOT), sources)
    assert not [
        r
        for r in catalog["references"]
        if r["resolution"] == "unmapped-frontend-reference"
    ]
    assert catalog["counts"]["assets"] == sum(
        len(r["assets"]) for r in sources["releases"]
    )
    assert all(
        a["url"].startswith(f"https://github.com/{index.REPO}/releases/download/")
        for r in catalog["releases"]
        for a in r["assets"]
    )


def test_offline_check_detects_stale_output(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / index.SOURCES).write_text(json.dumps(source_snapshot()))
    monkeypatch.setattr(index, "ROOT", tmp_path)
    monkeypatch.setattr(index, "tracked_paths", lambda root: [])
    assert index.main(["--check"]) == 1
    assert index.main(["--write"]) == 0
    assert index.main(["--check"]) == 0
    (tmp_path / index.OUTPUT).write_text("{}")
    assert index.main(["--check"]) == 1


@pytest.mark.parametrize("production_available", [True, False])
def test_refresh_pagination_drafts_and_manifest_identity(
    monkeypatch, production_available
):
    import io
    from urllib.error import URLError

    manifest = {
        "schema": "benchmark-website-publication/v1",
        "repository": index.REPO,
        "commit": "c" * 40,
        "files": [],
    }

    def fake_git(repo, *args):
        if args[0] == "rev-parse":
            return b"a" * 40
        if args[0] == "ls-tree":
            return b"index.html\0assets/view.js\0tests/test.js\0"
        if args[-1].endswith("benchmark-source-manifest.json"):
            return json.dumps({**manifest, "commit": "b" * 40}).encode()
        return b"fetch('./data/x.json')"

    def fake_open(*args, **kwargs):
        if not production_available:
            raise URLError("unavailable")
        return io.BytesIO(json.dumps(manifest).encode())

    def fake_api(args):
        assert "--paginate" in args and "--slurp" in args
        release = {
            "draft": False,
            "tag_name": "r1",
            "html_url": "https://example.org/r1",
            "id": 1,
            "published_at": None,
            "assets": [],
        }
        return json.dumps(
            [[release, {"draft": True}], [{**release, "tag_name": "r2", "id": 2}]]
        ).encode()

    monkeypatch.setattr(index, "git", fake_git)
    monkeypatch.setattr(index, "urlopen", fake_open)
    monkeypatch.setattr(index.subprocess, "check_output", fake_api)
    snapshot = index.refresh(Path("unused"), "main")
    assert len(snapshot["releases"]) == 2
    assert list(snapshot["website_inputs_sha256"]) == ["assets/view.js", "index.html"]
    assert (
        snapshot["website_publication_manifest"]["commit"]
        == ("c" if production_available else "b") * 40
    )
    assert bool(snapshot["production_manifest_error"]) is not production_available
