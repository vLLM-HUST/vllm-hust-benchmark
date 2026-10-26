"""Tiny synthetic packaging fixtures; never read campaign payloads or devices."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "scheduling_packager", Path(__file__).with_name("package_evidence_v1.py")
)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def fixture(root, terminal=True):
    run = root / "runs/example"
    (run / "results").mkdir(parents=True)
    (run / "producer").mkdir()
    (run / "compiler-cache").mkdir()
    (run / "compiler-cache/never-package.bin").write_bytes(b"excluded")
    (run / "results/run.json").write_text('{"collection_complete": false}\n')
    (run / "producer/example.py").write_text("# synthetic fixture\n")
    (run / "controller.json").write_text(
        json.dumps(
            {
                "finished_at": "2026-10-26T00:00:00Z" if terminal else None,
                "passed": False,
                "exit_code": 1 if terminal else None,
                "owned_members_after_cleanup": [],
                "errors": ["synthetic failure"],
            }
        )
    )
    (root / "bench").mkdir()
    (root / "bench/README.md").write_text("synthetic fixture\n")


def test_terminal_failure_retained_cache_excluded_and_repeatable(tmp_path):
    root = tmp_path / "campaign"
    fixture(root)
    first = packager.build(root, tmp_path / "one")
    second = packager.build(root, tmp_path / "two")
    assert first["archive"]["sha256"] == second["archive"]["sha256"]
    assert first["content_manifest"]["runs"][0]["controller_passed"] is False
    assert all(
        "compiler-cache" not in row["name"]
        for row in first["content_manifest"]["files"]
    )
    assert packager.verify(
        tmp_path / "one" / packager.ARCHIVE_NAME, tmp_path / "one/MANIFEST-v1.json"
    )["passed"]


def test_nonterminal_prevents_hashing_or_compression(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    fixture(root, terminal=False)
    monkeypatch.setattr(
        packager,
        "sha_file",
        lambda path: pytest.fail("Must reject before payload hashing"),
    )
    assert not packager.selection(root)["runs"][0]["terminal"]
    with pytest.raises(ValueError, match="every observed run"):
        packager.build(root, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_symlinks_and_traversal_rejected(tmp_path):
    root = tmp_path / "campaign"
    fixture(root)
    (root / "bench/link").symlink_to(root / "runs/example/controller.json")
    with pytest.raises(ValueError, match="regular files"):
        packager.selection(root)
    with pytest.raises(ValueError, match="Unsafe"):
        packager.safe_name("../outside")
