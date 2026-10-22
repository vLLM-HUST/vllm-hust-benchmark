import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "next_mod_custody", ROOT / "scripts/migrate_mod_git_evidence.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("repository", ["gdn-state-codec", "vllm-hust-opset"])
def test_frozen_originals(repository):
    directory = ROOT / "archive/mod-git-20261022" / repository
    MODULE.verify(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["source_visibility"] == "public"
    assert manifest["files"]
    assert len(manifest["source_commit"]) == 40


@pytest.mark.parametrize("prefix", ["../", "/", "", "results"])
def test_unsafe_selection_rejected_before_network(tmp_path, prefix):
    with pytest.raises(ValueError):
        MODULE.prepare(tmp_path, "unused", tmp_path / "out", prefixes=(prefix,))


def test_published_mod_links_resolve_to_benchmark_originals():
    registry = json.loads(
        (ROOT / "website-publication/data/ecosystem.json").read_text()
    )
    expected = {"gdn-state-codec", "vllm-hust-opset"}
    found = set()
    prefix = (
        "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/"
        "819b1975f346dca514f58880ca62518850100b2d/"
    )
    for component in registry["components"]:
        if component["id"] in expected:
            url = component["public_effect_url"]
            assert url.startswith(prefix)
            assert (ROOT / url.removeprefix(prefix)).is_file()
            found.add(component["id"])
    assert found == expected


def test_release_inventory_does_not_confuse_packages_with_cleared_evidence():
    receipt = json.loads(
        (ROOT / "archive/mod-git-20261022/release-inventory.json").read_text()
    )
    assert all(row["visibility"] == "public" for row in receipt["repositories"])
    by_repo = {row["repository"]: row for row in receipt["repositories"]}
    assert by_repo["gdn-state-codec"]["releases"] == []
    assert by_repo["vllm-hust-opset"]["releases"] == []
    mixed = receipt["split_batch_sample"]
    assert mixed["archive_members_under_bench_results"] == 75
    assert mixed["action"].startswith("retained;")


def test_cleanup_receipt_matches_original_inventory_and_green_gates():
    receipt = json.loads(
        (ROOT / "archive/mod-git-20261022/source-cleanup.json").read_text()
    )
    assert receipt["all_evidence_migrated"] is False
    assert receipt["all_repositories_complete"] is False
    assert receipt["release_assets_deleted"] == 0
    assert len(receipt["repositories"]) == 2
    downloaded = deleted = retained = 0
    for row in receipt["repositories"]:
        manifest = json.loads((ROOT / row["archive_manifest"]).read_text())
        assert row["source_commit"] == manifest["source_commit"]
        assert row["visibility"] == manifest["source_visibility"] == "public"
        assert row["all_lengths_and_sha256_match"] is True
        assert row["verified_download_files"] == len(manifest["files"])
        assert row["verified_download_bytes"] == sum(
            file["bytes"] for file in manifest["files"]
        )
        original = {
            file["path"]
            for file in manifest["files"]
            if file["path"].startswith(("evidence/", "results/"))
        }
        removed = set(row["deleted_data_paths"])
        kept = set(row["retained_code_paths"])
        assert not removed & kept
        assert removed | kept == original
        assert all(path.endswith(".patch") for path in kept)
        assert row["pr_checks"]
        assert all(check["conclusion"] == "SUCCESS" for check in row["pr_checks"])
        assert row["trusted_default_branch_gate"]["conclusion"] == "success"
        assert row["protection"]["strict"] and row["protection"]["enforce_admins"]
        assert {
            "context": "Benchmark-only data (organization)",
            "app_id": 15368,
        } in row["protection"]["checks"]
        downloaded += row["verified_download_files"]
        deleted += len(removed)
        retained += len(kept)
    assert (downloaded, deleted, retained) == (41, 21, 1)
