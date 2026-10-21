import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "mod_git_custody", ROOT / "scripts/migrate_mod_git_evidence.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "directory", sorted((ROOT / "archive/mod-git-20261021").glob("stateaxis-*"))
)
def test_stateaxis_original_bytes(directory):
    MODULE.verify(directory)
    receipt = json.loads((directory / "manifest.json").read_text())
    assert receipt["source_visibility"] == "public"
    assert len(receipt["source_commit"]) == 40
    assert any(r["path"].startswith("evidence/") for r in receipt["files"])
    if directory.name == "stateaxis-pipelined-weight-loading":
        assert len(receipt["invalid_json_paths"]) == 14
        assert all(
            (directory / "snapshot" / p).read_bytes() == b""
            for p in receipt["invalid_json_paths"]
        )


def test_corruption_and_missing_members_are_rejected(tmp_path):
    source = next((ROOT / "archive/mod-git-20261021").glob("stateaxis-*/manifest.json"))
    receipt = json.loads(source.read_text())
    receipt["files"] = receipt["files"][:1]
    (tmp_path / "manifest.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Missing or unindexed"):
        MODULE.verify(tmp_path)
    member = tmp_path / "snapshot" / receipt["files"][0]["path"]
    member.parent.mkdir(parents=True)
    member.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        MODULE.verify(tmp_path)


def test_source_cleanup_has_complete_archive_and_green_merge_receipts():
    archive = ROOT / "archive/mod-git-20261021"
    receipt = json.loads((archive / "source-cleanup.json").read_text())
    assert receipt["all_evidence_migrated"] is False
    assert receipt["all_repositories_complete"] is False
    assert len(receipt["repositories"]) == 4
    deleted_count = retained_count = downloaded_count = 0
    for row in receipt["repositories"]:
        manifest = json.loads((ROOT / row["archive_manifest"]).read_text())
        assert manifest["source_commit"] == row["source_commit"]
        assert row["visibility"] == manifest["source_visibility"] == "public"
        records = manifest["files"]
        original_paths = {
            record["path"]
            for record in records
            if record["path"].startswith("evidence/")
        }
        removed = set(row["deleted_data_paths"])
        retained = set(row["retained_code_paths"])
        assert not removed & retained
        assert removed | retained == original_paths
        assert all(path.endswith(".py") for path in retained)
        assert not any(path.endswith(".py") for path in removed)
        verified = row["download_back"]
        assert verified["all_lengths_and_sha256_match"] is True
        assert verified["verified_files"] == len(records)
        assert verified["verified_bytes"] == sum(record["bytes"] for record in records)
        assert row["pr_checks"]
        assert all(check["conclusion"] == "SUCCESS" for check in row["pr_checks"])
        assert row["trusted_default_branch_gate"]["conclusion"] == "success"
        assert row["protection"]["strict"] and row["protection"]["enforce_admins"]
        assert {
            "context": "Benchmark-only data (organization)",
            "app_id": 15368,
        } in row["protection"]["checks"]
        deleted_count += len(removed)
        retained_count += len(retained)
        downloaded_count += verified["verified_files"]
    assert (deleted_count, retained_count, downloaded_count) == (240, 3, 264)


def test_published_stateaxis_links_resolve_to_frozen_benchmark_originals():
    registry = json.loads(
        (ROOT / "website-publication/data/ecosystem.json").read_text()
    )
    expected = {
        "stateaxis-automatic-prefix-cache",
        "stateaxis-async-kv-transfer",
        "stateaxis-pipelined-weight-loading",
    }
    found = set()
    prefix = (
        "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/"
        "ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/"
    )
    for component in registry["components"]:
        if component["id"] not in expected:
            continue
        found.add(component["id"])
        url = component["public_effect_url"]
        assert url.startswith(prefix)
        assert component["latest_result"] in url
        relative = url.removeprefix(prefix)
        assert (ROOT / relative).is_file()
        manifest = json.loads(
            (
                ROOT / "archive/mod-git-20261021" / component["id"] / "manifest.json"
            ).read_text()
        )
        assert component["latest_result"] in {
            record["path"] for record in manifest["files"]
        }
    assert found == expected
