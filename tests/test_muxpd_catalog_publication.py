"""Check the MuxPD catalog join and exact published evidence custody."""

import hashlib
import json
from pathlib import Path

from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "website-publication/data"
REPORT = ROOT / "reports/muxpd-ecpa-20261025"
EVIDENCE_COMMIT = "19081ff39a25c3c8ac821ae7885688ea24a03e90"  # pragma: allowlist secret


def read(path):
    return json.loads(path.read_text())


def test_muxpd_four_tables_join_without_expanding_qualification():
    names = {
        "ecosystem.json": "ecosystem_v1.schema.json",
        "mod-taxonomy.json": "mod_taxonomy_v2.schema.json",
        "plugin-workload-navigation.json": "plugin_workload_navigation_v1.schema.json",
        "plugin-workshop-metadata.json": "plugin_workshop_metadata_v1.schema.json",
    }
    for name, schema in names.items():
        Draft7Validator(read(DATA / "schemas" / schema)).validate(read(DATA / name))
    component = next(
        row
        for row in read(DATA / "ecosystem.json")["components"]
        if row["id"] == "muxpd"
    )
    assert component["maturity"] == "experimental"
    assert component["maintainers"] == ["vLLM-HUST"]
    assert component["evidence_level"] == "hardware_verified"
    assert "6/8" in component["public_effect_en"]
    assert "No current-package performance gain" in component["public_effect_en"]
    assert (
        component["qualification_report"]
        == component["public_effect_url"]
        == (
            "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/"
            + EVIDENCE_COMMIT
            + "/reports/muxpd-ecpa-20261025/RELEASE-QUALIFICATION.md"
        )
    )
    assert (
        read(DATA / "mod-taxonomy.json")["components"]["muxpd"]["kind"] == "runtime_mod"
    )
    assert (
        "latency_slo"
        in read(DATA / "plugin-workload-navigation.json")["plugins"]["muxpd"]
    )
    metadata = read(DATA / "plugin-workshop-metadata.json")["plugins"]["muxpd"]
    assert [row["login"] for row in metadata["maintainers"]] == ["vLLM-HUST"]
    assert "muxpd" not in {
        row["id"] for row in read(DATA / "plugin-performance.json")["entries"]
    }


def test_downloaded_release_matches_frozen_index_and_asset_digests():
    receipt = read(REPORT / "evidence-release-download-verification.json")
    assert receipt["passed"] is True
    assert receipt["target_commit"] == EVIDENCE_COMMIT
    assert receipt["sha256sum_check_exit_code"] == 0
    assert (
        hashlib.sha256((REPORT / "evidence-release-api.json").read_bytes()).hexdigest()
        == receipt["api_snapshot"]["sha256"]
    )
    assert len(receipt["files"]) == 4
    for row in receipt["files"]:
        assert (
            row["state"] == "uploaded"
            and row["downloaded_bytes_equal_frozen_local"] is True
        )
        assert row["api_digest"] == "sha256:" + row["sha256"]
        if row["name"] in ("archive-index.json", "archive-verification.json"):
            payload = (REPORT / row["name"]).read_bytes()
            assert len(payload) == row["bytes"]
            assert hashlib.sha256(payload).hexdigest() == row["sha256"]
    verification = read(REPORT / "evidence-release-member-verification.json")
    assert verification == receipt["downloaded_archive_member_verification"]
    assert verification["verified"] is True and verification["files"] == 347


def test_live_publication_facts_refer_to_exact_archived_sources():
    index = read(REPORT / "website-publication-archive-index.json")
    members = {row["member"]: row for row in index["files"]}
    assert len(members) == len(index["files"])
    facts = read(REPORT / "website-publication-facts.json")
    for source in facts["sources"]:
        archived = members[source["member"]]
        assert (source["bytes"], source["sha256"]) == (
            archived["bytes"],
            archived["sha256"],
        )
    verification = read(REPORT / "website-publication-archive-verification.json")
    assert verification["verified"] and verification["files"] == len(members)
    assert facts["readback"]["passed"] is True
    assert facts["readback"]["all_manifest_files_fetched"] is False
    assert facts["readback"]["commands_executed"] is False
    assert facts["readback"]["full_tooltip_viewport_containment_claim"] is False
    assert facts["qualification_commit"] == EVIDENCE_COMMIT
    download = read(REPORT / "publication-release-download-verification.json")
    assert download["passed"] is True
    assert download["release_target"] == EVIDENCE_COMMIT
    assert download["all_member_hashes_verified"] == len(members)
    assert (
        hashlib.sha256(
            (REPORT / "publication-release-api.json").read_bytes()
        ).hexdigest()
        == download["api_snapshot_sha256"]
    )
    api = read(REPORT / "publication-release-api.json")
    assets = {row["name"]: row for row in api["assets"]}
    assert len(download["new_assets"]) == 4
    for row in download["new_assets"]:
        assert row["passed"] and row["api_state"] == "uploaded"
        assert row["api_digest"] == "sha256:" + row["sha256"]
        assert assets[row["name"]]["digest"] == row["api_digest"]
        assert assets[row["name"]]["size"] == row["bytes"]
        assert assets[row["name"]]["id"] == row["asset_id"]
        assert assets[row["name"]]["state"] == row["api_state"]
        if row["name"].endswith(".json"):
            raw = (REPORT / row["name"]).read_bytes()
            assert len(raw) == row["bytes"]
            assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    original = {
        row["name"]: row["sha256"]
        for row in read(REPORT / "evidence-release-download-verification.json")["files"]
    }
    assert download["original_four_assets_api_digests_unchanged"] == original
    for name, sha in original.items():
        assert assets[name]["digest"] == "sha256:" + sha
