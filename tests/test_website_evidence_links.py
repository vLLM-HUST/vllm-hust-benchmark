import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import build_website_evidence_links as builder
    from migrate_mod_git_evidence import verify
finally:
    sys.path.pop(0)


def test_generated_evidence_bindings_are_current_and_targets_exist():
    stored = json.loads((ROOT / builder.OUTPUT).read_text())
    assert stored == builder.build()
    assert len(stored["frontier_points"]) > 300
    targets = list(stored["rewrites"].values()) + list(stored["serving_plans"].values())
    for point in stored["frontier_points"].values():
        assert point["links"]
        targets.extend(point["links"])
    for target in targets:
        path = ROOT / target["path"]
        assert path.is_dir() if target["kind"] == "tree" else path.is_file()
        assert target["role"] in {
            "raw-archive",
            "raw-parts",
            "raw-manifest",
            "public-measurement-extract",
            "publication-record",
            "original-public-record",
            "verified-release-member-copy",
            "archived-measurement-report",
        }


def test_original_public_serving_records_are_byte_preserved():
    for capsule in (
        "website-links-20261022",
        "website-links-accounting-20261022",
        "vspec-browser-raw-20261022",
    ):
        verify(ROOT / "archive" / capsule)
    links = json.loads((ROOT / builder.OUTPUT).read_text())
    plans = json.loads(
        (ROOT / "website-publication/data/serving-plans.json").read_text()
    )
    for plan in plans["plans"]:
        original = json.loads(
            (ROOT / links["serving_plans"][plan["id"]]["path"]).read_text()
        )
        assert original["sourceRunId"] == plan["runId"]
        for field in (
            "sourceRunId",
            "measurementSeconds",
            "chips",
            "counts",
            "tokensPerSecondPerChip",
        ):
            assert original[field] == plan["accounting"][field]
        assert original.get("sourceRawSha256", original.get("sourceSha256")) == plan[
            "accounting"
        ].get("sourceRawSha256", plan["accounting"].get("sourceSha256"))


def test_small_release_access_copies_match_existing_custody_hashes():
    receipt = json.loads((ROOT / builder.DIRECT / "manifest.json").read_text())
    original = json.loads((ROOT / receipt["source_manifest"]).read_text())
    assert receipt["asset"] == original["asset"]
    for record in receipt["files"]:
        raw = (ROOT / builder.DIRECT / record["path"]).read_bytes()
        source = next(
            row for row in original["files"] if row["path"] == record["member"]
        )
        assert hashlib.sha256(raw).hexdigest() == record["sha256"] == source["sha256"]
        assert len(raw) == record["bytes"] == source["bytes"]
