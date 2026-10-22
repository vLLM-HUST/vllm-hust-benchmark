import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_negative_search_preserves_unresolved_identity():
    receipt = json.loads(
        (
            ROOT / "archive/mod-git-20261022/split-probe-search-r2/search-receipt.json"
        ).read_bytes()
    )
    prior = (ROOT / receipt["previous_audit"]["path"]).read_bytes()
    assert hashlib.sha256(prior).hexdigest() == receipt["previous_audit"]["sha256"]
    missing = [
        r for r in json.loads(prior)["references"] if r["canonical_raw_url"] is None
    ]
    assert (
        len(missing)
        == receipt["target_occurrences"]
        == receipt["remaining_occurrences"]
        == 33
    )
    assert (
        len({r["original_reference"] for r in missing})
        == receipt["target_unique_references"]
    )
    assert len({r["declared_sha256"] for r in missing if r["declared_sha256"]}) == 8
    assert receipt["newly_recovered_occurrences"] == 0
    release = receipt["release_search"]
    assert len(release["assets"]) == 8
    assert (
        sum(a["hashed_members"] for a in release["assets"])
        == release["hashed_members"]
        == 8020
    )
    for section in ("git_search", "workspace_search", "release_search"):
        assert receipt[section]["sha256_matches"] == []
        assert receipt[section]["limitations"]
    followup = receipt["followup_checks"]
    assert (
        followup["additional_large_logs"]
        == receipt["workspace_search"]["oversize_logs_skipped"]
    )
    assert followup["sha256_matches"] == []
    assert followup["archive_original_path_mentions"] == []
    assert followup["recoverable_git_maps"] == 0
