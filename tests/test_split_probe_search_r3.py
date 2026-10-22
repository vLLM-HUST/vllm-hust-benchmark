import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_search_completion_does_not_claim_recovery():
    receipt = json.loads(
        (
            ROOT / "archive/mod-git-20261022/split-probe-search-r3/search-receipt.json"
        ).read_bytes()
    )
    for key in ("previous_search", "reference_audit"):
        data = (ROOT / receipt[key]["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == receipt[key]["sha256"]
    audit = json.loads((ROOT / receipt["reference_audit"]["path"]).read_bytes())
    assert (
        sum(r["canonical_raw_url"] is None for r in audit["references"])
        == receipt["remaining_occurrences"]
        == 33
    )
    assert receipt["newly_recovered_occurrences"] == 0
    assert receipt["promisor_search"]["missing_before"] == 2872
    assert receipt["promisor_search"]["missing_after_fetch"] == 0
    assert receipt["promisor_search"]["recoverable_source_maps"] == 0
    nested = receipt["nested_archive_search"]
    for key, aggregate in (
        ("containers", "nested_containers"),
        ("members", "nested_regular_members"),
        ("bytes", "nested_bytes"),
    ):
        assert sum(r[key] for r in nested["per_input"]) == nested[aggregate]
    assert nested["matches"] == nested["limit_skips"] == []
    assert len(receipt["access_blockers"]) == 3
