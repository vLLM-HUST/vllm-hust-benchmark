"""Check the imported publication against recorded measurements, not desired gains."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/controller-r8-20261118"


def test_controller_publication_matches_sealed_pairs():
    product = json.loads(
        (
            ROOT / "website-publication/data/controller-product-results-v1.json"
        ).read_text()
    )
    assert product["schema"] == "controller-product-results/v1"
    current = next(
        entry
        for entry in product["releases"]
        if entry["id"] == product["default_release"]
    )
    paired = json.loads((REPORT / "multi-turn-paired-result.json").read_text())
    assert current["result"] == paired
    a, b = paired["direct"], paired["request_throttling_controller"]
    assert a["failed_requests"] == b["failed_requests"] == 0
    assert a["input_tokens"] == b["input_tokens"]
    assert a["output_tokens"] == b["output_tokens"]
    assert math.isclose(
        paired["time_reduction_percent"],
        100 * (1 - b["elapsed_seconds"] / a["elapsed_seconds"]),
    )
    fixed = json.loads((REPORT / "fixed-profile-pairs.json").read_text())
    assert fixed["all_four_pairs_complete"]
    assert [p["id"] for p in current["profiles"]] == [
        "mmlu",
        "hle",
        "frontier",
        "swe",
        "terminal",
    ]
    for profile in current["profiles"]:
        if profile["id"] == "hle":
            assert profile["status"] == "deferred"
            assert "direct" not in profile
            continue
        original = fixed["profiles"][profile["id"]]
        assert profile["direct"] == original["arms"]["direct"]
        assert profile["controller"] == original["arms"]["controller"]
        assert profile["time_reduction_percent"] == original["time_reduction_percent"]
        assert original["same_client_request_bodies"]
        assert original["same_per_case_engine_prompt_tokens"]


def test_public_request_evidence_retains_denominators_without_task_text():
    by_arm = {}
    for arm in ("direct", "controller"):
        rows = json.loads((REPORT / (arm + "-requests.json")).read_text())
        assert len(rows) == 472
        assert all(
            "request" not in row and "response" not in row and "content" not in row
            for row in rows
        )
        assert all(row["status"] == "ok" and row["http_status"] == 200 for row in rows)
        by_arm[arm] = {(row["dataset"], row["case_id"]): row for row in rows}
        assert len(by_arm[arm]) == 472
    assert by_arm["direct"].keys() == by_arm["controller"].keys()
    for key, direct in by_arm["direct"].items():
        controller = by_arm["controller"][key]
        assert direct["request_sha256"] == controller["request_sha256"]
        assert direct["usage"]["prompt_tokens"] == controller["usage"]["prompt_tokens"]
