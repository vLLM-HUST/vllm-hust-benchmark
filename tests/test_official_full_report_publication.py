"""Publish sealed full-suite observations without inventing pairs or qualification."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_full_report_matches_sealed_original_population_and_metrics():
    catalog = json.loads(
        (ROOT / "website-publication/data/engine-release-history-v1.json").read_bytes()
    )
    baseline = next(
        row for row in catalog["comparison_baselines"] if row["id"] == "official-0.23.0"
    )
    descriptor = baseline["observations"]
    raw = (ROOT / "website-publication" / descriptor["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
    assert len(raw) == descriptor["bytes"]
    observation = json.loads(raw)
    report_raw = (
        ROOT / "reports/official-b0-023-full-report-20261122-v1/original-report.json"
    ).read_bytes()
    assert hashlib.sha256(report_raw).hexdigest() == (
        "7afe8a3def16a567901e12f34c3382efc4973655aabb0ece071ba45448715302"
    )
    report = json.loads(report_raw)
    old = json.loads(
        (
            ROOT
            / "website-publication/data/official-b0-023-c8-20261121-v1/observations.json"
        ).read_bytes()
    )
    assert observation["profiles"][:2] == old["profiles"]
    rows = {row["id"]: row for row in observation["profiles"]}
    assert len(rows) == 11
    for name, denominator in report["planned_denominators"].items():
        row = rows[name]
        metric = report["metrics"][name]
        assert row["arm"]["requests"] == denominator
        assert row["arm"]["failed"] == metric["failed"]
        for key in (
            "output_tokens_per_second",
            "duration_seconds",
            "mean_ttft_ms",
            "mean_tpot_ms",
        ):
            assert row["arm"][key] == metric.get(key)
    assert rows["mmlu"]["arm"]["mean_ttft_ms"] is None
    for name in ("hle", "frontier"):
        assert rows[name]["quality"] == report["quality"][name + "-judge"]
        assert "correct" not in rows[name]["arm"]
    for name in ("swe", "terminal"):
        assert rows[name]["quality"]["quality_measured"] is False
        assert "correct" not in rows[name]["arm"]
    context = report["phases"]["context"]["performance-summary.json"]["value"]
    for phase in context["phases"]:
        if phase["phase"].endswith("-formal"):
            row = rows["real-context-" + phase["phase"]]
            assert row["tenants"] == phase["tenants"]
            assert row["per_tenant"] == phase["per_tenant"]
            for key, value in phase["metrics"].items():
                assert row["arm"][key] == value
    native = report["phases"]["native-timing"]["receipt.json"]["value"]
    assert observation["native_timing"]["repetitions"] == native["T_repetitions"]
    assert len(native["T_repetitions"]) == 3
    assert baseline["pairs"] is observation["pairs"] is None
    assert observation["F"] is observation["efficiency_percent"] is None
    assert observation["necessary_work_certified"] is False
    assert observation["independent_third_party"] is False
    assert observation["automatic_ft_audit"]["status"] == "ineligible"
    assert (
        observation["automatic_ft_audit"][
            "actual_kernel_memory_access_or_slot_values_observed"
        ]
        is False
    )
