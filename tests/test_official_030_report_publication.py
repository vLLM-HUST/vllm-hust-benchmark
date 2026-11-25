"""The new B0 must keep sealed populations and a failed arithmetic qualification."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/official-b0-030-full-report-20261122-v1"


def test_official030_observations_preserve_originals_and_automatic_refusal():
    catalog = json.loads(
        (ROOT / "website-publication/data/engine-release-history-v1.json").read_bytes()
    )
    baseline = next(
        row for row in catalog["comparison_baselines"] if row["id"] == "official-0.30.0"
    )
    raw = (
        ROOT
        / "website-publication/data/official-b0-030-full-report-20261122-v1/observations.json"
    ).read_bytes()
    assert len(raw) == 78937
    assert (
        hashlib.sha256(raw).hexdigest()
        == (
            "93c0d1916825005c694415b29295a1aa028ef5488ae6311884c1af52dc3114f1"  # pragma: allowlist secret -- immutable v1 observations SHA256
        )
    )
    observation = json.loads(raw)
    original_raw = (REPORT / "original-report.json").read_bytes()
    original = json.loads(original_raw)
    audit_raw = (REPORT / "automatic-ft-rejection.json").read_bytes()
    audit = json.loads(audit_raw)
    provenance = observation["source_provenance"]
    assert (
        hashlib.sha256(original_raw).hexdigest() == provenance["original_report_sha256"]
    )
    assert observation["original_campaign_status"] == original["status"] == "failed"
    assert (
        provenance["device_execution_commit"] != provenance["report_projection_commit"]
    )
    assert provenance["independent_ft_companion_commit"] not in (
        provenance["device_execution_commit"],
        provenance["report_projection_commit"],
    )
    rows = {row["id"]: row for row in observation["profiles"]}
    assert len(rows) == 9
    for name, planned in original["planned_denominators"].items():
        row = rows[name]
        assert row["arm"]["requests"] == planned
        metric = original["metrics"][name]
        for key in ("failed", "output_tokens_per_second", "duration_seconds"):
            assert row["arm"][key] == metric[key]
        for key in ("mean_ttft_ms", "mean_tpot_ms"):
            assert row["arm"][key] == metric.get(key)
    for name in ("hle", "frontier"):
        assert rows[name]["quality"] == original["quality"][name + "-judge"]
        assert "correct" not in rows[name]["arm"]
    for name in ("swe", "terminal"):
        assert rows[name]["quality"]["quality_measured"] is False
        assert "correct" not in rows[name]["arm"]
    context = original["phases"]["context"]["performance-summary.json"]["value"]
    for phase in context["phases"]:
        if not phase["phase"].endswith("-formal"):
            continue
        row = rows["real-context-" + phase["phase"]]
        assert row["tenants"] == phase["tenants"]
        assert row["windows"] == phase["windows"]
        assert row["per_tenant"] == phase["per_tenant"]
        assert all(row["arm"][key] == value for key, value in phase["metrics"].items())
    native = original["phases"]["native-timing"]["receipt.json"]["value"]
    assert observation["native_timing"]["repetitions"] == native["T_repetitions"]
    assert len(native["T_repetitions"]) == 3
    automatic = observation["automatic_ft_audit"]
    assert automatic["report_sha256"] == hashlib.sha256(audit_raw).hexdigest()
    assert automatic["status"] == audit["status"] == "ineligible"
    assert automatic["first_counterexample"] == audit["first_counterexample"]
    assert automatic["full_sdk_and_model_admission_performed"] is True
    assert automatic["actual_kernel_memory_access_or_slot_values_observed"] is False
    assert observation["F"] is observation["efficiency_percent"] is None
    assert observation["necessary_work_certified"] is False
    assert baseline["pairs"] is observation["pairs"] is None
    assert observation["independent_third_party"] is False
    continuous = observation["continuous_original_phase_summary"]
    assert (
        continuous["phases"]
        == original["phases"]["continuous"]["service/continuous.json"]["value"][
            "phases"
        ]
    )
    assert continuous["raw_sse_recomputed"] is False
    assert continuous["output_tokens_per_second"] is None
