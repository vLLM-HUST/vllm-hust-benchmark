"""Preserve the source candidate's measurements and the scope of its CPU refusal."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/official-b0-028-source-full-report-20261123-v1"


def test_source_candidate_keeps_originals_and_preflight_scope():
    manifest = json.loads((REPORT / "archive-manifest.json").read_bytes())
    for name, pin in manifest["files"].items():
        path = REPORT / name
        if path.is_file() and name.endswith(".json"):
            payload = path.read_bytes()
            assert len(payload) == pin["bytes"]
            assert hashlib.sha256(payload).hexdigest() == pin["sha256"]
    catalog = json.loads(
        (ROOT / "website-publication/data/engine-release-history-v1.json").read_bytes()
    )
    baseline = next(
        row
        for row in catalog["comparison_baselines"]
        if row["id"] == "candidate-0.28.0rc1-source"
    )
    assert baseline["status"] == "partial"
    assert baseline["source_identity"]["kind"] == "source-build-candidate"
    descriptor = baseline["observations"]
    raw = (ROOT / "website-publication" / descriptor["path"]).read_bytes()
    assert len(raw) == descriptor["bytes"]
    assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
    observation = json.loads(raw)
    original_raw = (REPORT / "original-report.json").read_bytes()
    original = json.loads(original_raw)
    audit_raw = (REPORT / "automatic-ft-rejection.json").read_bytes()
    audit = json.loads(audit_raw)
    assert observation["baseline_id"] == baseline["id"]
    provenance = observation["source_provenance"]
    assert (
        provenance["original_report_sha256"] == hashlib.sha256(original_raw).hexdigest()
    )
    assert observation["original_campaign_status"] == original["status"] == "failed"
    rows = {row["id"]: row for row in observation["profiles"]}
    assert len(rows) == 9
    assert original["planned_denominators"] == {
        "mmlu": 200,
        "hle": 200,
        "frontier": 160,
        "swe": 64,
        "terminal": 48,
    }
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
        for key in ("tenants", "windows", "per_tenant"):
            assert row[key] == phase[key]
        assert all(row["arm"][key] == value for key, value in phase["metrics"].items())
    native = original["phases"]["native-timing"]["receipt.json"]["value"]
    assert observation["native_timing"]["repetitions"] == native["T_repetitions"]
    assert len(native["T_repetitions"]) == 3
    assert observation["native_timing"]["original_timing_repeated"] is False
    assert observation["native_timing"]["F_interval"] is None
    automatic = observation["automatic_ft_audit"]
    assert automatic["report_sha256"] == hashlib.sha256(audit_raw).hexdigest()
    for key in ("status", "stage", "first_counterexample"):
        assert automatic[key] == audit[key]
    assert automatic["status"] == "ineligible"
    assert automatic["stage"] == "sealed-json-preflight"
    assert automatic["full_sdk_and_model_admission_performed"] is False
    assert automatic["companion_requests"] == 200
    assert automatic["full_admission_attempt_status"] == "failed"
    assert automatic["actual_kernel_memory_access_or_slot_values_observed"] is False
    counterexample = automatic["first_counterexample"]
    assert counterexample["call"] == 5 and counterexample["row"] == 0
    assert counterexample["expected_selected_prefix_kv_extent"] == 1383
    assert counterexample["observed_step1_kv_extent"] == 1385
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
    for key in ("output_tokens_per_second", "mean_ttft_ms", "mean_tpot_ms"):
        assert continuous[key] is None
