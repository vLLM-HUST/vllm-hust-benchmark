"""Publish sealed full-suite observations without inventing pairs or qualification."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_full_report_matches_sealed_original_population_and_metrics():
    catalog = json.loads(
        (ROOT / "website-publication/data/engine-release-history-v1.json").read_bytes()
    )
    baseline = next(
        row for row in catalog["comparison_baselines"] if row["id"] == "official-0.23.0"
    )
    raw = (
        ROOT
        / "website-publication/data/official-b0-023-full-report-20261122-v1/observations.json"
    ).read_bytes()
    assert len(raw) == 77029
    assert (
        hashlib.sha256(raw).hexdigest()
        == (
            "ec941421093cc7340de3f84fe88c960ac68c634f943e6def990d1d943c3b7665"  # pragma: allowlist secret -- immutable v1 observations SHA256
        )
    )
    observation = json.loads(raw)
    report_raw = (
        ROOT / "reports/official-b0-023-full-report-20261122-v1/original-report.json"
    ).read_bytes()
    assert (
        hashlib.sha256(report_raw).hexdigest()
        == (
            "7afe8a3def16a567901e12f34c3382efc4973655aabb0ece071ba45448715302"  # pragma: allowlist secret -- public original report SHA256
        )
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


REPLAY = "official-b0-original-sse-replay-20261125-v1"
PUBLICATION = ROOT / "website-publication"
C8_PHASES = ("4096-formal", "16384-formal")


def read_described_publication(descriptor):
    raw = (PUBLICATION / descriptor["path"]).read_bytes()
    assert len(raw) == descriptor["bytes"]
    assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
    return json.loads(raw)


@pytest.mark.parametrize(
    "baseline_id,profile_name,ttft_ms,tpot_ms,c8_populations",
    [
        ("official-0.23.0", "023", 280172.754024351, 23.079636707554588, ()),
        (
            "candidate-0.28.0rc1-source",
            "028-source",
            291775.83262527955,
            22.90179158145342,
            ((304, 77824), (172, 44032)),
        ),
        (
            "official-0.30.0",
            "030",
            281936.01960920496,
            22.707744850761333,
            ((293, 75008), (157, 40192)),
        ),
    ],
)
def test_sealed_request_replay_extends_independent_observations(
    baseline_id, profile_name, ttft_ms, tpot_ms, c8_populations
):
    catalog = json.loads(
        (PUBLICATION / "data/engine-release-history-v1.json").read_bytes()
    )
    baseline = next(
        row for row in catalog["comparison_baselines"] if row["id"] == baseline_id
    )
    descriptor = baseline["observations"]
    assert (
        descriptor["path"]
        == f"data/{REPLAY}/official-b0-{profile_name}-observations.json"
    )
    observation = read_described_publication(descriptor)
    assert observation["baseline_id"] == baseline_id
    assert baseline["pairs"] is observation["pairs"] is None
    assert observation["F"] is observation["efficiency_percent"] is None
    assert observation["necessary_work_certified"] is False
    replay_descriptor = observation["original_sse_replay"]
    assert replay_descriptor["path"] == f"data/{REPLAY}/replay-receipt.json"
    assert replay_descriptor["report_url"].endswith(f"/reports/{REPLAY}/README.md")
    assert (ROOT / f"reports/{REPLAY}/README.md").is_file()
    receipt = read_described_publication(replay_descriptor)
    assert receipt["schema"] == "vllm-hust-official-b0-original-sse-replay/v1"
    assert receipt["inference_reruns"] == 0
    reviewed = receipt["baselines"][baseline_id]
    assert reviewed["original_archive"] == baseline["evidence"]
    assert reviewed["original_observations"]["path"] != descriptor["path"]
    original = read_described_publication(reviewed["original_observations"])
    preserved = set(original) - {
        "profiles",
        "publication_scope",
        "continuous_original_phase_summary",
    }
    assert {key: observation[key] for key in preserved} == {
        key: original[key] for key in preserved
    }
    old_rows = {row["id"]: row for row in original["profiles"]}
    rows = {row["id"]: row for row in observation["profiles"]}
    assert len(rows) == len(observation["profiles"])
    for key, old_row in old_rows.items():
        if key != "mmlu":
            assert rows[key] == old_row
    old_mmlu = old_rows["mmlu"]
    stage = reviewed["stages"]["mmlu"]
    mmlu = stage["rows"]["mmlu"]
    assert stage["source_summary"] == old_mmlu["source_metric"]
    assert stage["original_manifest"] == old_mmlu["arm"]["source_seal_sha256"]
    assert mmlu["counts"]["successful"] == mmlu["counts"]["first_content_delta"] == 200
    assert mmlu["verified_request_artifacts"] == 400
    assert mmlu["ttft_ms"]["mean"] == ttft_ms
    assert mmlu["tpot_ms"]["mean"] == tpot_ms
    assert rows["mmlu"] == {
        **old_mmlu,
        "arm": {
            **old_mmlu["arm"],
            "mean_ttft_ms": ttft_ms,
            "mean_tpot_ms": tpot_ms,
            "physical_cards": stage["physical_cards"],
        },
        "latency_source": {
            "receipt_path": replay_descriptor["path"],
            "source_seal_sha256": stage["original_manifest"],
        },
    }

    if not c8_populations:
        assert rows.keys() == old_rows.keys()
        assert set(reviewed["stages"]) == {"mmlu"}
        return

    assert rows.keys() - old_rows.keys() == set(C8_PHASES)
    original_phases = original["continuous_original_phase_summary"]
    published_phases = observation["continuous_original_phase_summary"]
    assert published_phases["phases"] == original_phases["phases"]
    assert published_phases["raw_sse_recomputed"] is True
    assert published_phases["recomputed_profiles"] == list(C8_PHASES)
    stage = reviewed["stages"]["continuous"]
    assert set(stage["rows"]) == set(C8_PHASES)
    assert original_phases["source"] in (
        stage["source_summary"],
        stage["source_summary"]["sha256"],
    )
    for phase, (requests, tokens) in zip(C8_PHASES, c8_populations, strict=True):
        row, measured = rows[phase], stage["rows"][phase]
        assert row["workload_kind"] == "continuous-synthetic"
        assert row["raw_phase_id"] == phase
        assert row["source_metric"] == stage["source_summary"]
        arm = row["arm"]
        assert arm["role"] == "B0" and arm["failed"] == 0
        assert (
            arm["requests"]
            == arm["successful"]
            == measured["counts"]["successful"]
            == requests
        )
        assert measured["counts"]["first_reasoning_delta"] == requests
        assert measured["verified_request_artifacts"] == 2 * requests
        assert (
            arm["output_tokens"]
            == measured["counts"]["successful_output_tokens"]
            == tokens
        )
        assert arm["original_submission_seconds"] == 300
        assert arm["duration_seconds"] == measured["seconds_with_drain"]
        assert arm["output_tokens_per_second"] == pytest.approx(
            tokens / arm["duration_seconds"]
        )
        assert (
            arm["output_tokens_per_second"]
            == measured["successful_output_tokens_per_second"]
        )
        assert arm["mean_ttft_ms"] == measured["ttft_ms"]["mean"]
        assert arm["mean_tpot_ms"] == measured["tpot_ms"]["mean"]
        assert arm["source_seal_sha256"] == stage["original_manifest"]
        assert arm["physical_cards"] == stage["physical_cards"]
