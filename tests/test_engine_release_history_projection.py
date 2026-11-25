"""Publish the exact historical catalog while retaining unrelated withdrawal rules."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "history_projection", ROOT / "scripts/build_public_site.py"
)
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)


def prepare(tmp_path):
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    (source / "assets").mkdir()
    for name in [
        "index.html",
        "plugins.html",
        "dataset-introductions.html",
        "leaderboard-runs.html",
        "assets/site.js",
    ]:
        (source / name).write_text("")
    raw = (
        ROOT / "website-publication/data/engine-release-history-v1.json"
    ).read_bytes()
    (source / projection.PUBLIC_ENGINE_HISTORY_PATH).write_bytes(raw)
    (source / "data/unrelated.json").write_text(
        json.dumps(
            {"records": [{"id": "unrelated", "mods": ["BetterScale"], "value": 42}]}
        )
    )
    entry = dict(
        website_path=projection.PUBLIC_ENGINE_HISTORY_PATH,
        sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),
    )
    (source / "data/benchmark-source-manifest.json").write_text(
        json.dumps({"files": [entry]})
    )
    return source, raw


def test_release_history_survives_actual_public_build(tmp_path):
    source, raw = prepare(tmp_path)
    output = tmp_path / "public"
    projection.build(source, output, "test-policy")
    projection.validate(output)
    assert (output / projection.PUBLIC_ENGINE_HISTORY_PATH).read_bytes() == raw
    value = json.loads(raw)
    assert {entry["id"] for entry in value["entries"]} == {
        "r4",
        "r5",
        "r6",
        "r7",
        "r8",
        "r9",
        "r10",
    }
    assert {entry["id"] for entry in value["attempts"]} == {"r1", "r2", "r3"}
    assert json.loads((output / "data/unrelated.json").read_text()) == {"records": []}


def test_resigning_manifest_cannot_publish_changed_history(tmp_path):
    source, raw = prepare(tmp_path)
    changed = json.loads(raw)
    changed["entries"][0]["measurements"][0]["arms"][1]["output_tokens_per_second"] *= 2
    rewritten = (json.dumps(changed) + "\n").encode()
    path = source / projection.PUBLIC_ENGINE_HISTORY_PATH
    path.write_bytes(rewritten)
    manifest = source / "data/benchmark-source-manifest.json"
    value = json.loads(manifest.read_text())
    value["files"][0].update(
        sha256=hashlib.sha256(rewritten).hexdigest(), bytes=len(rewritten)
    )
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="release history"):
        projection.build(source, tmp_path / "public", "test-policy")


def test_primary_release_comparisons_use_fixed_B0_not_another_B1():
    catalog = json.loads(
        (ROOT / "website-publication/data/engine-release-history-v1.json").read_text()
    )
    baseline = catalog["baseline"]
    assert baseline["id"] == "original018-cann8.5.1"
    assert baseline["vllm"] == baseline["vllm_ascend"] == "0.18.0"
    for entry in catalog["entries"]:
        assert entry["baseline_id"] == baseline["id"]
        for measurement in entry["measurements"]:
            assert measurement["control"] == measurement["arms"][0]["role"] == "B0"
            assert measurement["baseline_id"] == baseline["id"]
    r7 = next(entry for entry in catalog["entries"] if entry["id"] == "r7")
    reference = json.loads(
        (
            ROOT
            / "website-publication/data/five-datasets-r6-20261114/summary-public-v3.json"
        ).read_text()
    )
    for row in r7["measurements"]:
        original = reference["continuous"]["B0"]["groups"][row["id"] + "-formal"]
        for metric in ["output_tokens_per_second", "mean_ttft_ms", "mean_tpot_ms"]:
            assert row["arms"][0][metric] == original[metric]
        assert (
            row["baseline_origin"]["seal_sha256"]
            == reference["continuous"]["B0"]["source_seal_sha256"]
        )
    assert all(row["control"] == "r6" for row in r7["development_measurements"])


def test_r8_fixed_five_measurements_survive_release_catalog_publication():
    """Completed sealed performance pairs must not disappear from the catalog."""
    report = json.loads(
        (ROOT / "reports/r8-fixed-five-20261118/summary-public-v2.json").read_text()
    )
    catalog = json.loads(
        (ROOT / "website-publication/data/engine-release-history-v1.json").read_text()
    )
    entry = next(row for row in catalog["entries"] if row["id"] == "r8")
    assert entry["measurements"] == report["measurements"]
    assert [row["id"] for row in entry["measurements"]] == [
        "mmlu",
        "hle",
        "frontier",
        "swe",
        "terminal",
    ]
    assert [row["arms"][0]["requests"] for row in entry["measurements"]] == [
        200,
        200,
        160,
        64,
        48,
    ]
    for row in entry["measurements"]:
        assert [arm["role"] for arm in row["arms"]] == ["B0", "r8"]
        for arm in row["arms"]:
            assert arm["failed"] == 0
            assert arm["output_tokens"] / arm["duration_seconds"] == pytest.approx(
                arm["output_tokens_per_second"]
            )
    assert len(report["seal_audit"]) == 16
    assert sum(row["files"] for row in report["seal_audit"]) == 13021


def test_r8_context_tenant_supplement_keeps_own_seals_windows_and_budget():
    report_path = ROOT / "reports/r8-context-tenant-20261119/summary-public-v1.json"
    report = json.loads(report_path.read_text())
    catalog = json.loads(
        (ROOT / "website-publication/data/engine-release-history-v1.json").read_text()
    )
    entry = next(row for row in catalog["entries"] if row["id"] == "r8")
    assert entry["context_tenant_supplement"] == report
    assert report["engine_release"] == entry["release"]
    assert report["baseline_id"] == entry["baseline_id"]
    assert report["execution_complete"] is True
    assert report["quality_measured"] is False
    assert report["efficiency_percent"] is None
    assert [row["phase"] for row in report["comparisons"]] == [
        "4096-formal",
        "16384-formal",
        "32768-formal",
        "multi-tenant-formal",
    ]
    assert report["protocol"]["output_tokens"] == 256
    for row in report["comparisons"]:
        for role in ("B0", "B1"):
            assert len(row[role]["windows"]) == 5
            assert row[role]["metrics"]["failed"] == 0
    source = next(
        row
        for row in entry["sources"]
        if row["url"].endswith("r8-context-tenant-publication-v1.json")
    )
    assert source["sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    chart = ROOT / "website-publication" / report["chart"]["path"]
    assert hashlib.sha256(chart.read_bytes()).hexdigest() == report["chart"]["sha256"]


def test_r9_single_model_data_cannot_be_borrowed_as_original018_acceptance():
    import gzip
    import subprocess
    import sys

    catalog = json.loads(
        (ROOT / "website-publication/data/engine-release-history-v1.json").read_bytes()
    )
    entry = next(row for row in catalog["entries"] if row["id"] == "r9")
    assert [row["id"] for row in entry["measurements"]] == [
        "mmlu",
        "hle",
        "frontier",
        "swe",
        "terminal",
        "4096-formal",
        "16384-formal",
    ]
    publication = entry["submission_assessment"]
    path = ROOT / "website-publication" / publication["path"]
    raw = path.read_bytes()
    assert len(raw) == publication["bytes"]
    assert hashlib.sha256(raw).hexdigest() == publication["sha256"]
    paired = json.loads(raw)
    assert paired["execution_complete"] and paired["b0_inference_reruns"] == 0
    assert entry["measurements"] == paired["measurements"]
    assert all(
        row["control"] == "B0" and [arm["role"] for arm in row["arms"]] == ["B0", "r9"]
        for row in paired["measurements"]
    )
    cross_card = {
        row["id"]
        for row in paired["measurements"]
        if row["arms"][0]["physical_cards"] != row["arms"][1]["physical_cards"]
    }
    assert cross_card == {"hle", "swe", "terminal"}
    for row in paired["measurements"]:
        assert row["arms"][0]["physical_cards"] == [1, 6]
        assert row["arms"][1]["physical_cards"] == (
            [0, 3] if row["id"] in cross_card else [1, 6]
        )
    for phase in entry["context_tenant_supplement"]["comparisons"]:
        assert phase["B0"]["physical_cards"] == [1, 6]
        assert phase["B1"]["physical_cards"] == [0, 3]
        assert phase["B1"]["metrics"]["failed"] == 0
    native = entry["compute_reference"]
    assert native["arms"]["B0"]["physical_cards"] == [1, 6]
    assert native["arms"]["B1"]["physical_cards"] == [0, 3]
    assert native["full_reference_denominator_qualified"] is True
    assert native["original_efficiency_used"] is False
    assert len(native["arms"]["B1"]["repetitions"]) == 3

    expected = {
        "4.1": (True, False),
        "4.2": (True, True),
        "4.3": (True, False),
        "4.4": (True, True),
    }
    for metric, decisions in expected.items():
        assert (
            tuple(
                paired["targets"][metric][gate]["met"]
                for gate in ("milestone", "final")
            )
            == decisions
        )
    assert paired["targets"]["4.2"]["final_datasets"]["hle"]["met"] is False
    cost = paired["targets"]["4.4"]
    assert cost["cost_basis"] == "published_rental"
    assert cost["rental_quote"]["cny_per_card_month"] == 1875
    for key, expected_gain in (("mmlu", 66.1313904), ("four_tenant", 63.2543189)):
        row = cost["formal_cost"][key]
        unit_cost = {}
        for role, arm in row["arms"].items():
            unit_cost[role] = (
                10**6
                * 1875
                * arm["occupied_cards"]
                * float(arm["window_seconds"])
                / (30 * 86400 * arm["successful_output_tokens"])
            )
            assert float(arm["unit_cny_per_million_output_tokens"]) == pytest.approx(
                unit_cost[role]
            )
        assert 100 * (1 - unit_cost["B1"] / unit_cost["B0"]) == pytest.approx(
            expected_gain
        )
        assert float(row["reduction_percent"]) == pytest.approx(expected_gain)
    assert len(entry["mods"]) == len(set(entry["mods"])) == 12
    descriptor = entry["model_integration"]
    path = ROOT / "website-publication" / descriptor["summary"]["path"]
    summary = json.loads(path.read_bytes())
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == descriptor["summary"]["sha256"]
    )
    assert summary["scope"]["original018_pair"] is False
    assert summary["scope"]["project_full_five"] is False
    assert summary["scope"]["qwen_performance_repeated"] is False
    assert summary["candidate"]["release"] == entry["release"]
    assert summary["reference"]["kind"] == "previously-published-website-reference"
    original = gzip.decompress(
        (ROOT / summary["source"]["benchmark_path"]).read_bytes()
    )
    assert hashlib.sha256(original).hexdigest() == summary["source"]["original_sha256"]
    native = json.loads(original)
    assert summary["quality"] == {
        k: native["current_mmlu200"][k]
        for k in ("requests", "correct", "truncated", "failed", "unparsed")
    }
    for row in summary["workloads"]:
        for metric in ("output_tokens_per_second", "mean_ttft_ms", "mean_tpot_ms"):
            assert row["change_percent"][metric] == pytest.approx(
                100 * (row["candidate"][metric] / row["reference"][metric] - 1)
            )
    chart = ROOT / "website-publication" / descriptor["chart"]["path"]
    assert (
        hashlib.sha256(chart.read_bytes()).hexdigest() == descriptor["chart"]["sha256"]
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "reports/r9-intern-native-20261120/reproduce_publication.py"),
            "--check",
        ],
        check=True,
    )


def test_independent_fixed_b0_survives_projection_and_rejects_resigned_changes(
    tmp_path,
):
    source, _ = prepare(tmp_path)
    path = ROOT / "website-publication/data/fixed-b0-original018-v1.json"
    raw = path.read_bytes()
    target = source / projection.PUBLIC_FIXED_B0_PATH
    target.write_bytes(raw)
    manifest = source / "data/benchmark-source-manifest.json"
    data = json.loads(manifest.read_bytes())
    data["files"].append(
        dict(
            website_path=projection.PUBLIC_FIXED_B0_PATH,
            sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw),
        )
    )
    manifest.write_text(json.dumps(data))
    output = tmp_path / "baseline-public"
    projection.build(source, output, "test-policy")
    projection.validate(output)
    assert (output / projection.PUBLIC_FIXED_B0_PATH).read_bytes() == raw
    changed = json.loads(raw)
    changed["profiles"][0]["arm"]["output_tokens_per_second"] *= 2
    raw = json.dumps(changed).encode()
    target.write_bytes(raw)
    data["files"][-1].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="release history"):
        projection.build(source, tmp_path / "resigned-baseline", "test-policy")


def test_r10_archive_preserves_the_sealed_contiguous_restore_pairs():
    catalog = json.loads(
        (ROOT / "website-publication/data/engine-release-history-v1.json").read_bytes()
    )
    source = (ROOT / "reports/r10-fixed-five-20261120/summary.json").read_bytes()
    entry = catalog["archived_r10"]
    installed = entry["stable_evidence"]
    assert entry["publication_status"] == "stable"
    assert entry["original_candidate_release"] == json.loads(source)["engine_release"]
    measured_path = "reports/r10-contiguous-restore-fixed-five-20261121/summary.json"
    measured_raw = (ROOT / measured_path).read_bytes()
    measured = json.loads(measured_raw)
    assert installed["kind"] == "contiguous-restore"
    assert (
        measured["release"]
        == installed["measured_release"]
        == entry["release"] + "-contiguous-restore"
    )
    assert measured["baseline_id"] == entry["baseline_id"] == catalog["baseline"]["id"]
    for row in entry["measurements"]:
        pair = measured["measurements"][row["id"]]
        assert row["arms"] == [
            {**pair["B0"], "role": "B0"},
            {**pair["candidate"], "role": "r10"},
        ]
    assert [row["id"] for row in entry["measurements"]] == [
        "mmlu",
        "hle",
        "frontier",
        "swe",
        "terminal",
        "4096-formal",
        "16384-formal",
    ]
    assert [
        arm["requests"] for row in entry["measurements"][:5] for arm in row["arms"]
    ] == [200, 200, 200, 200, 160, 160, 64, 64, 48, 48]
    source_root = "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/main/"
    evidence = {binding["url"]: binding for binding in entry["sources"]}
    measured_binding = evidence[source_root + measured_path]
    assert measured_binding["sha256"] == hashlib.sha256(measured_raw).hexdigest()
    assert measured_binding["bytes"] == len(measured_raw)
    original = "reports/r10-fixed-five-20261120/summary.json"
    assert (
        evidence[source_root + original]["sha256"] == hashlib.sha256(source).hexdigest()
    )
    for filename in ("README.md", "summary.json"):
        path = f"reports/r10-native-rms-20261121/{filename}"
        payload = (ROOT / path).read_bytes()
        binding = evidence[source_root + path]
        assert binding["sha256"] == hashlib.sha256(payload).hexdigest()
        assert binding["bytes"] == len(payload)
    assert all(
        arm["failed"] == 0 and arm["end_to_end_p99_ms"] >= arm["end_to_end_p95_ms"] > 0
        for row in entry["measurements"]
        for arm in row["arms"]
    )


def test_r10_current_assessment_binds_the_complete_revised_35b_run():
    name = "r10-35b-paired-full-20261125"
    folder = ROOT / "website-publication/data" / name
    raw = (ROOT / "reports" / name / "summary.json").read_bytes()
    assert (folder / "summary.json").read_bytes() == raw
    source = json.loads(raw)
    catalog = json.loads(
        (ROOT / "website-publication/data/engine-release-history-v1.json").read_bytes()
    )
    entry = next(row for row in catalog["entries"] if row["id"] == "r10")
    assert entry["release"] == source["engine_release"] == "full-answer-b1-20261124-r10"
    assessment_raw = (folder / "acceptance.json").read_bytes()
    assessment = json.loads(assessment_raw)
    assert assessment["execution_complete"] is True
    assert entry["submission_assessment"] == {
        "path": f"data/{name}/acceptance.json",
        "bytes": len(assessment_raw),
        "sha256": hashlib.sha256(assessment_raw).hexdigest(),
    }
    for id_, gates in source["targets"].items():
        assert gates == {
            gate: assessment["targets"][id_][gate]["met"]
            for gate in ("milestone", "final")
        }
    assert entry["compute_reference"]["arms"]["B1"]["efficiency_lower_percent"] == (
        source["native_compute"]["arms"]["B1"]["efficiency_lower_percent"]
    )
    assert entry["r10_revision"]["assessment_4_3"] == assessment["targets"]["4.3"]
    assert assessment["targets"]["4.3"]["percentile_method"] == "nearest_rank"
    assert entry["context_tenant_supplement"]["arm_seals"] == source["context"]["arm_seals"]
    assert entry["context_tenant_supplement"]["quality_measured"] is False
    assert entry["context_tenant_supplement"]["effective_flops_certified"] is False
    for projected, actual in zip(
        entry["context_tenant_supplement"]["comparisons"],
        source["context"]["phases"], strict=True
    ):
        for role in ("B0", "B1"):
            assert projected[role] == actual[role]
    fixed = [row for row in entry["measurements"] if row["id"] in source["fixed_five"]]
    assert [row["id"] for row in fixed] == ["mmlu", "hle", "frontier", "swe", "terminal"]
    for row in fixed:
        for role, arm in zip(("B0", "B1"), row["arms"], strict=True):
            original = source["fixed_five"][row["id"]]["arms"][role]
            assert arm["source_seal_sha256"] == original["source_seal_sha256"]
            assert arm["output_tokens_per_second"] == original["output_tokens_per_second"]
    continuous = json.loads((folder / "continuous-c8.json").read_bytes())
    assert continuous["concurrency"] == 8
    for row in entry["measurements"][5:]:
        for role, arm in zip(("B0", "B1"), row["arms"], strict=True):
            original = continuous["arms"][role]
            assert arm["source_seal_sha256"] == original["source_seal"]
            assert arm["output_tokens_per_second"] == (
                original["formal"][row["id"]]["output_tokens_per_second"]
            )
