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
    assert catalog["default_release"] == catalog["order"][0] == "r9"
    assert entry["measurements"] == []
    assert not entry.get("compute_reference") and not entry.get(
        "context_tenant_supplement"
    )
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
