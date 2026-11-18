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
    assert {entry["id"] for entry in value["entries"]} == {"r4", "r5", "r6", "r7", "r8"}
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
