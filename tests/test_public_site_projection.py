"""Public visibility must not change retained measurements or original sources."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/build_public_site.py"
spec = importlib.util.spec_from_file_location("public_projection", MODULE)
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_projection_retains_other_results_and_input(tmp_path):
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    (source / "assets").mkdir()
    for name in [
        "index.html",
        "plugins.html",
        "dataset-introductions.html",
        "leaderboard-runs.html",
    ]:
        (source / name).write_text(
            '<main><a href="betterscale.html">Research</a><p>Native</p></main>'
        )
    (source / "assets/site.js").write_text("// shared presentation\n")
    (source / "betterscale.html").write_text("withdrawn implementation")
    retained = {
        "id": "native-run",
        "engine": "Native",
        "throughput": 123.45,
        "status": "measured",
    }
    records = {
        "points": [
            retained,
            {"id": "run-private", "engine": "BetterScale", "throughput": 999},
        ],
        "refs": ["native-run", "run-private"],
    }
    raw = (json.dumps(records) + "\n").encode()
    (source / "data/leaderboard.json").write_bytes(raw)
    manifest = {
        "repository": "vLLM-HUST/vllm-hust-benchmark",
        "files": [
            {
                "website_path": "data/leaderboard.json",
                "benchmark_path": "website-publication/data/leaderboard.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        ],
    }
    (source / "data/benchmark-source-manifest.json").write_text(json.dumps(manifest))
    before = snapshot(source)
    output = tmp_path / "public"
    projection.build(source, output, "test-policy")
    projection.validate(output)
    assert snapshot(source) == before
    projected = json.loads((output / "data/leaderboard.json").read_text())
    assert projected == {"points": [retained], "refs": ["native-run"]}
    assert "betterscale.html" not in (output / "index.html").read_text()
    entry = json.loads((output / "data/benchmark-source-manifest.json").read_text())[
        "files"
    ][0]
    assert entry["unprojected_sha256"] == hashlib.sha256(raw).hexdigest()
    assert entry["sha256"] != entry["unprojected_sha256"]


def test_refuses_to_overwrite_or_write_inside_source(tmp_path):
    with pytest.raises(ValueError):
        projection.build(tmp_path, tmp_path / "public", "test")
    with pytest.raises(ValueError):
        projection.build(tmp_path, tmp_path, "test")


def test_combined_record_is_withheld_as_a_whole():
    record = {"id": "combined", "mods": ["PegaFlow", "LatchMoE"], "value": 42}
    assert projection.project_json([record], set()) == []


def test_withdrawn_comparison_does_not_leave_an_unverifiable_native_point():
    record = {
        "id": "comparison-off",
        "configuration": {"engine": "Native"},
        "metrics": {"throughput": 42},
        "evidence": {"url": "https://example.org/FRONTIER-PREFIX-ROUTING-MEAN.md"},
    }
    assert projection.project_json([record], set()) == []


def test_manifest_tampering_is_detected(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/result.json").write_text("{}")
    manifest = {
        "files": [
            {"website_path": "data/result.json", "sha256": "incorrect", "bytes": 2}
        ]
    }
    (tmp_path / "data/benchmark-source-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        projection.validate(tmp_path)


def test_frontier_prunes_missing_series_without_changing_points():
    point = {
        "id": "retained",
        "cohort_id": "a",
        "load": {"concurrency_series": "native"},
        "metrics": {"output_tps": 42},
    }
    data = {
        "schema_version": "leaderboard-frontier/v1",
        "cohorts": [
            {
                "id": "a",
                "workload": {
                    "contract": {"display_series_ids": ["native", "withdrawn"]}
                },
            },
            {"id": "empty", "workload": {"contract": {}}},
        ],
        "points": [point],
    }
    removed = set()
    projection.prune_frontier(data, removed)
    assert data["points"] == [point]
    assert data["cohorts"] == [
        {"id": "a", "workload": {"contract": {"display_series_ids": ["native"]}}}
    ]
    assert removed == {"empty"}


def test_object_valued_id_is_not_treated_as_a_record_reference():
    assert projection.project_json(
        {"nested": {"id": {"type": "descriptive"}}}, {"gone"}
    ) == {"nested": {"id": {"type": "descriptive"}}}
