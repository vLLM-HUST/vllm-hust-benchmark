"""A partial C8 archive cannot inherit paired or full-suite qualification."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_official_c8_descriptor_and_partial_scope():
    catalog = json.loads(
        (ROOT / "website-publication/data/engine-release-history-v1.json").read_bytes()
    )
    baseline = next(
        row for row in catalog["comparison_baselines"] if row["id"] == "official-0.23.0"
    )
    descriptor = baseline["observations"]
    raw = (ROOT / "website-publication" / descriptor["path"]).read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
        and len(raw) == descriptor["bytes"]
    )
    observation = json.loads(raw)
    assert (
        baseline["status"] == "partial"
        and baseline["pairs"] is observation["pairs"] is None
    )
    assert observation["F"] is observation["efficiency_percent"] is None
    assert observation["source_execution_status"] == "failed"
    assert (
        observation["independent_third_party"]
        is observation["necessary_work_certified"]
        is False
    )
    assert observation["runtime"]["image_digest"] == baseline["image_digest"]
    assert observation["hardware"]["driver_version"] is None
    assert observation["hardware"]["driver_version_status"] == "not-recorded"
    assert observation["hardware"]["npu_smi_version"] == "26.0.rc1"
    assert [row["id"] for row in observation["profiles"]] == [
        "4096-formal",
        "16384-formal",
    ]
    assert all(
        row["workload_kind"] == "continuous-synthetic"
        for row in observation["profiles"]
    )
    assert [row["arm"]["requests"] for row in observation["profiles"]] == [267, 137]
    for row in observation["profiles"]:
        arm = row["arm"]
        assert arm["role"] == "B0" and arm["failed"] == 0
        assert (
            arm["output_tokens"] / arm["duration_seconds"]
            == arm["output_tokens_per_second"]
        )
        assert arm["duration_seconds"] > arm["original_submission_seconds"] == 300


def test_raw_execution_and_cpu_reporter_sources_are_separate():
    manifest = json.loads(
        (
            ROOT / "reports/official-b0-023-c8-20261121-v1/archive-manifest.json"
        ).read_bytes()
    )
    assert (
        manifest["raw_execution_source"]["commit"]
        == "7ffd13b7fe9f5c9d0cdf010d2ef6a977706c34c6"
    )
    assert (
        manifest["cpu_reporter_source"]["commit"]
        == "12cf909492e63d5015cb5e548d7e3e83641db9ef"
    )
    assert manifest["model_requests_executed_by_replay"] == 0
    assert manifest["original_campaign_status"] == "failed"
