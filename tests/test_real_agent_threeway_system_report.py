"""Published real-Agent system-window values retain source and adverse tails."""

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/swe-bench-pro-real-agent-threeway128-1h-20261022-r2"


def test_threeway_source_and_censoring_boundaries():
    data = json.loads((REPORT / "report.json").read_text())
    assert data["status"] == "VERIFIED_SYSTEM_WINDOW"
    assert data["quality_gate"] is False
    assert data["complete_batch_makespan_claim"] is False
    assert data["configuration"]["client_concurrency_limit"] is None
    assert data["archive"]["files"] == 6293
    assert data["archive"]["whole_sha"] == data["archive"]["every_member_sha"] == "PASS"
    assert len(data["summary"]) == 12
    n, t, c = (
        data["full_window"][a]
        for a in ["native", "thunderagent_original", "controller_original"]
    )
    assert (
        c["engine_output_tokens_per_s"]
        > t["engine_output_tokens_per_s"]
        > n["engine_output_tokens_per_s"]
    )
    assert c["p95_latency_s"] > n["p95_latency_s"]
    assert c["p99_latency_s"] > n["p99_latency_s"]
    assert c["mean_latency_s"] < n["mean_latency_s"]
    assert (
        data["arm_ray_status"]["thunderagent_original"]
        == "FAILED_POST_AGENT_DRAIN_GUARD"
    )
    assert all(a["pending"] > 0 for a in data["client_boundary"].values())
    assert len(json.loads((REPORT / "data/source-sha256.json").read_text())) == 778


def test_published_numeric_and_reproduction_bytes():
    packed = json.loads((REPORT / "data/compression.json").read_text())
    for item in packed.values():
        packed_bytes = (
            (REPORT / item["gzip"]).read_bytes()
            if "gzip" in item
            else b"".join((REPORT / part).read_bytes() for part in item["gzip_parts"])
        )
        raw = gzip.decompress(packed_bytes)
        assert len(raw) == item["original_bytes"]
        assert hashlib.sha256(raw).hexdigest() == item["original_sha256"]
    rows = list(
        csv.DictReader(
            io.StringIO(
                gzip.decompress(
                    (REPORT / "data/normal-client-returns.csv.gz").read_bytes()
                ).decode()
            )
        )
    )
    report = json.loads((REPORT / "report.json").read_text())
    for arm, summary in report["full_window"].items():
        subset = [r for r in rows if r["arm"] == arm]
        assert all(0 <= float(r["t"]) < 3600 for r in subset)
        assert len(subset) / 3600 == summary["client_returns_per_s"]
        assert (
            sum(float(r["output"]) for r in subset) / 3600
            == summary["client_output_tokens_per_s"]
        )


def test_canonical_matrix_is_same_archived_measurement():
    report = json.loads((REPORT / "report.json").read_text())
    publication = ROOT / "website-publication/data"
    assert (
        json.loads(
            (
                publication / "swe-bench-pro-real-agent-threeway-20261022/report.json"
            ).read_text()
        )
        == report
    )
    matrix = json.loads((publication / "dataset-matrix-v1.json").read_text())
    dataset = next(d for d in matrix["datasets"] if d["id"] == "swe-bench-pro")
    assert dataset["evidence_type"] == "real-agent-system-window"
    assert (
        dataset["parts"][0]["scores"]["request-throttling-controller"]["value"]
        == report["full_window"]["controller_original"]["engine_output_tokens_per_s"]
    )
