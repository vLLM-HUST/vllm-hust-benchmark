"""Actual scalar projections must recompute and keep original telemetry bytes."""

import hashlib
import importlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/full-answer-joint-20261030"


def test_published_scalars_bind_actual_complete_pair(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    math = importlib.import_module("recompute_full_answer_joint")
    pair = json.loads((EVIDENCE / "joint-summary.json").read_text())
    derived = math.paired_metrics(pair)
    for key in (
        "paired_quality",
        "output_throughput_change_percent",
        "rental_cost_reduction_percent",
    ):
        assert derived[key] == pair[key]
    for arm, values in zip(pair["arms"], derived["arms"], strict=True):
        for key in (
            "completed",
            "failed",
            "correct",
            "truncated",
            "quality_by_finish",
            "timing",
            "successful_output_tokens",
            "output_tokens_per_second",
        ):
            assert arm[key] == values[key]
        assert arm["cost"]["request_phase_cny"] == values["request_phase_cny"]
        assert (
            arm["cost"]["cny_per_million_output_tokens"]
            == values["cny_per_million_output_tokens"]
        )
        assert arm["runtime"]["role"] == arm["role"]
        assert all(len(row["raw_sse_sha256"]) == 64 for row in arm["rows"])
    assert pair["arms"][0]["run"] == "joint-b0-r2"
    assert pair["arms"][1]["run"] == "joint-b1-r4"
    assert pair["same_run_quality_and_performance"]
    assert pair["statistical_noninferiority_established"] is False
    assert pair["protocol"]["b1_host_state_budget_bytes_per_rank"] == 4 * 2**30
    assert "/root/" not in (EVIDENCE / "joint-summary.json").read_text()


def test_telemetry_is_unchanged_and_excludes_private_task_content():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected
    telemetry = json.loads((EVIDENCE / "telemetry-manifest.json").read_text())
    with tarfile.open(EVIDENCE / "original-telemetry.tar.gz") as archive:
        assert {member.name for member in archive} == set(telemetry["files"])
        for name, expected in telemetry["files"].items():
            assert not Path(name).is_absolute() and ".." not in Path(name).parts
            assert not name.endswith(".sse.jsonl.gz")
            assert Path(name).name not in (
                "request-plan.json",
                "plan.json",
                "service.log",
            )
            raw = archive.extractfile(name).read()
            assert hashlib.sha256(raw).hexdigest() == expected
            assert b"/root/" not in raw and b"Authorization" not in raw


def test_continuous_current_and_failed_configuration_are_separate():
    pair = json.loads((EVIDENCE / "continuous-summary.json").read_text())
    assert pair["b0"]["runtime"]["vllm_version"] == "0.18.0"
    assert pair["b0"]["runtime"]["ascend_version"] == "0.18.0"
    assert pair["b1_host_state_budget_bytes_per_rank"] == 4 * 2**30
    assert [
        p["requests"] for p in pair["b1"]["phases"] if p["name"].endswith("formal")
    ] == [408, 183]
    assert all(p["failures"] == 0 for p in pair["b1"]["phases"])
    assert pair["b1_activation"]["request_scoped_recompute_admissions"] == 655
    assert pair["b1_activation"]["reused_tokens"] == 0
    failure = json.loads((EVIDENCE / "retained-r3-failure.json").read_text())
    assert failure["host_budget_per_rank_bytes"] == 8 * 2**30
    assert failure["phases"]["4096-formal"]["failed"] == 8
    assert failure["sixteen_k_executed"] is False
    assert failure["failure"]["code"] == 207001
    history = json.loads((EVIDENCE / "attempt-history.json").read_text())["attempts"]
    assert [a["receipt"]["status"] for a in history] == [
        "failed",
        "failed",
        "failed",
        "completed",
    ]
    assert history[1]["receipt"]["model_run_started"] is False
    assert all(a["receipt"]["all_reserved_devices_released"] for a in history)


def test_current_publication_matches_report_and_actual_pair(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    builder = importlib.import_module("build_midterm_v103")
    published = json.loads((ROOT / builder.OUTPUT).read_text())
    rebuilt = builder.build(
        published["source_commit"], lambda p: (ROOT / p).read_bytes()
    )
    assert rebuilt == published
    assert published["current_summary"]["quality"]["b0"]["correct"] == 143
    assert published["current_summary"]["quality"]["b1"]["candidate_correct"] == 153
    assert (
        published["full_answer_joint"]["arms"][0]["runtime"]["vllm_version"] == "0.18.0"
    )
    assert published["wholechain_reference"]["B1"]["eta_percent"] is None
