import hashlib
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "reports/acceptance-completion-20261029"


def read(name):
    return json.loads((DIRECTORY / name).read_text())


def test_retained_quality_recomputes_without_borrowing_8192_scores():
    data = read("out256-quality-audit.json")
    for arm, correct in zip(data["arms"], (21, 19), strict=True):
        assert len(arm["rows"]) == arm["total"] == 200
        assert sum(r["correct"] for r in arm["rows"]) == arm["correct"] == correct
        assert sum(r["request_success"] for r in arm["rows"]) == 200
        assert sum(r["output_tokens"] == 256 for r in arm["rows"]) == 200
        assert all(
            set(r)
            == {
                "index",
                "case_id",
                "output_sha256",
                "output_characters",
                "output_tokens",
                "request_success",
                "parsed",
                "correct",
                "contains_answer_marker",
            }
            for r in arm["rows"]
        )
    assert data["new_model_runs"] == 0
    assert data["paired_correct_changes"] == {"improved": 2, "regressed": 4}


def test_reference_diagnostic_is_not_formal_acceptance():
    data = read("reference-reassessment-20261029.json")
    row = data["graph_recomputation"][0]
    assert row["reference_tflops_per_chip"] == pytest.approx(19.081866760274465)
    assert row["reference_seconds"] / 127.59352582134306 * 100 == pytest.approx(
        row["efficiency_percent"]
    )
    assert row["efficiency_percent"] == pytest.approx(37.23021889660151)
    assert data["members_byte_verified"] == 61
    assert data["current_b0_reference_seconds"] is None
    assert data["current_v15_attainment"] is None
    assert data["new_device_measurements"] == 0


def test_raw_continuous_capsule_recomputes_and_failure_remains(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    audit = importlib.import_module("audit_fixed018_continuous")
    value = audit.audit_capsule(DIRECTORY)
    assert value == read("continuous-summary.json")
    formal = [p for p in value["b0"]["phases"] if p["name"].endswith("formal")]
    assert [p["requests"] for p in formal] == [153, 73]
    assert all(
        p["elapsed_including_drain"] > 300 and p["failures"] == 0 for p in formal
    )
    failed = value["b1_attempts"][1]
    assert failed["successful_warmup"] == 24
    assert failed["failed_requests"] == 8 and failed["successful_formal"] == 0
    assert failed["activation"]["request_scoped_recompute_admissions"] == 32
    assert failed["long16k_executed"] is False
    assert value["paired_continuous_stability_pass"] is False
    assert value["paired_16k_throughput_gain_percent"] is None


def test_five_task_counts_and_unknown_lifecycle_are_retained():
    data = read("five-task-stability.json")
    assert len(data["review"]) == 5 and data["five_task_stability_acceptance"] is None
    assert [r["cases"] for r in data["solver_runs"]] == [668, 1143, 160, 160]
    assert [r["context_ge16k"] for r in data["solver_runs"]] == [380, 706, 43, 41]
    for item in data["sources"]:
        assert (
            hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    runs = read("hle-lifecycle-audit.json")["runs"]
    assert runs[0]["receipt_fields"]["status"] == "revision-run"
    assert runs[2]["receipt_fields"]["status"] == "failed"
    assert runs[3]["receipt_fields"]["all_devices_released"] is True


def test_manifest_binds_audit_and_original_source_bytes():
    for name, digest in read("manifest.json")["files"].items():
        assert hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() == digest
