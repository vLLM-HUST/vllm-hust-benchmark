from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from vllm_hust_benchmark.baseline_recovery import build_recovery_audit


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_current_baselines_require_rerun_after_input_identity_revision() -> None:
    report = build_recovery_audit(REPO_ROOT, generated_at="2026-08-02T00:00:00Z")
    summary = report["summary"]
    assert summary["scanned"] == 19
    assert summary["active_public_candidates"] == 9
    assert summary["recoverable"] == 0
    assert summary["recoverable"] + summary["rerun_required"] == 9
    assert summary["provisional_or_specialty"] == 10
    assert len(report["rerun_specs"]) == summary["rerun_required"]
    assert (
        sum(record["disposition"] == "rerun-required" for record in report["records"])
        == summary["rerun_required"]
    )
    assert (
        sum(
            record["disposition"] == "not-public-candidate"
            for record in report["records"]
        )
        == 10
    )
    assert "REPEAT_COUNT=3" in report["rerun_args"]
    assert "MIN_SUCCESSFUL_REPEATS=3" in report["rerun_args"]
    assert "FORCE_RUN_EXISTING" not in report["rerun_command"]
    assert report["rerun_args"][0] == "env"
    assert "scripts/run-official-ascend-goal-baseline-matrix.sh" in report["rerun_args"]


def test_visionarena_remains_blocked_by_max_model_len_contract() -> None:
    report = build_recovery_audit(REPO_ROOT, generated_at="2026-08-02T00:00:00Z")
    record = next(
        item for item in report["records"] if "visionarena-online" in item["target_id"]
    )
    mismatch = next(
        item
        for item in record["exact_mismatches"]
        if item["field"] == "server_parameters.max_model_len"
    )
    assert mismatch == {
        "field": "server_parameters.max_model_len",
        "expected": 32768,
        "actual": 30720,
        "kind": "value-mismatch",
    }
    assert "verified-attestation-missing" in record["reasons"]


def test_manifest_is_audited_but_does_not_imply_verification() -> None:
    report = build_recovery_audit(REPO_ROOT, generated_at="2026-08-02T00:00:00Z")
    record = next(
        item
        for item in report["records"]
        if "random-online-qwen25-14b-910b2" in item["target_id"]
    )
    assert record["evidence"]["referenced_by_manifest"] is True
    assert record["evidence"]["independent_files"] == []
    assert "verified-attestation-missing" in record["reasons"]


def test_prefix_lengths_are_normalized_without_false_input_output_mismatch() -> None:
    report = build_recovery_audit(REPO_ROOT, generated_at="2026-08-02T00:00:00Z")
    record = next(
        item
        for item in report["records"]
        if item["target_id"].endswith("prefix-repetition-online-qwen25-14b-910b2")
    )
    fields = {item["field"] for item in record["exact_mismatches"]}
    assert "client_parameters.input_len" not in fields
    assert "client_parameters.output_len" not in fields
    assert record["evidence"]["registry_binding"] == "unverified"
    assert "target-registry-hash-missing-or-mismatched" in record["reasons"]


def test_audit_is_json_serializable() -> None:
    report = build_recovery_audit(REPO_ROOT, generated_at="2026-08-02T00:00:00Z")
    json.dumps(report)


def test_cli_stdout_is_machine_readable_and_require_recoverable_fails() -> None:
    command = [
        sys.executable,
        "scripts/audit_official_baseline_recovery.py",
        "--repo-root",
        str(REPO_ROOT),
    ]
    completed = subprocess.run(
        command, cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["summary"]["recoverable"] == 0
    assert "baseline recovery audit:" in completed.stderr

    required = subprocess.run(
        [*command, "--require-recoverable"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert required.returncode == 2
    assert json.loads(required.stdout)["summary"]["recoverable"] == 0
