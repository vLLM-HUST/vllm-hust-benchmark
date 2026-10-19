"""Verify same-workload follow-up artifacts without requiring NPU hardware."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest


REPORT = Path(__file__).resolve().parents[1] / "reports/vllm-hust-midterm-20261019"


@pytest.mark.parametrize(
    "candidate,seats", [("e24", "24"), ("kvmat", "16"), ("kvmat-e24", "24")]
)
def test_followup_evidence(candidate, seats, tmp_path):
    path = REPORT / f"b1-{candidate}-screen.json"
    result = json.loads(path.read_text())
    archive_path = REPORT / result["archive"]["file"]
    assert (
        hashlib.sha256(archive_path.read_bytes()).hexdigest()
        == result["archive"]["sha256"]
    )
    with tarfile.open(archive_path) as archive:
        archive.extractall(tmp_path, filter="data")
    root = tmp_path / "20261019-midterm-aligned"
    for name, digest in result["archive"]["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    command = json.loads((root / f"launch-{candidate}-command.json").read_text())
    assert command[command.index("--max-num-seqs") + 1] == seats
    config = json.loads(command[command.index("--additional-config") + 1])
    assert config["state_resident_seats"] == 36
    assert config["state_cache_host_bytes"] == 8589934592
    assert not result["quality_rescored"]
    if result["status"] != "measured":
        assert "measured" not in result
        assert not result["selected_as_b1"]
        return
    spec = importlib.util.spec_from_file_location(
        "followup_verifier", REPORT / "verify_aligned_pair.py"
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    assert verifier.verify(root, result["run_id"]) == result["measured"]
    assert result["throughput_change_vs_prior_pct"] == 100 * (
        result["measured"]["output_tps"] / result["prior_candidate_output_tps"] - 1
    )
    if result["throughput_change_vs_prior_pct"] <= 0:
        assert not result["selected_as_b1"]
    if candidate.startswith("kvmat"):
        control = result["control_evidence"]
        assert control["partial_reuse"] == "not-exercised"
        assert control["engine_salt_admissions"] >= 200


def test_publication_is_derived_from_evidence_and_selects_each_metric(monkeypatch):
    monkeypatch.syspath_prepend(str(REPORT))
    path = (
        REPORT.parents[1]
        / "leaderboard-data/dataset-validation/dataset_validation_qwen35_mmlu_school028_inf_out256.json"
    )
    saved = json.loads(path.read_text())
    spec = importlib.util.spec_from_file_location(
        "followup_publish", REPORT / "publish_mmlu_followup.py"
    )
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    assert publisher.build(saved["source"]["commit"]) == saved
    assert not saved["scenario"]["accuracy_evaluated"]
    assert len(saved["results"]) == 3
    for cell in saved["results"]:
        assert len(cell["candidate_values"]) == 6
        values = [row["value"] for row in cell["candidate_values"]]
        expected = (
            max if cell["comparison"]["direction"] == "higher_is_better" else min
        )(values)
        assert cell["value"] == expected
        assert cell["delta_pct"] == 100 * (expected / cell["baseline_value"] - 1)
        assert "Task accuracy was not rescored" in cell["note"]


def test_latency_validator_rejects_larger_selected_value():
    from vllm_hust_benchmark.dataset_validation import (
        DatasetValidationError,
        validate_artifact,
    )

    path = (
        REPORT.parents[1]
        / "leaderboard-data/dataset-validation/dataset_validation_qwen35_mmlu_school028_inf_out256.json"
    )
    saved = json.loads(path.read_text())
    validate_artifact(saved, expected_scenario_id=saved["scenario"]["id"])
    cell = next(row for row in saved["results"] if row["metric_id"] == "ttft_mean")
    worst = max(cell["candidate_values"], key=lambda row: row["value"])
    cell["selected_candidate_id"] = worst["candidate_id"]
    cell["value"] = worst["value"]
    with pytest.raises(
        DatasetValidationError, match="not the best candidate value for lower"
    ):
        validate_artifact(saved, expected_scenario_id=saved["scenario"]["id"])


def test_mod_source_audit_is_pinned_and_hashes_match():
    audit = json.loads((REPORT / "b1-mod-source-audit.json").read_text())
    assert len(audit["repositories"]) == 9
    for repo in audit["repositories"]:
        assert len(repo["commit"]) == 40
        assert repo["default_branch"]
        assert any(source["path"] == "README.md" for source in repo["sources"])
        for source in repo["sources"]:
            assert f"/blob/{repo['commit']}/" in source["url"]
            assert (
                hashlib.sha256(source["text"].encode()).hexdigest() == source["sha256"]
            )
    for source in audit["measured_host_sources"]:
        assert hashlib.sha256(source["text"].encode()).hexdigest() == source["sha256"]
