import importlib.util
import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vllm-hust-midterm-20261019"
SPEC = importlib.util.spec_from_file_location(
    "b1_audit", REPORT / "audit_b1_candidates.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_published_audit_is_reproducible():
    actual = MODULE.audit(ROOT / "leaderboard-data/dataset-validation")
    saved = json.loads((REPORT / "b1-selection-audit.json").read_text())
    assert actual == {k: v for k, v in saved.items() if k != "website_audit"}
    assert actual["selected_cell_count"] == 30
    assert actual["non_best_selections"] == 0


def test_audit_detects_wrong_selection_and_respects_direction(tmp_path):
    cell = {
        "dataset_id": "fixture",
        "metric_id": "latency",
        "comparison": {"direction": "lower_is_better"},
        "selected_candidate_id": "slow",
        "candidate_values": [
            {"candidate_id": "fast", "value": 1},
            {"candidate_id": "slow", "value": 2},
        ],
    }
    (tmp_path / "dataset_validation_fixture.json").write_text(
        json.dumps({"results": [cell]})
    )
    assert MODULE.audit(tmp_path)["non_best_selections"] == 1
    cell["selected_candidate_id"] = "fast"
    (tmp_path / "dataset_validation_fixture.json").write_text(
        json.dumps({"results": [cell]})
    )
    assert MODULE.audit(tmp_path)["non_best_selections"] == 0


def test_report_and_publication_share_definition():
    definition = "当前已实测最优的vLLM-HUST优化方案"
    for path in [
        REPORT / "中期测试报告.md",
        REPORT / "INTERNAL-ACCEPTANCE-v1.md",
        ROOT / "leaderboard-data/dataset-validation/README.md",
    ]:
        assert definition in path.read_text()


def test_screen_archive_and_candidate_identity(tmp_path):
    result = json.loads((REPORT / "b1-e32-screen.json").read_text())
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
    command = json.loads((root / "launch-candidate-command.json").read_text())
    assert command[command.index("--max-num-seqs") + 1] == "32"
    config = json.loads(command[command.index("--additional-config") + 1])
    assert config["state_resident_seats"] == 36
    assert not result["selected_as_b1"]
    assert not result["quality_rescored"]
    if result["status"] == "measured":
        spec = importlib.util.spec_from_file_location(
            "screen_verify", REPORT / "verify_aligned_pair.py"
        )
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        assert verifier.verify(root, result["run_id"]) == result["measured"]
        assert result["throughput_change_vs_prior_pct"] == 100 * (
            result["measured"]["output_tps"] / result["prior_candidate_output_tps"] - 1
        )
        report = (REPORT / "中期测试报告.md").read_text()
        assert f"{result['measured']['output_tps']:.6f} token/秒" in report
        assert f"下降{abs(result['throughput_change_vs_prior_pct']):.2f}%" in report
        assert f"提高{result['throughput_change_vs_baseline_pct']:.2f}%" in report
    else:
        assert "measured" not in result
