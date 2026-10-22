"""CPU interface safety only; no benchmark data or model execution."""

from pathlib import Path
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parents[1]


def test_full_text_json_receipt_is_not_full_structeval_claim():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/structeval-text-json-full-npu-20261022.json"
        ).read_text("utf-8")
    )
    assert receipt["unique_tasks"] == 50
    assert len(set(receipt["selected_samples"])) == 50
    assert receipt["all_text_to_json_requested"] is True
    assert receipt["full_structeval_size"] == 2035
    assert len(receipt["tasks"]) == 50
    assert (
        sum(t["final_eval_score"] for t in receipt["tasks"]) / 50
        == receipt["mean_official_final_score"]
    )


def test_help_without_optional_task_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_structeval_original_json.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--all-json" in result.stdout


def test_external_endpoint_refused_before_inputs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_structeval_original_json.py"),
            "--data-file",
            str(tmp_path / "missing"),
            "--official-archive",
            str(tmp_path / "missing.zip"),
            "--output",
            str(tmp_path / "out"),
            "--endpoint",
            "https://example.com/v1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "only local HTTP NPU" in result.stderr
    assert not (tmp_path / "out").exists()
