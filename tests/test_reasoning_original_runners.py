"""CPU CLI checks; not task execution evidence."""

from pathlib import Path
import subprocess
import sys
import importlib.util
import json

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gsm8k_delivery_preserves_native_scope_and_no_rerun_boundary():
    docs = ROOT / "docs/workload-data"
    ledger = json.loads((docs / "dataset-validation-status.json").read_text("utf-8"))
    item = ledger["delivery_inventory"]["GSM8K"]
    receipt = json.loads((ROOT / item["delivery_receipt"]).read_text("utf-8"))
    cumulative = json.loads(
        (docs / "validation" / receipt["native_execution_receipt"]).read_text("utf-8")
    )
    assert item["checked"] and item["status"] == "AVAILABLE"
    assert (
        receipt["verified_unique_original_tasks"]
        == cumulative["cumulative"]["unique_task_count"]
        == 204
    )
    assert (
        receipt["available_original_tasks"] == cumulative["original_test_size"] == 1319
    )
    assert receipt["source_checkbox_eligible"]
    assert not receipt["full_suite_validated"]
    assert not receipt["pooled_accuracy_reported"]
    assert receipt["new_model_calls"] == 0
    assert (ROOT / receipt["public_runner"]).is_file()
    assert (ROOT / receipt["documentation"]).is_file()


def test_arc_native_tail_receipt_covers_both_suites_and_actual_ids():
    docs = ROOT / "docs/workload-data"
    ledger = json.loads((docs / "dataset-validation-status.json").read_text("utf-8"))
    item = ledger["delivery_inventory"]["AI2_ARC"]
    delivery = json.loads((ROOT / item["delivery_receipt"]).read_text("utf-8"))
    assert item["checked"] and item["status"] == "AVAILABLE"
    assert delivery["verified_unique_original_tasks"] == 12
    assert sum(len(row["original_doc_ids"]) for row in delivery["subprotocols"]) == 12
    assert not delivery["full_suite_validated"]
    receipt = json.loads(
        (
            ROOT / "docs/workload-data/validation/arc-tail-native-audit-20261024.json"
        ).read_text("utf-8")
    )
    assert receipt["status"] == "ORIGINAL_NATIVE_EXECUTION_AND_SCORING_AUDITED"
    assert receipt["audit_new_model_calls"] == 0
    assert receipt["full_test_execution"] is False
    checks = {item["suite"]: item for item in receipt["checks"]}
    for suite, ids in [
        ("ARC-Challenge", [1169, 1170, 1171]),
        ("ARC-Easy", [2373, 2374, 2375]),
    ]:
        item = checks[suite]
        assert item["original_doc_ids"] == ids
        assert item["actual_samples"] == 3
        assert item["candidate_likelihood_requests"] == 12
        assert item["inputs_choices_references_equal_original"] is True
        assert "acc,none" in item["metrics"]
        assert "acc_norm,none" in item["metrics"]
        assert (
            item["runner_sha256"]
            == __import__("hashlib")
            .sha256((ROOT / "scripts/run_arc_original_samples.py").read_bytes())
            .hexdigest()
        )


def test_arc_native_selection_supports_defaults_tail_and_full_without_rewriting():
    spec = importlib.util.spec_from_file_location(
        "arc_native", ROOT / "scripts/run_arc_original_samples.py"
    )
    arc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arc)
    assert arc.selected_indices("ARC-Easy") == [0, 1, 2]
    assert arc.selected_indices("ARC-Easy", 2373, 3) == [2373, 2374, 2375]
    assert arc.selected_indices("ARC-Challenge", full=True) == list(range(1172))
    assert len(arc.selected_indices("ARC-Easy", full=True)) == 2376
    for start, count in [(-1, 3), (1171, 2), (0, 0)]:
        with pytest.raises(ValueError, match="range"):
            arc.selected_indices("ARC-Challenge", start, count)
    with pytest.raises(ValueError, match="full"):
        arc.selected_indices("ARC-Easy", 1, 3, True)
    receipt = json.loads(
        (
            ROOT / "docs/workload-data/validation/reasoning-ci-assets-20261024.json"
        ).read_text("utf-8")
    )
    for suite, count in arc.TEST_COUNTS.items():
        file = next(
            item
            for item in receipt["files"]
            if item["source_id"] == "AI2_ARC"
            and item["file"] == f"{suite}/test-00000-of-00001.parquet"
        )
        assert file["original_rows_field_checked"] == count


@pytest.mark.parametrize(
    "name", ["run_gsm8k_incremental_samples.py", "run_mmlu_pro_original_samples.py"]
)
def test_help_without_task_dependencies(name):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--data-root" in result.stdout


def test_incremental_range_rejected_before_data_or_model_access(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_gsm8k_incremental_samples.py"),
            "--data-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--endpoint",
            "http://127.0.0.1:1/v1/chat/completions",
            "--model",
            "unused",
            "--start",
            "1318",
            "--count",
            "6",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "within the 1319" in result.stderr
    assert not (tmp_path / "out").exists()


def test_arc_runner_help_lists_both_original_suites_without_optional_dependencies():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_arc_original_samples.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ARC-Challenge" in result.stdout
    assert "ARC-Easy" in result.stdout
