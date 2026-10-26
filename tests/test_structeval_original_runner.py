"""CPU interface safety only; no benchmark data or model execution."""

from pathlib import Path
import subprocess
import sys
import json
import importlib.util

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_nonrendering_protocol_selection_keeps_original_rows_and_boundaries():
    spec = importlib.util.spec_from_file_location(
        "structeval_selection", ROOT / "scripts/run_structeval_original_json.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rows = [
        {
            "input_type": "Text",
            "output_type": "JSON",
            "rendering": False,
            "task_id": str(index),
        }
        for index in range(5)
    ]
    rows.insert(1, {"input_type": "Text", "output_type": "HTML", "rendering": True})
    selected, available = runner.select_original_rows(rows, start=3, count=2)
    assert available == 5
    assert [index for index, row in selected] == [4, 5]
    assert [row["task_id"] for index, row in selected] == ["3", "4"]
    assert len(runner.select_original_rows(rows, full=True)[0]) == 5
    with pytest.raises(ValueError, match="range"):
        runner.select_original_rows(rows, start=4, count=2)
    with pytest.raises(ValueError, match="no original"):
        runner.select_original_rows(rows, output_type="CSV")


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


def test_nonrendering_native_audit_keeps_source_incomplete():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/structeval-nonrender-protocols-audit-20261024.json"
        ).read_text("utf-8")
    )
    protocols = receipt["protocols"]
    assert len(protocols) == len({p["protocol"] for p in protocols}) == 18
    assert "Text-to-JSON" not in {p["protocol"] for p in protocols}
    assert all(p["tasks"] == 3 and p["official_rescore_equal"] for p in protocols)
    ids = [task_id for p in protocols for task_id in p["task_ids"]]
    assert len(ids) == len(set(ids)) == receipt["new_unique_tasks"] == 54
    assert receipt["historical_text_to_json_tasks_reused"] == 50
    assert receipt["audit_model_calls"] == 0
    assert receipt["official_saved_output_rescore_equal"] is True
    assert receipt["full_source_completed"] is False
    assert receipt["nonrendering_total_tasks"] == 950
    assert receipt["full_source_tasks"] == 2035


def test_completed_protocol_audit_adds_only_previously_unexecuted_rows():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/structeval-nonrender-complete-protocols-audit-20261024.json"
        ).read_text("utf-8")
    )
    protocols = receipt["completed_additional_protocols"]
    assert protocols
    assert all(p["new_tasks"] == 47 and p["reused_tasks"] == 3 for p in protocols)
    assert all(len(set(p["all_original_task_ids"])) == 50 for p in protocols)
    assert receipt["new_tasks_audited"] == 47 * len(protocols)
    assert receipt["verified_unique_native_tasks"] == 104 + 47 * len(protocols)
    assert receipt["audit_model_calls"] == 0
    assert receipt["official_saved_output_rescore_equal"]
    assert not receipt["full_source_completed"]
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]["STRUCTEVAL"]
    assert not ledger["checked"]
    # The earlier audit is immutable; current counts include a separate,
    # explicitly linked incremental checkpoint rather than rewriting history.
    count = receipt["verified_unique_native_tasks"]
    seen = {p["protocol"] for p in protocols} | {"Text-to-JSON"}
    chain = ledger["completed_protocols_checkpoint_chain"]
    assert chain and chain[-1] == ledger["latest_completed_protocols_checkpoint"]
    for path in chain:
        checkpoint = json.loads((ROOT / path).read_text("utf-8"))
        assert checkpoint["previous_unique_original_tasks_reused"] == count
        names = {p["protocol"] for p in checkpoint["protocols"]}
        assert names and not seen & names
        assert checkpoint["new_unique_original_tasks_audited"] == 47 * len(names)
        assert all(
            p["new_original_tasks"] == 47 and p["previous_original_tasks_reused"] == 3
            for p in checkpoint["protocols"]
        )
        count += checkpoint["new_unique_original_tasks_audited"]
        seen.update(names)
        assert checkpoint["verified_unique_native_tasks"] == count
        assert checkpoint["full_nonrendering_protocols"] == len(seen)
        assert checkpoint["audit_model_calls"] == 0
        assert not checkpoint["full_source_completed"]
    assert ledger["verified_unique_native_tasks"] == count
    held = receipt.get("held_native_exit_evidence")
    if held:
        assert held["native_exit_code"] == 0
        assert held["controller_launch_held"]
        assert held["protocol"] in {p["protocol"] for p in protocols}


def test_saved_output_handoff_and_readback_do_not_claim_rendered_delivery():
    checkpoint = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/structeval-nonrender-checkpoint3-20261025.json"
        ).read_text("utf-8")
    )
    saved = checkpoint["public_saved_output_handoff_20261026"]
    readback = checkpoint["full_original_provider_readback_20261026"]
    preflight = checkpoint["render_environment_preflight_20261026"]
    assert saved["unique_original_tasks"] == 950
    assert saved["complete_protocols"] == 19
    assert saved["official_scores_equal_saved"]
    assert saved["new_model_calls"] == 0
    assert not saved["source_checkbox_eligible"]
    assert readback["original_rows_and_index_rows"] == 2035
    assert readback["original_fields_per_row"] == 10
    assert (
        readback["public_and_research_roundtrips"]
        == 3 * readback["protocol_groups_including_rendering_flag"]
    )
    assert (
        readback["remaining_rendered_vqa_tasks"] + saved["unique_original_tasks"]
        == 2035
    )
    assert not readback["full_native_source_delivery_claim"]
    assert not preflight["generated_code_executed"]
    assert not preflight["filesystem_process_memory_isolation_qualified"]
    assert preflight["model_calls"] == 0
    assert not checkpoint["full_source_completed"]


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
