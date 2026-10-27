"""CPU interface/claim checks, not original dataset or model evidence."""

import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_v2_runner():
    spec = importlib.util.spec_from_file_location(
        "longbench_v2_runner", ROOT / "scripts/run_longbench_v2_original_samples.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def capacity_software_fixture(count=3):
    return {
        "model": "software-fixture-model",
        "model_max_length": 131072,
        "selected": [
            {
                "row_index": index,
                "id": f"software-fixture-{index}",
                "prompt_sha256": f"{index:064x}",
                "full_prompt_tokens": 100,
                "status": "FULL_INPUT_FITS",
            }
            for index in range(count)
        ],
    }


@pytest.mark.parametrize("count", [1, 3, 4, 503])
def test_v2_explicit_selection_has_no_three_task_hardcoding(count):
    capacity = capacity_software_fixture(count)
    selected = load_v2_runner().validate_capacity_selection(
        capacity, "software-fixture-model"
    )
    assert selected is capacity["selected"]
    assert [row["row_index"] for row in selected] == list(range(count))


def test_v2_historical_three_selection_is_accepted_without_changes():
    capacity = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/longbench-v2-capacity-selection-20261022.json"
        ).read_text("utf-8")
    )
    selected = load_v2_runner().validate_capacity_selection(capacity, capacity["model"])
    assert selected is capacity["selected"]
    assert [row["row_index"] for row in selected] == [1, 4, 6]


def test_v2_explicit_256k_binding_preserves_full_input_and_original_budget():
    runner = load_v2_runner()
    capacity = capacity_software_fixture()
    capacity["model_max_length"] = 262144
    capacity["selected"][0]["full_prompt_tokens"] = 262016
    assert runner.validate_capacity_selection(capacity, capacity["model"], 262144)
    capacity["selected"][0]["full_prompt_tokens"] += 1
    with pytest.raises(ValueError, match="complete original inputs"):
        runner.validate_capacity_selection(capacity, capacity["model"], 262144)
    with pytest.raises(ValueError, match="model/window"):
        runner.validate_capacity_selection(capacity, capacity["model"])


def test_v2_live_deployment_binding_rejects_missing_wrong_or_duplicate_model():
    runner = load_v2_runner()
    valid = {"id": "software-fixture", "max_model_len": 262144}
    runner.validate_deployment([valid], "software-fixture", 262144)
    for models in [
        [],
        [valid, valid],
        [{"id": "other", "max_model_len": 262144}],
        [{"id": "software-fixture", "max_model_len": 131072}],
    ]:
        with pytest.raises(ValueError, match="live model/window"):
            runner.validate_deployment(models, "software-fixture", 262144)


def test_v2_updated_entry_does_not_relabel_historical_npu_evidence():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]["LONGBENCH_V2"]
    assert ledger["explicit_selection_entry_software_checked"]
    assert not ledger["entry_native_after_update_checked"]
    assert not ledger["checked"]
    assert ledger["delivery_receipt"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("row_index", -1),
        ("row_index", 503),
        ("row_index", True),
        ("full_prompt_tokens", 0),
        ("full_prompt_tokens", -1),
        ("full_prompt_tokens", True),
        ("full_prompt_tokens", 130945),
        ("status", "RESOURCE_LIMIT_FULL_CONTEXT"),
        ("id", ""),
        ("prompt_sha256", "incomplete-software-fixture"),
    ],
)
def test_v2_invalid_or_overflow_task_rejects_entire_selection(field, value):
    capacity = capacity_software_fixture()
    capacity["selected"][0][field] = value
    with pytest.raises(ValueError):
        load_v2_runner().validate_capacity_selection(capacity, "software-fixture-model")
    assert len(capacity["selected"]) == 3  # No filtering into an easier subset.


def test_v2_selection_order_and_identity_are_fail_closed():
    runner = load_v2_runner()
    for field in ("row_index", "id", "prompt_sha256"):
        capacity = capacity_software_fixture()
        capacity["selected"][1][field] = capacity["selected"][0][field]
        with pytest.raises(ValueError, match="duplicate"):
            runner.validate_capacity_selection(capacity, "software-fixture-model")
    capacity = capacity_software_fixture()
    capacity["selected"].reverse()
    with pytest.raises(ValueError, match="original row order"):
        runner.validate_capacity_selection(capacity, "software-fixture-model")
    for count in (0, 504):
        with pytest.raises(ValueError, match="one to 503"):
            runner.validate_capacity_selection(
                capacity_software_fixture(count), "software-fixture-model"
            )
    capacity = capacity_software_fixture()
    capacity["selected"][0]["full_prompt_tokens"] = 130944
    assert runner.validate_capacity_selection(capacity, "software-fixture-model")
    with pytest.raises(ValueError, match="model/window"):
        runner.validate_capacity_selection(capacity, "different-software-fixture-model")


def test_qasper_public_handoff_is_original_three_not_full_suite_or_raw_diagnostic():
    directory = ROOT / "docs/workload-data/validation"
    receipt = json.loads(
        (directory / "longbench-v1-qasper-public-handoff-npu-20261023.json").read_text(
            encoding="utf-8"
        )
    )
    raw = json.loads(
        (directory / "longbench-v1-qasper-raw-diagnostic-npu-20261023.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "END_TO_END_EXECUTED_AND_SCORED"
    assert receipt["unique_original_tasks"] == 3
    assert receipt["full_suite_claim"] is receipt["performance_claim"] is False
    assert receipt["official_f1_percent"] == 22.52
    assert [s["preflight_prompt_tokens"] for s in receipt["samples"]] == [
        4103,
        3440,
        4461,
    ]
    assert all(
        s["finish_reason"] == "stop"
        and s["usage"]["prompt_tokens"] == s["preflight_prompt_tokens"]
        for s in receipt["samples"]
    )
    assert (
        receipt["generation_config"]["max_tokens"]
        == raw["generation_config"]["max_tokens"]
        == 128
    )
    assert receipt["selected_ids"] == raw["selected_ids"]
    assert [s["prompt_sha256"] for s in receipt["samples"]] == [
        s["prompt_sha256"] for s in raw["samples"]
    ]
    assert raw["source_checkbox_eligible"] is False
    assert all(s["finish_reason"] == "length" for s in raw["samples"])
    assert (
        receipt["helper_sha256"]
        == (
            "333174542868a42ac1eaed7ad180cc0db8bac2deaab5cb013ea5e67ab6c49ee2"  # pragma: allowlist secret - historical executed helper checksum
        )
    )


def test_qasper_range_full_keep_original_positions_and_fail_closed():
    spec = importlib.util.spec_from_file_location(
        "longbench_qasper_runner", ROOT / "scripts/run_longbench_v1_qasper_original.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rows = [
        {"id": index, "context": "PRIVATE_SOFTWARE_FIXTURE"} for index in range(200)
    ]
    selected = runner.select_original_records(rows, start=197, count=3)
    assert [index for index, _ in selected] == [197, 198, 199]
    assert all(row is rows[index] for index, row in selected)
    assert runner.select_original_records(rows) == list(enumerate(rows[:3]))
    assert runner.select_original_records(rows, full=True) == list(enumerate(rows))
    for start, count in [(-1, 3), (0, 0), (199, 2)]:
        with pytest.raises(ValueError, match="range"):
            runner.select_original_records(rows, start=start, count=count)
    with pytest.raises(ValueError, match="full"):
        runner.select_original_records(rows, start=3, full=True)


def test_qasper_cpu_audit_does_not_claim_new_native_execution():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/longbench-qasper-cpu-audit-20261024.json"
        ).read_text("utf-8")
    )
    assert receipt["original_rows_and_reference_fields_equal"] == 200
    assert receipt["prepared_original_rows"] == 200
    assert receipt["prepare_resolve_roundtrips"] == 6
    assert receipt["saved_native_tasks_reused"] == 3
    assert receipt["original_prompts_and_budget_equal"]
    assert receipt["official_saved_prediction_rescore_equal"]
    assert receipt["audit_model_calls"] == receipt["new_native_tasks"] == 0
    assert not receipt["new_range_runner_native_validated"]
    assert not receipt["full_qasper_native_completed"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.com/v1",
        "http://127.0.0.1:19466",
        "http://127.0.0.1:19466/v1/chat/completions",
        "http://127.0.0.1:19466/v1?unexpected=1",
        "http://localhost:19466/other/v1",
    ],
)
def test_external_endpoint_rejected_before_loading_original_data(tmp_path, endpoint):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_longbench_v2_original_samples.py"),
            "--data-file",
            str(tmp_path / "missing.json"),
            "--official-archive",
            str(tmp_path / "missing.zip"),
            "--capacity-selection",
            str(tmp_path / "selection.json"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--output",
            str(tmp_path / "out"),
            "--endpoint",
            endpoint,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "only local HTTP NPU endpoints" in result.stderr
    assert not (tmp_path / "out").exists()


def test_help_without_optional_official_environment():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_longbench_v2_original_samples.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--capacity-selection" in result.stdout


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
@pytest.mark.parametrize("suffix", ["/v1", "/v1/"])
def test_valid_local_base_url_reaches_data_identity_check(tmp_path, host, suffix):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_longbench_v2_original_samples.py"),
            "--data-file",
            str(tmp_path / "missing-original.json"),
            "--official-archive",
            str(tmp_path / "missing.zip"),
            "--capacity-selection",
            str(tmp_path / "selection.json"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--output",
            str(tmp_path / "out"),
            "--endpoint",
            f"http://{host}:19466{suffix}",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "FileNotFoundError" in result.stderr
    assert "missing-original.json" in result.stderr
    assert "only local HTTP NPU endpoints" not in result.stderr
    assert not (tmp_path / "out").exists()


def test_failed_quality_and_budget_are_not_erased():
    r = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/longbench-v2-original-attempt-npu-20261022.json"
        ).read_text("utf-8")
    )
    assert r["correct_tasks"] == 0 and r["unique_tasks"] == 3
    assert r["source_checkbox_eligible"] is False
    assert [task["length_label"] for task in r["tasks"]] == ["short"] * 3
    assert [request["finish_reason"] for request in r["requests"]].count("length") == 1
    assert all(request["full_input_verified"] for request in r["requests"])


def test_public_handoff_is_same_tasks_not_full_suite_or_quality_success():
    directory = ROOT / "docs/workload-data/validation"
    initial = json.loads(
        (directory / "longbench-v2-original-attempt-npu-20261022.json").read_text(
            "utf-8"
        )
    )
    handoff = json.loads(
        (directory / "longbench-v2-public-handoff-npu-20261022.json").read_text("utf-8")
    )
    assert handoff["selected_samples"] == initial["selected_samples"]
    assert handoff["full_prompt_tokens"] == [
        r["full_prompt_tokens"] for r in initial["requests"]
    ]
    assert handoff["finish_reasons"] == ["stop", "stop", "length"]
    assert handoff["new_unique_tasks"] == 0 and handoff["correct_tasks"] == 0
    assert handoff["source_checkbox_eligible"] is False
    assert handoff["full_suite_claim"] is False
    assert handoff["process_returncode"] == 0
    assert (
        handoff["helper_sha256"]
        == "eb3fe8597204234ece99a1d7bd5f3e671ce89e735d84840d9f2831bdad80114d"  # pragma: allowlist secret - historical executed helper checksum
    )
