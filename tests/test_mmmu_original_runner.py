"""CPU fixtures cannot establish real MMMU native completion."""

import base64
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
runner = importlib.import_module("run_mmmu_original")
source = importlib.import_module("prepare_mmmu_author_source")
auditor = importlib.import_module("verify_mmmu_original_saved")


def test_original_multimodal_order_and_repeated_images():
    first = b"\x89PNG\r\n\x1a\noriginal-one"
    second = b"\xff\xd8original-two"
    result = runner.image_messages(
        "A<image 2>B<image 1>C<image 2>D",
        {"image_1": {"bytes": first}, "image_2": {"bytes": second}},
    )[0]["content"]
    assert [p["type"] for p in result] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert result[0]["text"] == "A" and result[-1]["text"] == "D"
    assert base64.b64decode(result[1]["image_url"]["url"].split(",")[1]) == second
    assert base64.b64decode(result[3]["image_url"]["url"].split(",")[1]) == first


@pytest.mark.parametrize(
    "prompt,row",
    [
        ("<image 1>", {}),
        ("text", {"image_1": {"bytes": b"unused"}}),
        ("<image 1>", {"image_1": {"path": "/secret"}}),
    ],
)
def test_no_image_dropping_or_path_substitution(prompt, row):
    with pytest.raises(ValueError):
        runner.image_messages(prompt, row)


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_actual_usage_retains_length_and_wrong_answers(finish):
    result = {
        "choices": [{"finish_reason": finish, "message": {"content": "wrong answer"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 128, "total_tokens": 138},
    }
    assert runner.validate_response(result, 128) == "wrong answer"
    result["usage"]["total_tokens"] = 0
    with pytest.raises(ValueError):
        runner.validate_response(result, 128)


def test_default_plan_no_source_or_output_access(monkeypatch, tmp_path, capsys):
    output = tmp_path / "missing"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--official-checkout",
            "absent",
            "--data-root",
            "absent",
            "--output",
            str(output),
        ],
    )
    runner.main()
    assert not output.exists()
    assert '"model_calls": 0' in capsys.readouterr().out


def test_source_slug_valid_for_archive_transport():
    assert source.REPOSITORY == "MMMU-Benchmark/MMMU"
    assert not source.REPOSITORY.startswith("https:")


def test_saved_audit_plan_never_calls_endpoint(monkeypatch, tmp_path, capsys):
    output = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--official-checkout",
            "absent",
            "--data-root",
            "absent",
            "--run",
            "absent",
            "--output",
            str(output),
        ],
    )
    auditor.main()
    assert not output.exists()
    assert '"new_model_calls": 0' in capsys.readouterr().out


def test_source_plan_no_download(monkeypatch, tmp_path, capsys):
    output = tmp_path / "source"
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(output)])
    monkeypatch.setattr(
        source,
        "acquire_archive",
        lambda *a, **kw: pytest.fail("No acquisition in plan"),
    )
    source.main()
    assert not output.exists()
    assert "PLAN_ONLY_NO_DOWNLOAD" in capsys.readouterr().out


def test_current_delivery_requires_native_and_provider_evidence():
    root = Path(__file__).resolve().parents[1]
    ledger = json.loads(
        (root / "docs/workload-data/dataset-validation-status.json").read_text(
            encoding="utf-8"
        )
    )["delivery_inventory"]["MMMU"]
    receipt = json.loads(
        (root / ledger["delivery_receipt"]).read_text(encoding="utf-8")
    )
    assert ledger["checked"] and receipt["source_checkbox_eligible"]
    assert receipt["source"]["original_files"] == 92
    assert receipt["source"]["original_rows"] == 11550
    native = receipt["native_validation"]
    assert native["tasks"] == 90 and native["multi_image_tasks"] == 4
    assert sum(native["finish_counts"].values()) == 90
    assert native["finish_counts"]["length"] == 59
    assert (
        native["native_entry_exit_code"]
        == native["saved_result_public_audit_exit_code"]
        == 0
    )
    assert native["author_prompt_and_scorer_unmodified"]
    assert not native["full_validation_or_quality_pass_claim"]
    provider = receipt["provider_and_research"]
    assert (
        provider["all_original_rows_compared"]
        == provider["all_original_index_rows_verified"]
        == 11550
    )
    assert provider["public_and_fixed_research_roundtrips"] == 152
    assert provider["all90_native_task_ids_in_roundtrips"]
    assert provider["installed_payloads_byte_equal"] == 44
    assert not provider["observations_or_research_analysis_migrated"]
