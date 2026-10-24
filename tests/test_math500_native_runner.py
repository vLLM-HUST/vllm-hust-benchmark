"""CPU-only contract tests for the pinned upstream MATH-500 Inspect entrypoint."""

import json
import hashlib
from pathlib import Path

import pytest

from scripts.run_math500_native import (
    INSPECT_TASK,
    REPOSITORY_ROOT,
    build_command,
    require_external_path,
    require_inspect_version,
    validate_base_url,
)


def test_default_command_runs_fixed_first_three_with_pinned_upstream_task(
    tmp_path: Path,
):
    command = build_command(
        inspect_bin="inspect",
        model_id="Qwen/Qwen3.5-27B",
        base_url="http://127.0.0.1:19466/v1/",
        log_dir=tmp_path / "inspect-logs",
    )

    assert command[:3] == ["inspect", "eval", INSPECT_TASK]
    assert command[command.index("--model") + 1] == "vllm/Qwen/Qwen3.5-27B"
    assert command[command.index("--model-base-url") + 1] == "http://127.0.0.1:19466/v1"
    assert command[command.index("--limit") + 1] == "3"
    assert command[command.index("--max-retries") + 1] == "0"
    assert "--retry-on-error" not in command
    assert "--epochs" not in command
    assert "--epochs-reducer" not in command


def test_full_command_preserves_upstream_epoch_and_scorer_config_without_sample_limit(
    tmp_path: Path,
):
    command = build_command(
        inspect_bin="inspect",
        model_id="served-qwen",
        base_url="http://127.0.0.1:8000/v1",
        log_dir=tmp_path / "inspect-logs",
        full=True,
    )

    assert command[:3] == ["inspect", "eval", INSPECT_TASK]
    assert "--limit" not in command
    assert "--epochs" not in command
    assert "--epochs-reducer" not in command


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8000",
        "file:///v1",
        "http://user:secret@localhost:8000/v1",  # pragma: allowlist secret - invalid URL fixture, not credentials
        "http://localhost:8000/v1?token=secret",
    ],
)
def test_rejects_non_v1_or_credential_bearing_endpoint(base_url: str):
    with pytest.raises(ValueError, match="/v1 endpoint"):
        validate_base_url(base_url)


def test_rejects_hf_cache_and_logs_inside_repository(tmp_path: Path):
    with pytest.raises(ValueError, match="outside the repository"):
        require_external_path(REPOSITORY_ROOT / "dataset-cache", label="HF_HOME")
    with pytest.raises(ValueError, match="outside the repository"):
        require_external_path(
            REPOSITORY_ROOT / "private-logs", label="Inspect log directory"
        )


def test_requires_the_pinned_inspect_release(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "scripts.run_math500_native.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="0.3.283\n"),
    )
    require_inspect_version("inspect")

    monkeypatch.setattr(
        "scripts.run_math500_native.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="0.3.284\n"),
    )
    with pytest.raises(RuntimeError, match="inspect-ai==0.3.283"):
        require_inspect_version("inspect")


def test_delivery_requires_native_inspect_and_fixed_artifact_receipts():
    docs = REPOSITORY_ROOT / "docs/workload-data"
    ledger = json.loads((docs / "dataset-validation-status.json").read_text("utf-8"))
    item = ledger["delivery_inventory"]["MATH500"]
    guide = (docs / "MATH500_ORIGINAL_TASK.zh-CN.md").read_text("utf-8")
    checklist = (docs / "DATASET_CHECKLIST.zh-CN.md").read_text("utf-8")

    assert item["status"] == "AVAILABLE"
    assert item["checked"] is True
    delivery = json.loads(
        (REPOSITORY_ROOT / item["delivery_receipt"]).read_text("utf-8")
    )
    assert delivery["source_checkbox_eligible"] is True
    assert delivery["verified_sample_epochs"] == 12
    assert not delivery["full_suite_validated"]
    assert delivery["scoring"].startswith("Original model_graded_fact")
    artifact = json.loads(
        (docs / "validation" / delivery["installed_ci_reader_receipt"]).read_text(
            "utf-8"
        )
    )
    assert artifact["source_commit"] == delivery["source_commit"]
    assert artifact["original_rows_field_checked"] == 500
    assert artifact["original_sample_roundtrips"] == 3
    assert artifact["model_calls"] == 0
    assert INSPECT_TASK in guide
    assert "model_graded_fact" in guide
    assert "不冒充H4官方分数" in checklist


def test_native_receipt_covers_three_original_tasks_and_all_four_epochs():
    receipt = json.loads(
        (
            REPOSITORY_ROOT
            / "docs/workload-data/validation/math500-upstream-inspect-native-20261024.json"
        ).read_text("utf-8")
    )
    assert receipt["upstream_task"] == INSPECT_TASK
    assert receipt["unique_original_tasks"] == 3
    assert receipt["sample_epochs_completed"] == len(receipt["samples"]) == 12
    assert receipt["scorer"] == "model_graded_fact"
    assert receipt["epoch_reducer"] == "pass_at_1"
    assert receipt["task_model_self_grading"] is True
    assert not receipt["source_checkbox_eligible"]
    assert not receipt["full_suite_validated"]
    assert not receipt["raw_data_published"]
    for index in [0, 1, 2]:
        samples = [
            sample for sample in receipt["samples"] if sample["original_index"] == index
        ]
        assert {sample["epoch"] for sample in samples} == {1, 2, 3, 4}
        assert all(
            sample["stop_reason"] == "stop" and sample["scores"] for sample in samples
        )
    script = REPOSITORY_ROOT / receipt["wrapper"]
    assert (
        hashlib.sha256(script.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == receipt["formatted_public_wrapper_sha256"]
    )
    assert receipt["formatting_only_ast_equivalent"] is True
    assert receipt["wrapper_sha256"] != receipt["formatted_public_wrapper_sha256"]
