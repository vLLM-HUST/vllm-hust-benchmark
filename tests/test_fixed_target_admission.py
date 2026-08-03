"""Tests for the fixed-target admission gate in ``aggregate_to_website``.

The gate scans leaderboard snapshot entries against the fixed-target
registry and quarantines entries that misalign with active profiles:
missing required server parameters, config drift, retired targets, or
specialty profiles without an explicit contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from vllm_hust_benchmark.fixed_target_registry import FixedTargetProfile
from vllm_hust_benchmark.integration import (
    _quarantine_misaligned_snapshot_entries,
    _scan_fixed_target_misaligned_entries,
)


def _make_entry(
    *,
    entry_id: str = "test-entry-001",
    model_repo_id: str = "Qwen/Qwen2.5-14B-Instruct",
    model_precision: str = "FP16",
    chip_model: str = "910B2",
    chip_count: int = 1,
    workload_name: str = "random-online",
    gpu_memory_utilization: float | None = 0.6,
    max_model_len: int | None = 32768,
    submitted_at: str = "2026-07-25T00:00:00Z",
) -> dict:
    """Build a test entry. Pass ``None`` to omit a field."""
    server_params: dict = {}
    if gpu_memory_utilization is not None:
        server_params["gpu_memory_utilization"] = gpu_memory_utilization
    if max_model_len is not None:
        server_params["max_model_len"] = max_model_len
    return {
        "entry_id": entry_id,
        "engine": "vllm-hust",
        "model": {
            "repo_id": model_repo_id,
            "canonical_id": f"hf:{model_repo_id}",
            "precision": model_precision,
        },
        "hardware": {
            "chip_model": chip_model,
            "chip_count": chip_count,
        },
        "workload": {
            "name": workload_name,
            "input_length": 1024,
            "output_length": 256,
        },
        "same_spec": {
            "spec_id": "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2",
            "scenario": workload_name,
            "resolved_server_parameters": server_params,
        },
        "metadata": {
            "submitted_at": submitted_at,
        },
    }


def _write_snapshot(tmp_path: Path, entries: list[dict]) -> None:
    """Write entries to ``leaderboard_single.json`` under ``tmp_path``."""
    snapshot_path = tmp_path / "leaderboard_single.json"
    snapshot_path.write_text(json.dumps(entries), encoding="utf-8")


def test_missing_gpu_memory_utilization_quarantined(tmp_path: Path) -> None:
    entry = _make_entry(gpu_memory_utilization=None)
    _write_snapshot(tmp_path, [entry])

    misaligned = _scan_fixed_target_misaligned_entries(tmp_path)

    assert len(misaligned) == 1
    assert misaligned[0]["entry_id"] == "test-entry-001"
    assert misaligned[0]["disposition"] == "quarantine"
    assert "missing_gpu_memory_utilization" in misaligned[0]["reason"]


def test_config_drift_quarantined(tmp_path: Path) -> None:
    entry = _make_entry(gpu_memory_utilization=0.9)
    _write_snapshot(tmp_path, [entry])

    misaligned = _scan_fixed_target_misaligned_entries(tmp_path)

    assert len(misaligned) == 1
    assert misaligned[0]["disposition"] == "quarantine"
    assert misaligned[0]["reason"] == "config_drift"


def test_wrong_profile_quarantined(tmp_path: Path) -> None:
    """Vision model entry with stale max_model_len (30720) instead of
    the vision default (32768) triggers config_drift."""
    entry = _make_entry(
        model_repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        workload_name="visionarena-online",
        max_model_len=30720,
    )
    _write_snapshot(tmp_path, [entry])

    misaligned = _scan_fixed_target_misaligned_entries(tmp_path)

    assert len(misaligned) == 1
    assert misaligned[0]["disposition"] == "quarantine"
    assert "config_drift" in misaligned[0]["reason"]


def test_specialty_without_contract(tmp_path: Path) -> None:
    """Multi-chip entry matching a specialty profile is flagged."""
    entry = _make_entry(chip_count=2)
    _write_snapshot(tmp_path, [entry])

    misaligned = _scan_fixed_target_misaligned_entries(tmp_path)

    assert len(misaligned) == 1
    assert misaligned[0]["disposition"] == "specialty"
    assert misaligned[0]["reason"] == "specialty_without_contract"


def test_aligned_entry_kept(tmp_path: Path) -> None:
    """A fully-aligned entry must not appear in the misaligned list."""
    entry = _make_entry()
    _write_snapshot(tmp_path, [entry])

    misaligned = _scan_fixed_target_misaligned_entries(tmp_path)

    assert len(misaligned) == 0


def test_retired_target_quarantined(tmp_path: Path) -> None:
    """An entry matching a retired profile is quarantined."""
    retired_profile = FixedTargetProfile(
        target_id="official-ascend-jan-2026-v0.18.0",
        target_version="v1",
        profile_name="retired-test-profile",
        model="Qwen/Qwen2.5-14B-Instruct",
        hardware_chip_model="910B2",
        chip_count=1,
        model_precision="FP16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        max_model_len=32768,
        workload_name="random-online",
        status="retired",
    )

    entry = _make_entry()
    _write_snapshot(tmp_path, [entry])

    misaligned = _scan_fixed_target_misaligned_entries(
        tmp_path, registry=(retired_profile,)
    )

    assert len(misaligned) == 1
    assert misaligned[0]["disposition"] == "quarantine"
    assert misaligned[0]["reason"] == "retired_target"


def test_historical_legacy_not_bypassed(tmp_path: Path) -> None:
    """A historical entry (submitted_at before the contract activation) with
    missing server parameters is still quarantined — the fixed-target gate
    does not grandfather legacy entries."""
    entry = _make_entry(
        submitted_at="2026-07-02T10:06:46Z",
        gpu_memory_utilization=None,
    )
    _write_snapshot(tmp_path, [entry])

    misaligned = _scan_fixed_target_misaligned_entries(tmp_path)

    assert len(misaligned) == 1
    assert misaligned[0]["disposition"] == "quarantine"


def test_quarantine_misaligned_snapshot_entries(tmp_path: Path) -> None:
    """``_quarantine_misaligned_snapshot_entries`` removes misaligned entries
    from the snapshot file in-place, leaving aligned entries untouched."""
    entry_aligned = _make_entry(entry_id="aligned-001")
    entry_misaligned = _make_entry(
        entry_id="misaligned-001", gpu_memory_utilization=None
    )
    _write_snapshot(tmp_path, [entry_aligned, entry_misaligned])

    misaligned = _scan_fixed_target_misaligned_entries(tmp_path)
    assert len(misaligned) == 1

    _quarantine_misaligned_snapshot_entries(tmp_path, misaligned)

    remaining = json.loads(
        (tmp_path / "leaderboard_single.json").read_text(encoding="utf-8")
    )
    assert len(remaining) == 1
    assert remaining[0]["entry_id"] == "aligned-001"
