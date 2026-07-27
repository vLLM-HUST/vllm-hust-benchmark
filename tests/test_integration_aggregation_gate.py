"""Tests for the submission admission gate in ``aggregate_to_website``.

The gate rejects submission directories that are FAILED (STATUS file starts
with ``FAILED``), NO_STATUS (missing or empty STATUS file), or temporary
(directory name matches ``tmp|temp|wip|scratch|adhoc`` patterns) before they
enter the formal aggregation pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from vllm_hust_benchmark.integration import (
    RepoLayout,
    _find_superseded_coexistence_conflicts,
    _scan_submission_admission_failures,
    aggregate_to_website,
)


def _write_mock_submission(
    target_dir: Path,
    *,
    entry_id: str,
    submitted_at: str,
    repeat_group: str | None = None,
    repeat_index: int | None = None,
    supersedes: str | list[str] | None = None,
    canonical_id: str = "hf:Qwen/Qwen2.5-14B-Instruct",
    chip_model: str = "910B2",
    precision: str = "FP16",
    workload_name: str = "random-online",
    chip_count: int = 1,
    config_type: str = "single_gpu",
    engine: str = "vllm-hust",
    engine_version: str = "0.18.0.post1",
) -> None:
    """Write a minimal ``run_leaderboard.json`` with the given fields.

    The fields are chosen to match what ``build_series_signature`` reads:
    ``model.canonical_id``, ``hardware.chip_model``, ``model.precision``,
    ``workload.name``, ``hardware.chip_count``, ``config_type``, ``engine``,
    ``engine_version``.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "entry_id": entry_id,
        "engine": engine,
        "engine_version": engine_version,
        "config_type": config_type,
        "hardware": {"chip_model": chip_model, "chip_count": chip_count},
        "model": {"canonical_id": canonical_id, "precision": precision},
        "workload": {"name": workload_name},
        "metadata": {"submitted_at": submitted_at},
    }
    if repeat_group is not None:
        payload["metadata"]["repeat_group"] = repeat_group
    if repeat_index is not None:
        payload["metadata"]["repeat_index"] = repeat_index
    if supersedes is not None:
        payload["metadata"]["supersedes"] = supersedes
    (target_dir / "run_leaderboard.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def test_failed_status_rejected(tmp_path: Path) -> None:
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    failed_dir = source_dir / "failed-run"
    failed_dir.mkdir()
    (failed_dir / "STATUS").write_text("FAILED: server crashed\n", encoding="utf-8")

    failures = _scan_submission_admission_failures(source_dir)

    assert len(failures) == 1
    assert failures[0]["reason"] == "FAILED"
    assert "failed-run" in failures[0]["dir"]
    assert "server crashed" in failures[0]["detail"]


def test_no_status_rejected(tmp_path: Path) -> None:
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    missing_dir = source_dir / "missing-status-run"
    missing_dir.mkdir()
    # Neither a STATUS file nor a run_leaderboard.json → truly incomplete.
    assert not (missing_dir / "run_leaderboard.json").is_file()

    failures = _scan_submission_admission_failures(source_dir)

    assert len(failures) == 1
    assert failures[0]["reason"] == "NO_STATUS"
    assert "missing-status-run" in failures[0]["dir"]


def test_backfill_dir_without_status_passes(tmp_path: Path) -> None:
    """Backfill/baseline pipelines don't write a STATUS file by design.

    A directory without a STATUS file but WITH a valid ``run_leaderboard.json``
    must NOT be flagged as NO_STATUS — it's a valid backfill/baseline submission.
    """
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    backfill_dir = source_dir / "historical-pr-some-run"
    backfill_dir.mkdir()
    (backfill_dir / "run_leaderboard.json").write_text(
        '{"entry_id": "test"}\n', encoding="utf-8"
    )

    failures = _scan_submission_admission_failures(source_dir)

    assert failures == []


def test_empty_status_rejected(tmp_path: Path) -> None:
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    empty_dir = source_dir / "empty-status-run"
    empty_dir.mkdir()
    (empty_dir / "STATUS").write_text("   \n  \t\n", encoding="utf-8")

    failures = _scan_submission_admission_failures(source_dir)

    assert len(failures) == 1
    assert failures[0]["reason"] == "NO_STATUS"
    assert "empty-status-run" in failures[0]["dir"]


def test_temporary_directory_rejected(tmp_path: Path) -> None:
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    (source_dir / "tmp-prefix-recheck").mkdir()

    failures = _scan_submission_admission_failures(source_dir)

    assert len(failures) == 1
    assert failures[0]["reason"] == "temporary"
    assert "tmp-prefix-recheck" in failures[0]["dir"]


def test_clean_directory_passes(tmp_path: Path) -> None:
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    clean_dir = source_dir / "clean-run"
    clean_dir.mkdir()
    (clean_dir / "STATUS").write_text("OK\n", encoding="utf-8")

    failures = _scan_submission_admission_failures(source_dir)

    assert failures == []


def test_aggregate_to_website_returns_2_on_admission_failure(
    capsys, tmp_path: Path
) -> None:
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=website_repo,
    )

    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    failed_dir = source_dir / "failed-run"
    failed_dir.mkdir()
    (failed_dir / "STATUS").write_text("FAILED: server crashed\n", encoding="utf-8")

    exit_code = aggregate_to_website(
        layout=layout,
        source_dir=source_dir,
        execute=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "admission gate rejected" in captured.err
    assert "FAILED" in captured.err
    assert "aggregate_results.py" not in captured.out


def test_signature_conflict_rejected(tmp_path: Path) -> None:
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()

    _write_mock_submission(
        source_dir / "old-run",
        entry_id="old-entry-1",
        submitted_at="2026-07-20T00:00:00Z",
    )
    _write_mock_submission(
        source_dir / "new-run",
        entry_id="new-entry-2",
        submitted_at="2026-07-22T00:00:00Z",
    )

    conflicts = _find_superseded_coexistence_conflicts(source_dir)

    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["old_entry_id"] == "old-entry-1"
    assert conflict["new_entry_id"] == "new-entry-2"
    assert conflict["old_dir"].endswith("old-run")
    assert conflict["new_dir"].endswith("new-run")
    assert conflict["signature"]


def test_signature_conflict_resolved_by_supersedes(tmp_path: Path) -> None:
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()

    _write_mock_submission(
        source_dir / "old-run",
        entry_id="old-entry-1",
        submitted_at="2026-07-20T00:00:00Z",
    )
    _write_mock_submission(
        source_dir / "new-run",
        entry_id="new-entry-2",
        submitted_at="2026-07-22T00:00:00Z",
        supersedes="old-entry-1",
    )

    conflicts = _find_superseded_coexistence_conflicts(source_dir)

    assert conflicts == []


def test_repeat_group_not_conflict(tmp_path: Path) -> None:
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()

    _write_mock_submission(
        source_dir / "repeat-run-0",
        entry_id="repeat-entry-1",
        submitted_at="2026-07-20T00:00:00Z",
        repeat_group="campaign-alpha",
        repeat_index=0,
    )
    _write_mock_submission(
        source_dir / "repeat-run-1",
        entry_id="repeat-entry-2",
        submitted_at="2026-07-22T00:00:00Z",
        repeat_group="campaign-alpha",
        repeat_index=1,
    )

    conflicts = _find_superseded_coexistence_conflicts(source_dir)

    assert conflicts == []


def test_different_signatures_not_conflict(tmp_path: Path) -> None:
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()

    _write_mock_submission(
        source_dir / "run-a",
        entry_id="entry-a",
        submitted_at="2026-07-20T00:00:00Z",
        workload_name="random-online",
    )
    _write_mock_submission(
        source_dir / "run-b",
        entry_id="entry-b",
        submitted_at="2026-07-22T00:00:00Z",
        workload_name="sharegpt-online",
    )

    conflicts = _find_superseded_coexistence_conflicts(source_dir)

    assert conflicts == []
