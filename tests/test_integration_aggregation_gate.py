"""Tests for the submission admission gate in ``aggregate_to_website``.

The gate rejects submission directories that are FAILED (STATUS file starts
with ``FAILED``), NO_STATUS (missing or empty STATUS file), or temporary
(directory name matches ``tmp|temp|wip|scratch|adhoc`` patterns) before they
enter the formal aggregation pipeline.
"""

from __future__ import annotations

from pathlib import Path

from vllm_hust_benchmark.integration import (
    RepoLayout,
    _scan_submission_admission_failures,
    aggregate_to_website,
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
    (source_dir / "missing-status-run").mkdir()

    failures = _scan_submission_admission_failures(source_dir)

    assert len(failures) == 1
    assert failures[0]["reason"] == "NO_STATUS"
    assert "missing-status-run" in failures[0]["dir"]


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
