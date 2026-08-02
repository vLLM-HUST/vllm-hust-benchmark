"""Tests for the submission admission gate in ``aggregate_to_website``.

The gate rejects failed, incomplete, temporary, or malformed submission
directories before they enter the formal aggregation pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.integration import (
    RepoLayout,
    _build_rejected_superseded_report,
    _find_superseded_coexistence_conflicts,
    _scan_submission_admission_failures,
    _write_rejected_superseded_report,
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
    supersedes_reason: str | None = None,
    canonical_id: str = "hf:Qwen/Qwen2.5-14B-Instruct",
    chip_model: str = "910B2",
    precision: str = "FP16",
    workload_name: str = "random-online",
    chip_count: int = 1,
    config_type: str = "single_gpu",
    engine: str = "vllm-hust",
    engine_version: str = "0.18.0.post1",
    engine_commit: str | None = None,
    plugin_commit: str | None = None,
) -> None:
    """Write a minimal ``run_leaderboard.json`` with the given fields.

    The fields are chosen to match what ``build_series_signature`` reads:
    ``model.canonical_id``, ``hardware.chip_model``, ``model.precision``,
    ``workload.name``, ``hardware.chip_count``, ``config_type``, ``engine``,
    ``engine_version``. ``engine_commit`` / ``plugin_commit`` populate
    ``metadata.runtime_provenance`` so the secondary code-combo grouping in
    ``_find_superseded_coexistence_conflicts`` can be exercised.
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
    if supersedes_reason is not None:
        payload["metadata"]["supersedes_reason"] = supersedes_reason
    if engine_commit is not None or plugin_commit is not None:
        payload["metadata"]["runtime_provenance"] = {
            "engine": {"commit": engine_commit or ""},
            "plugin": {"commit": plugin_commit or ""},
        }
    (target_dir / "run_leaderboard.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_manifest(target_dir)


def _write_manifest(
    target_dir: Path, artifact_name: str = "run_leaderboard.json"
) -> None:
    manifest = {
        "schema_version": "leaderboard-export-manifest/v2",
        "generated_at": "2026-08-01T00:00:00Z",
        "entries": [{"leaderboard_artifact": artifact_name}],
    }
    (target_dir / "leaderboard_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
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
    _write_manifest(backfill_dir)

    failures = _scan_submission_admission_failures(source_dir)

    assert failures == []


def test_ci_dir_without_status_rejected_even_with_artifact(tmp_path: Path) -> None:
    """CI output is incomplete until the runner records a successful status."""
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    ci_dir = source_dir / "ci-30554037879-1-7363d82b"
    ci_dir.mkdir()
    (ci_dir / "run_leaderboard.json").write_text(
        '{"entry_id": "test"}\n', encoding="utf-8"
    )

    failures = _scan_submission_admission_failures(source_dir)

    assert len(failures) == 1
    assert failures[0]["reason"] == "NO_STATUS"
    assert "CI publication" in failures[0]["detail"]


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
    (clean_dir / "run_leaderboard.json").write_text(
        '{"entry_id": "test"}\n', encoding="utf-8"
    )
    _write_manifest(clean_dir)

    failures = _scan_submission_admission_failures(source_dir)

    assert failures == []


@pytest.mark.parametrize("status", ["BLOCKED", "CANCELLED", "RUNNING"])
def test_non_publishable_status_rejected(tmp_path: Path, status: str) -> None:
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    run_dir = source_dir / "incomplete-run"
    run_dir.mkdir()
    (run_dir / "STATUS").write_text(status + "\n", encoding="utf-8")

    failures = _scan_submission_admission_failures(source_dir)

    assert len(failures) == 1
    assert failures[0]["reason"] == "RUN_STATUS"
    assert status in failures[0]["detail"]


@pytest.mark.parametrize(
    ("artifact", "manifest", "expected_reason"),
    [
        (False, False, "MISSING_ARTIFACT"),
        (True, False, "MISSING_MANIFEST"),
        (True, True, "INVALID_MANIFEST"),
    ],
)
def test_incomplete_artifact_pair_rejected(
    tmp_path: Path,
    artifact: bool,
    manifest: bool,
    expected_reason: str,
) -> None:
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    run_dir = source_dir / "candidate-run"
    run_dir.mkdir()
    (run_dir / "STATUS").write_text("OK\n", encoding="utf-8")
    if artifact:
        (run_dir / "run_leaderboard.json").write_text(
            '{"entry_id": "test"}\n', encoding="utf-8"
        )
    if manifest:
        _write_manifest(run_dir, artifact_name="different.json")

    failures = _scan_submission_admission_failures(source_dir)

    assert len(failures) == 1
    assert failures[0]["reason"] == expected_reason


def _write_backfill_submission(
    target_dir: Path,
    *,
    entry_id: str,
    git_vllm_hust: str,
    git_vllm_ascend_hust: str,
) -> None:
    """Write a historical-pr-backfill submission dir with env-manifest."""
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "STATUS").write_text("OK\n", encoding="utf-8")
    (target_dir / "run_leaderboard.json").write_text(
        json.dumps(
            {
                "entry_id": entry_id,
                "metadata": {
                    "data_source": "real-online-historical-pr-backfill",
                },
            }
        ),
        encoding="utf-8",
    )
    (target_dir / "leaderboard_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "leaderboard-export-manifest/v2",
                "entries": [
                    {
                        "idempotency_key": "test",
                        "leaderboard_artifact": "run_leaderboard.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (target_dir / "env-manifest.json").write_text(
        json.dumps(
            {
                "git_info": {
                    "vllm_hust": git_vllm_hust,
                    "vllm_ascend_hust": git_vllm_ascend_hust,
                }
            }
        ),
        encoding="utf-8",
    )


def test_backfill_with_not_available_git_info_rejected(tmp_path: Path) -> None:
    """Historical-pr-backfill submissions with ``not available`` git_info in
    env-manifest.json must be rejected by the admission gate, not silently
    admitted as structurally valid.
    """
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    _write_backfill_submission(
        source_dir / "historical-pr-pr99-base",
        entry_id="test-99",
        git_vllm_hust="not available",
        git_vllm_ascend_hust="not available",
    )

    failures = _scan_submission_admission_failures(source_dir)

    assert len(failures) == 1
    assert failures[0]["reason"] == "PROVENANCE_INCOMPLETE"
    assert "not available" in failures[0]["detail"]


def test_backfill_with_real_git_info_passes(tmp_path: Path) -> None:
    """Historical-pr-backfill submissions with real git commit provenance
    in env-manifest.json pass the admission gate.
    """
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    _write_backfill_submission(
        source_dir / "historical-pr-pr100-base",
        entry_id="test-100",
        git_vllm_hust="vllm-hust-test-commit",
        git_vllm_ascend_hust="ascend-hust-test-commit",
    )

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


def test_different_plugin_commit_not_conflict(tmp_path: Path) -> None:
    """Same signature + same engine commit but DIFFERENT plugin commits are
    independent PR comparison runs (e.g. PR#66 vs PR#70 vs PR#77 each
    testing a different plugin commit) and may coexist without supersedes.
    """
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()

    _write_mock_submission(
        source_dir / "historical-pr-pr77",
        entry_id="entry-pr77",
        submitted_at="2026-07-02T10:06:46Z",
        engine_commit="ceec19abb0",
        plugin_commit="51e577b17b",
    )
    _write_mock_submission(
        source_dir / "historical-pr-pr66",
        entry_id="entry-pr66",
        submitted_at="2026-07-01T14:51:46Z",
        engine_commit="ceec19abb0",
        plugin_commit="e0686f12d1",
    )
    _write_mock_submission(
        source_dir / "historical-pr-pr70",
        entry_id="entry-pr70",
        submitted_at="2026-07-01T14:54:33Z",
        engine_commit="ceec19abb0",
        plugin_commit="312ca80a90",
    )

    conflicts = _find_superseded_coexistence_conflicts(source_dir)

    assert conflicts == []


def test_different_engine_commit_not_conflict(tmp_path: Path) -> None:
    """Same signature + same plugin commit but DIFFERENT engine commits are
    independent comparison runs (different vLLM-HUST engine commits) and
    may coexist without supersedes.
    """
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()

    _write_mock_submission(
        source_dir / "engine-a",
        entry_id="entry-engine-a",
        submitted_at="2026-07-02T10:06:46Z",
        engine_commit="aaaaaaaaaa",
        plugin_commit="51e577b17b",
    )
    _write_mock_submission(
        source_dir / "engine-b",
        entry_id="entry-engine-b",
        submitted_at="2026-07-01T14:51:46Z",
        engine_commit="bbbbbbbbbb",
        plugin_commit="51e577b17b",
    )

    conflicts = _find_superseded_coexistence_conflicts(source_dir)

    assert conflicts == []


def test_same_code_combo_still_conflicts(tmp_path: Path) -> None:
    """Same signature AND same (engine_commit, plugin_commit) without
    supersedes annotation still triggers a conflict — the secondary
    grouping must not weaken the original coexistence rule for genuinely
    duplicate runs of the same code combination.
    """
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()

    _write_mock_submission(
        source_dir / "old-run",
        entry_id="old-entry-1",
        submitted_at="2026-07-01T12:55:12Z",
        engine_commit="ceec19abb0",
        plugin_commit="51e577b17b",
    )
    _write_mock_submission(
        source_dir / "new-run",
        entry_id="new-entry-2",
        submitted_at="2026-07-02T10:06:46Z",
        engine_commit="ceec19abb0",
        plugin_commit="51e577b17b",
    )

    conflicts = _find_superseded_coexistence_conflicts(source_dir)

    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["old_entry_id"] == "old-entry-1"
    assert conflict["new_entry_id"] == "new-entry-2"


def test_same_code_combo_resolved_by_supersedes(tmp_path: Path) -> None:
    """Same signature + same code combo + explicit ``supersedes`` annotation
    resolves the conflict (verifies the secondary grouping logic still
    honours the supersedes check within a code-combo subgroup).
    """
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()

    _write_mock_submission(
        source_dir / "old-run",
        entry_id="old-entry-1",
        submitted_at="2026-07-01T12:55:12Z",
        engine_commit="ceec19abb0",
        plugin_commit="51e577b17b",
    )
    _write_mock_submission(
        source_dir / "new-run",
        entry_id="new-entry-2",
        submitted_at="2026-07-02T10:06:46Z",
        engine_commit="ceec19abb0",
        plugin_commit="51e577b17b",
        supersedes="old-entry-1",
    )

    conflicts = _find_superseded_coexistence_conflicts(source_dir)

    assert conflicts == []


def test_mixed_code_combos_only_flag_same_combo(tmp_path: Path) -> None:
    """Mixing 4 entries with the real-world historical-pr layout:

    - entry-pr77 (ceec19abb0, 51e577b17b)  — new
    - entry-pr77-perfgate (ceec19abb0, 51e577b17b) — old, same code combo
    - entry-pr66 (ceec19abb0, e0686f12d1) — different plugin commit
    - entry-pr70 (ceec19abb0, 312ca80a90) — different plugin commit

    Only entry-pr77-perfgate conflicts with entry-pr77 (same code combo,
    no supersedes); entry-pr66 and entry-pr70 are independent comparison
    runs and must NOT be flagged.
    """
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()

    _write_mock_submission(
        source_dir / "historical-pr-pr77",
        entry_id="entry-pr77",
        submitted_at="2026-07-02T10:06:46Z",
        engine_commit="ceec19abb0",
        plugin_commit="51e577b17b",
    )
    _write_mock_submission(
        source_dir / "historical-pr-pr77-perfgate",
        entry_id="entry-pr77-perfgate",
        submitted_at="2026-07-01T12:55:12Z",
        engine_commit="ceec19abb0",
        plugin_commit="51e577b17b",
    )
    _write_mock_submission(
        source_dir / "historical-pr-pr66",
        entry_id="entry-pr66",
        submitted_at="2026-07-01T14:51:46Z",
        engine_commit="ceec19abb0",
        plugin_commit="e0686f12d1",
    )
    _write_mock_submission(
        source_dir / "historical-pr-pr70",
        entry_id="entry-pr70",
        submitted_at="2026-07-01T14:54:33Z",
        engine_commit="ceec19abb0",
        plugin_commit="312ca80a90",
    )

    conflicts = _find_superseded_coexistence_conflicts(source_dir)

    assert len(conflicts) == 1
    assert conflicts[0]["old_entry_id"] == "entry-pr77-perfgate"
    assert conflicts[0]["new_entry_id"] == "entry-pr77"


def _make_minimal_layout(tmp_path: Path) -> RepoLayout:
    """Build a minimal ``RepoLayout`` with a fake website aggregation script."""
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text(
        "import sys, pathlib\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "for name in ['leaderboard_single.json', 'leaderboard_multi.json',\n"
        "             'leaderboard_compare.json', 'last_updated.json']:\n"
        "    (out / name).write_text('[]' if name.startswith('leaderboard') else '{}')\n",
        encoding="utf-8",
    )
    benchmark_repo = tmp_path / "vllm-hust-benchmark"
    benchmark_repo.mkdir()
    return RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=benchmark_repo,
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=website_repo,
    )


def test_report_empty_when_clean() -> None:
    report = _build_rejected_superseded_report([], [], [])

    assert report["schema_version"] == "rejected-superseded-report/v1"
    assert report["rejected_submissions"] == []
    assert report["superseded_entries"] == []
    assert report["excluded_plugin_commits"] == []
    assert "generated_at" in report


def test_report_generated_on_rejection(tmp_path: Path) -> None:
    layout = _make_minimal_layout(tmp_path)

    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    failed_dir = source_dir / "failed-run"
    failed_dir.mkdir()
    (failed_dir / "STATUS").write_text("FAILED: server crashed\n", encoding="utf-8")

    output_dir = tmp_path / "out"

    exit_code = aggregate_to_website(
        layout=layout,
        source_dir=source_dir,
        output_dir=output_dir,
        execute=False,
    )

    assert exit_code == 2
    report_path = output_dir / "rejected_superseded_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "rejected-superseded-report/v1"
    assert len(report["rejected_submissions"]) == 1
    assert report["rejected_submissions"][0]["reason"] == "FAILED"
    assert "failed-run" in report["rejected_submissions"][0]["dir"]


def test_report_generated_on_success(tmp_path: Path) -> None:
    layout = _make_minimal_layout(tmp_path)

    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    clean_dir = source_dir / "clean-run"
    clean_dir.mkdir()
    (clean_dir / "STATUS").write_text("OK\n", encoding="utf-8")
    (clean_dir / "run_leaderboard.json").write_text(
        '{"entry_id": "test"}\n', encoding="utf-8"
    )
    _write_manifest(clean_dir)

    output_dir = tmp_path / "out"

    exit_code = aggregate_to_website(
        layout=layout,
        source_dir=source_dir,
        output_dir=output_dir,
        execute=True,
    )

    assert exit_code == 0
    report_path = output_dir / "rejected_superseded_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "rejected-superseded-report/v1"
    assert report["rejected_submissions"] == []
    assert isinstance(report["superseded_entries"], list)
    assert isinstance(report["excluded_plugin_commits"], list)


def test_report_signature_conflict_rejection(tmp_path: Path) -> None:
    layout = _make_minimal_layout(tmp_path)

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

    output_dir = tmp_path / "out"

    exit_code = aggregate_to_website(
        layout=layout,
        source_dir=source_dir,
        output_dir=output_dir,
        execute=False,
    )

    assert exit_code == 2
    report_path = output_dir / "rejected_superseded_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report["rejected_submissions"]) == 1
    assert report["rejected_submissions"][0]["reason"] == "signature_conflict"
    assert "old-run" in report["rejected_submissions"][0]["dir"]


def test_write_rejected_superseded_report_creates_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "out"
    report = _build_rejected_superseded_report([], [], [])

    _write_rejected_superseded_report(destination, report)

    report_path = destination / "rejected_superseded_report.json"
    assert report_path.is_file()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "rejected-superseded-report/v1"


def test_report_superseded_entries_from_metadata(tmp_path: Path) -> None:
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    _write_mock_submission(
        source_dir / "new-run",
        entry_id="new-entry-1",
        submitted_at="2026-07-22T00:00:00Z",
        supersedes="old-run-dir-name",
        supersedes_reason="previous run had wrong config",
    )

    report = _build_rejected_superseded_report(
        [],
        [],
        [],
        source_dir=source_dir,
    )

    assert len(report["superseded_entries"]) == 1
    entry = report["superseded_entries"][0]
    assert entry["old_entry_id"] == "old-run-dir-name"
    assert entry["new_entry_id"] == "new-entry-1"
    assert entry["supersedes_reason"] == "previous run had wrong config"
    assert entry["archive_path"] is None


def test_report_superseded_archive_path_found(tmp_path: Path) -> None:
    source_dir = tmp_path / "submissions"
    source_dir.mkdir()
    _write_mock_submission(
        source_dir / "new-run",
        entry_id="new-entry-1",
        submitted_at="2026-07-22T00:00:00Z",
        supersedes="old-run-dir-name",
    )

    archive_root = tmp_path / "archive" / "suspect"
    (archive_root / "topic-a" / "old-run-dir-name").mkdir(parents=True)

    from vllm_hust_benchmark.integration import _scan_superseded_entries

    entries = _scan_superseded_entries(source_dir, archive_suspect_root=archive_root)
    assert len(entries) == 1
    assert entries[0]["archive_path"] is not None
    assert "old-run-dir-name" in entries[0]["archive_path"]
