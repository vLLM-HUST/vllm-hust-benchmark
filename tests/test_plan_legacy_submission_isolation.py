from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plan_legacy_submission_isolation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plan_legacy_submission_isolation", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_or_verify_plan = MODULE.build_or_verify_plan
main = MODULE.main
PlanError = MODULE.PlanError
plan_core = MODULE._plan_core
plan_fingerprint = MODULE._plan_fingerprint


def _write_submission(
    root: Path,
    name: str,
    *,
    historical: bool,
    nested_file: bool = False,
) -> Path:
    submission = root / "submissions" / name
    submission.mkdir(parents=True)
    artifact = {
        "entry_id": f"entry-{name}",
        "metadata": {
            "data_source": (
                "real-online-historical-pr-backfill" if historical else "legacy"
            ),
            "git_commit": "a" * 40,
            "runtime_provenance": {
                "engine": {"commit": "b" * 40},
                "plugin": {"commit": "c" * 40},
            },
        },
    }
    (submission / "run_leaderboard.json").write_text(
        json.dumps(artifact) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "leaderboard-export-manifest/v1",
        "entries": [{"leaderboard_artifact": "run_leaderboard.json"}],
    }
    (submission / "leaderboard_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    if nested_file:
        (submission / "raw").mkdir()
        (submission / "raw" / "result.json").write_text("{}\n", encoding="utf-8")
    return submission


def _build(repo: Path, *, existing_index: Path | None = None) -> dict:
    return build_or_verify_plan(
        repo_root=repo,
        source_root_value="submissions",
        archive_root_value="archive/legacy/incomplete-evidence",
        archive_date="2026-08-05",
        existing_index=existing_index,
    )


def _save_index(repo: Path, plan: dict) -> Path:
    path = repo / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_plan_is_deterministic_complete_and_read_only(tmp_path: Path) -> None:
    second = _write_submission(tmp_path, "z-checksum", historical=False)
    first = _write_submission(
        tmp_path, "a-provenance", historical=True, nested_file=True
    )
    before = sorted(
        (path.relative_to(tmp_path).as_posix(), path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    )

    plan = _build(tmp_path)
    repeated = _build(tmp_path)

    assert plan == repeated
    assert plan["verification"]["ok"] is True
    assert [entry["original_path"] for entry in plan["entries"]] == [
        "submissions/a-provenance",
        "submissions/z-checksum",
    ]
    assert [entry["failure_reason"] for entry in plan["entries"]] == [
        "PROVENANCE_INCOMPLETE",
        "CHECKSUM_INCOMPLETE",
    ]
    assert plan["entries"][0]["archive_path"].endswith("/2026-08-05/a-provenance")
    assert plan["entries"][0]["entry_id"] == "entry-a-provenance"
    assert plan["entries"][0]["engine_commit"] == "b" * 40
    assert plan["entries"][0]["plugin_commit"] == "c" * 40
    assert plan["entries"][0]["inventory"]["directories"] == ["raw"]
    assert [item["path"] for item in plan["entries"][0]["inventory"]["files"]] == [
        "leaderboard_manifest.json",
        "raw/result.json",
        "run_leaderboard.json",
    ]
    assert not (first / "env-manifest.json").exists()
    assert not (second / "checksums.sha256").exists()
    after = sorted(
        (path.relative_to(tmp_path).as_posix(), path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert after == before


def test_existing_index_detects_source_hash_change(tmp_path: Path) -> None:
    submission = _write_submission(tmp_path, "legacy", historical=False)
    index = _save_index(tmp_path, _build(tmp_path))
    (submission / "run_leaderboard.json").write_text("{}\n", encoding="utf-8")

    verified = _build(tmp_path, existing_index=index)

    assert verified["verification"]["ok"] is False
    assert "source_hash_changed" in {
        error["kind"] for error in verified["verification"]["errors"]
    }


def test_existing_index_recognizes_copy_and_completed_move(tmp_path: Path) -> None:
    submission = _write_submission(tmp_path, "legacy", historical=False)
    plan = _build(tmp_path)
    index = _save_index(tmp_path, plan)
    target = (
        tmp_path
        / "archive"
        / "legacy"
        / "incomplete-evidence"
        / "2026-08-05"
        / "legacy"
    )
    shutil.copytree(submission, target)

    copied = _build(tmp_path, existing_index=index)
    assert copied["verification"]["ok"] is True
    assert copied["verification"]["states"] == [
        {"path": "submissions/legacy", "state": "copied_source_still_active"}
    ]

    shutil.rmtree(submission)
    archived = _build(tmp_path, existing_index=index)
    assert archived["verification"]["ok"] is True
    assert archived["verification"]["states"] == [
        {"path": "submissions/legacy", "state": "already_archived"}
    ]


def test_existing_index_detects_archive_target_conflict(tmp_path: Path) -> None:
    submission = _write_submission(tmp_path, "legacy", historical=False)
    index = _save_index(tmp_path, _build(tmp_path))
    target = (
        tmp_path
        / "archive"
        / "legacy"
        / "incomplete-evidence"
        / "2026-08-05"
        / "legacy"
    )
    shutil.copytree(submission, target)
    (target / "run_leaderboard.json").write_text("tampered\n", encoding="utf-8")

    verified = _build(tmp_path, existing_index=index)

    assert verified["verification"]["ok"] is False
    assert "archive_target_conflict" in {
        error["kind"] for error in verified["verification"]["errors"]
    }


def test_existing_index_detects_new_unindexed_failure(tmp_path: Path) -> None:
    _write_submission(tmp_path, "first", historical=False)
    index = _save_index(tmp_path, _build(tmp_path))
    _write_submission(tmp_path, "second", historical=True)

    verified = _build(tmp_path, existing_index=index)

    assert verified["verification"]["ok"] is False
    assert "unindexed_admission_failure" in {
        error["kind"] for error in verified["verification"]["errors"]
    }


def test_cli_has_no_apply_mode_and_returns_two_on_conflict(
    tmp_path: Path, capsys
) -> None:
    submission = _write_submission(tmp_path, "legacy", historical=False)
    plan = _build(tmp_path)
    index = _save_index(tmp_path, plan)
    (submission / "run_leaderboard.json").write_text("changed\n", encoding="utf-8")

    result = main(
        [
            "--repo-root",
            str(tmp_path),
            "--archive-date",
            "2026-08-05",
            "--verify-index",
            str(index),
        ]
    )

    assert result == 2
    output = json.loads(capsys.readouterr().out)
    assert output["verification"]["ok"] is False


def test_missing_source_root_and_invalid_date_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(PlanError, match="source root is not a directory"):
        _build(tmp_path)

    (tmp_path / "submissions").mkdir()
    with pytest.raises(PlanError, match="real date"):
        build_or_verify_plan(
            repo_root=tmp_path,
            source_root_value="submissions",
            archive_root_value="archive/legacy/incomplete-evidence",
            archive_date="2026-02-30",
        )


def test_existing_index_requires_untampered_fingerprint(tmp_path: Path) -> None:
    _write_submission(tmp_path, "legacy", historical=False)
    plan = _build(tmp_path)
    del plan["plan_sha256"]
    index = _save_index(tmp_path, plan)

    with pytest.raises(PlanError, match="plan_sha256 is required"):
        _build(tmp_path, existing_index=index)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("original_path", "submissions/..", "unsafe original path"),
        (
            "archive_path",
            "archive/legacy/incomplete-evidence/2026-08-05/..",
            "unsafe archive path",
        ),
    ],
)
def test_existing_index_rejects_parent_directory_entry(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    _write_submission(tmp_path, "legacy", historical=False)
    plan = _build(tmp_path)
    plan["entries"][0][field] = value
    plan["plan_sha256"] = plan_fingerprint(plan_core(plan))
    index = _save_index(tmp_path, plan)

    with pytest.raises(PlanError, match=message):
        _build(tmp_path, existing_index=index)


def test_repository_index_records_completed_move() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    index = (
        repo_root
        / "archive"
        / "legacy"
        / "incomplete-evidence"
        / "2026-08-07"
        / "index.json"
    )
    payload = json.loads(index.read_text(encoding="utf-8"))

    result = build_or_verify_plan(
        repo_root=repo_root,
        source_root_value="submissions",
        archive_root_value="archive/legacy/incomplete-evidence",
        archive_date="2026-08-07",
        existing_index=index,
    )

    assert payload["verification"]["ok"] is True
    assert {state["state"] for state in payload["verification"]["states"]} == {
        "already_archived"
    }
    assert result["verification"]["ok"] is True
    assert {state["state"] for state in result["verification"]["states"]} == {
        "already_archived"
    }
