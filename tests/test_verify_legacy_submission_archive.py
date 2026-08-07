from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_legacy_submission_archive.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "verify_legacy_submission_archive", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PlanError = MODULE.PlanError
index_fingerprint = MODULE.index_fingerprint
inventory_with_fingerprint = MODULE.inventory_with_fingerprint
main = MODULE.main
verify_supplemental_index = MODULE.verify_supplemental_index


def _write_artifact(directory: Path, *, entry_id: str, target_id: str | None) -> None:
    directory.mkdir(parents=True)
    metadata = {"spec_id": "spec-test"}
    if target_id is not None:
        metadata["target_id"] = target_id
    (directory / "run_leaderboard.json").write_text(
        json.dumps({"entry_id": entry_id, "metadata": metadata}) + "\n",
        encoding="utf-8",
    )
    (directory / "leaderboard_manifest.json").write_text("{}\n", encoding="utf-8")


def _build_index(repo: Path, *, missing_target_value: str | None = None) -> dict:
    archive_date = "2026-08-07"
    archived_duplicate = (
        repo
        / "archive"
        / "legacy"
        / "superseded-coexistence"
        / archive_date
        / "duplicate"
    )
    retained = repo / "submissions" / "retained"
    archived_missing = (
        repo / "archive" / "legacy" / "missing-target-id" / archive_date / "missing"
    )
    _write_artifact(archived_duplicate, entry_id="duplicate-entry", target_id=None)
    _write_artifact(retained, entry_id="duplicate-entry", target_id="target-retained")
    _write_artifact(
        archived_missing, entry_id="missing-entry", target_id=missing_target_value
    )
    payload = {
        "archive_date": archive_date,
        "benchmark_commit": "a" * 40,
        "coexistence": [
            {
                "archive_path": (
                    "archive/legacy/superseded-coexistence/2026-08-07/duplicate"
                ),
                "entry_id": "duplicate-entry",
                "reason": "superseded-coexistence",
                "retained_inventory": inventory_with_fingerprint(retained),
                "retained_path": "submissions/retained",
                "selection_reason": "retained copy has more complete evidence",
                "source_inventory": inventory_with_fingerprint(archived_duplicate),
                "source_path": "submissions/duplicate",
            }
        ],
        "missing_target_id": [
            {
                "archive_path": "archive/legacy/missing-target-id/2026-08-07/missing",
                "entry_id": "missing-entry",
                "errors": ["metadata.target_id must be explicitly recorded"],
                "reason": "metadata.target_id missing",
                "source_inventory": inventory_with_fingerprint(archived_missing),
                "source_path": "submissions/missing",
                "spec_id": "spec-test",
            }
        ],
        "schema_version": MODULE.SCHEMA_VERSION,
    }
    payload["index_sha256"] = index_fingerprint(payload)
    return payload


def _write_index(repo: Path, payload: dict) -> Path:
    path = repo / "supplemental-index.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _resign(payload: dict) -> None:
    payload["index_sha256"] = index_fingerprint(payload)


def test_valid_supplemental_index_and_cli(tmp_path: Path, capsys) -> None:
    index = _write_index(tmp_path, _build_index(tmp_path))

    result = verify_supplemental_index(repo_root=tmp_path, index_path=index)
    exit_code = main(["--repo-root", str(tmp_path), "--index", str(index)])

    assert result["ok"] is True
    assert result["entry_count"] == 2
    assert result["coexistence_count"] == 1
    assert result["missing_target_id_count"] == 1
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_rejects_tampered_index_fingerprint(tmp_path: Path) -> None:
    payload = _build_index(tmp_path)
    payload["missing_target_id"][0]["spec_id"] = "tampered"
    index = _write_index(tmp_path, payload)

    with pytest.raises(PlanError, match="index_sha256 does not match"):
        verify_supplemental_index(repo_root=tmp_path, index_path=index)


def test_rejects_unsafe_archive_path(tmp_path: Path) -> None:
    payload = _build_index(tmp_path)
    payload["missing_target_id"][0]["archive_path"] = "archive/legacy/../escape"
    _resign(payload)
    index = _write_index(tmp_path, payload)

    with pytest.raises(PlanError, match="unsafe missing_target_id.*archive_path"):
        verify_supplemental_index(repo_root=tmp_path, index_path=index)


def test_rejects_archive_inventory_change(tmp_path: Path) -> None:
    payload = _build_index(tmp_path)
    index = _write_index(tmp_path, payload)
    archived = tmp_path / payload["missing_target_id"][0]["archive_path"]
    (archived / "run_leaderboard.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(PlanError, match="inventory mismatch"):
        verify_supplemental_index(repo_root=tmp_path, index_path=index)


def test_rejects_tampered_inventory_fingerprint(tmp_path: Path) -> None:
    payload = _build_index(tmp_path)
    payload["missing_target_id"][0]["source_inventory"]["files"][0]["size"] += 1
    _resign(payload)
    index = _write_index(tmp_path, payload)

    with pytest.raises(PlanError, match="tree_sha256 does not match"):
        verify_supplemental_index(repo_root=tmp_path, index_path=index)


def test_rejects_archive_root_symlink(tmp_path: Path) -> None:
    payload = _build_index(tmp_path)
    index = _write_index(tmp_path, payload)
    archived = tmp_path / payload["missing_target_id"][0]["archive_path"]
    target = tmp_path / "archive-symlink-target"
    shutil.copytree(archived, target)
    shutil.rmtree(archived)
    archived.symlink_to(target, target_is_directory=True)

    with pytest.raises(PlanError, match="archive tree is missing or unsafe"):
        verify_supplemental_index(repo_root=tmp_path, index_path=index)


def test_rejects_source_still_active(tmp_path: Path) -> None:
    payload = _build_index(tmp_path)
    index = _write_index(tmp_path, payload)
    (tmp_path / "submissions" / "missing").mkdir()

    with pytest.raises(PlanError, match="source path is still active"):
        verify_supplemental_index(repo_root=tmp_path, index_path=index)


def test_rejects_missing_selection_reason(tmp_path: Path) -> None:
    payload = _build_index(tmp_path)
    del payload["coexistence"][0]["selection_reason"]
    _resign(payload)
    index = _write_index(tmp_path, payload)

    with pytest.raises(PlanError, match="selection_reason is required"):
        verify_supplemental_index(repo_root=tmp_path, index_path=index)


def test_rejects_missing_target_entry_with_target_id(tmp_path: Path) -> None:
    payload = _build_index(tmp_path, missing_target_value="unexpected-target")
    index = _write_index(tmp_path, payload)

    with pytest.raises(PlanError, match="archived artifact has metadata.target_id"):
        verify_supplemental_index(repo_root=tmp_path, index_path=index)


def test_repository_supplemental_index_is_current() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    index = (
        repo_root
        / "archive"
        / "legacy"
        / "supplemental-isolation"
        / "2026-08-07"
        / "index.json"
    )
    payload = json.loads(index.read_text(encoding="utf-8"))

    result = verify_supplemental_index(repo_root=repo_root, index_path=index)

    assert result["ok"] is True
    assert result["coexistence_count"] == 1
    assert result["missing_target_id_count"] == 6
    assert result["entry_count"] == 7
    coexistence = payload["coexistence"][0]
    assert "measurement_block.json" in coexistence["selection_reason"]
    assert "raw_benchmark_result.json" in coexistence["selection_reason"]
    source_files = {item["path"] for item in coexistence["source_inventory"]["files"]}
    retained_files = {
        item["path"] for item in coexistence["retained_inventory"]["files"]
    }
    assert retained_files - source_files == {
        "measurement_block.json",
        "raw_benchmark_result.json",
    }
