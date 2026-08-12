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
    / "verify_superseded_coexistence_migration.py"
)
SPEC = importlib.util.spec_from_file_location(
    "verify_superseded_coexistence_migration", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

MigrationIndexError = MODULE.MigrationIndexError
effective_spec_fingerprint = MODULE.effective_spec_fingerprint
main = MODULE.main
verify_migration_index = MODULE.verify_migration_index


def _same_spec(*, numeric_strings: bool = False) -> dict:
    return {
        "schema_version": "benchmark-same-spec/v1",
        "spec_id": "official-test-spec",
        "scenario": "agent-research-online",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "model_parameters": "14B",
        "model_precision": "FP16",
        "model_quantization": "",
        "hardware_vendor": "Huawei",
        "hardware_chip_model": "910B2",
        "chip_count": 1,
        "node_count": 1,
        "resolved_server_parameters": {
            "gpu_memory_utilization": "0.6" if numeric_strings else 0.6,
            "max_model_len": "32768" if numeric_strings else 32768,
            "model": "/models/qwen",
            "port": 8000,
        },
        "resolved_client_parameters": {
            "model": "/models/qwen",
            "num_prompts": 32,
            "port": 8000,
        },
        "resolved_spec_hash": "raw-hash-may-differ",
    }


def _write_entry(
    directory: Path,
    *,
    entry_id: str,
    same_spec: dict,
    supersedes: str | None = None,
) -> None:
    directory.mkdir(parents=True)
    metadata = {"supersedes": supersedes} if supersedes else {}
    (directory / "run_leaderboard.json").write_text(
        json.dumps({"entry_id": entry_id, "metadata": metadata}) + "\n",
        encoding="utf-8",
    )
    (directory / "resolved_same_spec.json").write_text(
        json.dumps(same_spec) + "\n", encoding="utf-8"
    )


def _build_index(repo: Path) -> tuple[dict, Path]:
    archive_date = "2026-08-11"
    archive_parent = (
        repo
        / "archive"
        / "suspect"
        / "superseded-coexistence-historical-pr-backfill-20260811"
    )
    archived = archive_parent / "old-run"
    retained = repo / "submissions" / "new-run"
    archived_spec = _same_spec(numeric_strings=True)
    retained_spec = _same_spec()
    _write_entry(archived, entry_id="old-entry", same_spec=archived_spec)
    _write_entry(
        retained,
        entry_id="new-entry",
        same_spec=retained_spec,
        supersedes="old-entry",
    )
    payload = {
        "schema_version": MODULE.SCHEMA_VERSION,
        "archive_date": archive_date,
        "benchmark_commit": "a" * 40,
        "entries": [
            {
                "archived_entry_id": "old-entry",
                "archived_path": (
                    "archive/suspect/"
                    "superseded-coexistence-historical-pr-backfill-20260811/old-run"
                ),
                "retained_entry_id": "new-entry",
                "retained_path": "submissions/new-run",
                "selection_reason": "later equivalent run retained",
                "effective_spec_sha256": effective_spec_fingerprint(archived_spec),
            }
        ],
        "production_fallback": False,
    }
    index = archive_parent / "index.json"
    index.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload, index


def test_valid_migration_index_and_cli(tmp_path: Path, capsys) -> None:
    _, index = _build_index(tmp_path)

    result = verify_migration_index(repo_root=tmp_path, index_path=index)
    exit_code = main(["--repo-root", str(tmp_path), "--index", str(index)])

    assert result["ok"] is True
    assert result["entry_count"] == 1
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_rejects_unsafe_archive_path(tmp_path: Path) -> None:
    payload, index = _build_index(tmp_path)
    payload["entries"][0]["archived_path"] = "archive/suspect/../escape"
    index.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MigrationIndexError, match="unsafe entries.*archived_path"):
        verify_migration_index(repo_root=tmp_path, index_path=index)


def test_rejects_archive_symlink(tmp_path: Path) -> None:
    payload, index = _build_index(tmp_path)
    archived = tmp_path / payload["entries"][0]["archived_path"]
    target = tmp_path / "archive-target"
    shutil.copytree(archived, target)
    shutil.rmtree(archived)
    archived.symlink_to(target, target_is_directory=True)

    with pytest.raises(MigrationIndexError, match="contains a symlink"):
        verify_migration_index(repo_root=tmp_path, index_path=index)


def test_rejects_index_symlink(tmp_path: Path) -> None:
    _, index = _build_index(tmp_path)
    target = tmp_path / "migration-index-target.json"
    index.rename(target)
    index.symlink_to(target)

    with pytest.raises(MigrationIndexError, match="index must not be a symlink"):
        verify_migration_index(repo_root=tmp_path, index_path=index)


def test_rejects_missing_supersedes_reference(tmp_path: Path) -> None:
    payload, index = _build_index(tmp_path)
    retained = tmp_path / payload["entries"][0]["retained_path"]
    artifact = json.loads((retained / "run_leaderboard.json").read_text())
    artifact["metadata"].pop("supersedes")
    (retained / "run_leaderboard.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )

    with pytest.raises(MigrationIndexError, match="metadata.supersedes"):
        verify_migration_index(repo_root=tmp_path, index_path=index)


def test_rejects_effective_spec_mismatch(tmp_path: Path) -> None:
    payload, index = _build_index(tmp_path)
    retained = tmp_path / payload["entries"][0]["retained_path"]
    spec = json.loads((retained / "resolved_same_spec.json").read_text())
    spec["resolved_server_parameters"]["max_model_len"] = 4096
    (retained / "resolved_same_spec.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )

    with pytest.raises(MigrationIndexError, match="effective specs do not match"):
        verify_migration_index(repo_root=tmp_path, index_path=index)


def test_rejects_incomplete_effective_spec(tmp_path: Path) -> None:
    payload, index = _build_index(tmp_path)
    archived = tmp_path / payload["entries"][0]["archived_path"]
    spec = json.loads((archived / "resolved_same_spec.json").read_text())
    spec.pop("model_precision")
    (archived / "resolved_same_spec.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )

    with pytest.raises(MigrationIndexError, match="missing required fields"):
        verify_migration_index(repo_root=tmp_path, index_path=index)


def test_rejects_production_fallback(tmp_path: Path) -> None:
    payload, index = _build_index(tmp_path)
    payload["production_fallback"] = True
    index.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MigrationIndexError, match="production_fallback must be false"):
        verify_migration_index(repo_root=tmp_path, index_path=index)


def test_repository_migration_index_is_current() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    index = (
        repo_root
        / "archive"
        / "suspect"
        / "superseded-coexistence-historical-pr-backfill-20260811"
        / "index.json"
    )

    result = verify_migration_index(repo_root=repo_root, index_path=index)

    assert result["ok"] is True
    assert result["benchmark_commit"] == (
        "d09950ba95883e770c0f593ecf900d9fbc84c218"  # pragma: allowlist secret
    )
    assert result["entry_count"] == 2
