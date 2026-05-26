import json
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft7Validator

from vllm_hust_benchmark.submission_artifacts import iter_submission_artifact_paths
from vllm_hust_benchmark.submission_artifacts import load_submission_artifact
from vllm_hust_benchmark.submission_artifacts import (
    normalize_submission_artifacts_in_tree,
)
from vllm_hust_benchmark.submission_artifacts import (
    normalize_submission_artifact_contract,
)
from vllm_hust_benchmark.submission_artifacts import validate_manifest_artifacts


def test_validate_manifest_artifacts_uses_manifest_referenced_artifact(
    tmp_path: Path,
) -> None:
    submission_dir = tmp_path / "submission-a"
    submission_dir.mkdir()
    artifact_path = submission_dir / "custom_leaderboard.json"
    artifact_path.write_text(
        json.dumps({"model": {"name": "Qwen2.5-7B-Instruct"}}),
        encoding="utf-8",
    )
    (submission_dir / "leaderboard_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "leaderboard-export-manifest/v2",
                "generated_at": "2026-05-26T00:00:00Z",
                "entries": [{"leaderboard_artifact": "custom_leaderboard.json"}],
            }
        ),
        encoding="utf-8",
    )

    validator = Draft7Validator(
        {
            "type": "object",
            "required": ["model"],
            "properties": {
                "model": {
                    "type": "object",
                    "required": [
                        "canonical_id",
                        "repo_id",
                        "short_name",
                        "display_name",
                        "name",
                    ],
                }
            },
        }
    )

    errors = validate_manifest_artifacts(
        manifests=[submission_dir / "leaderboard_manifest.json"],
        validator=validator,
    )

    assert errors == []


def test_normalize_submission_artifacts_in_tree_backfills_legacy_manifest(
    tmp_path: Path,
) -> None:
    legacy_dir = tmp_path / "existing-run"
    legacy_dir.mkdir()
    artifact_path = legacy_dir / "run_leaderboard.json"
    artifact_path.write_text(
        json.dumps({"model": {"name": "Qwen2.5-14B-Instruct"}}),
        encoding="utf-8",
    )

    changed = normalize_submission_artifacts_in_tree(tmp_path)

    manifest_path = legacy_dir / "leaderboard_manifest.json"
    assert manifest_path.is_file()
    assert changed == [artifact_path]

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["model"]["name"] == "Qwen/Qwen2.5-14B-Instruct"
    assert payload["model"]["canonical_id"] == "hf:Qwen/Qwen2.5-14B-Instruct"


def test_checked_in_submissions_are_already_normalized() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_paths = iter_submission_artifact_paths(repo_root / "submissions")

    assert artifact_paths

    for artifact_path in artifact_paths:
        payload = load_submission_artifact(artifact_path)
        normalized = normalize_submission_artifact_contract(deepcopy(payload))

        assert payload == normalized, artifact_path
