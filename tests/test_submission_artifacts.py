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


def test_normalize_submission_artifact_contract_backfills_versions_from_repo() -> None:
    artifact = {
        "model": {"name": "Qwen/Qwen2.5-7B-Instruct"},
        "engine_version": "0.17.2rc1.dev450+g289b51ab2.d20260417",
        "versions": {
            "protocol": "N/A",
            "backend": "N/A",
            "core": "N/A",
            "benchmark": "0.1.0",
        },
        "metadata": {
            "github_repository": "vLLM-HUST/vllm-hust",
            "engine_version": "0.17.2rc1.dev450+g289b51ab2.d20260417",
        },
    }

    normalized = normalize_submission_artifact_contract(deepcopy(artifact))

    assert normalized["versions"]["core"] == "0.17.2rc1.dev450+g289b51ab2.d20260417"
    assert normalized["versions"]["backend"] == "N/A"


def test_normalize_submission_artifact_contract_backfills_same_spec_backend() -> None:
    artifact = {
        "model": {"name": "Qwen/Qwen2.5-14B-Instruct"},
        "engine_version": "v0.17.2.post1-1079-gd4a408c47",
        "versions": {
            "protocol": "N/A",
            "backend": "N/A",
            "core": "0.17.2.post1-1079-gd4a408c47",
            "benchmark": "0.1.0",
        },
        "metadata": {
            "github_repository": "vLLM-HUST/vllm-hust",
            "engine_version": "v0.17.2.post1-1079-gd4a408c47",
            "data_source": "vllm-ascend-hust-ci-same-spec",
        },
    }

    normalized = normalize_submission_artifact_contract(deepcopy(artifact))

    assert normalized["versions"]["backend"] == "0.1.0"


def test_normalize_submission_artifacts_in_tree_pairs_plugin_with_core_submission(
    tmp_path: Path,
) -> None:
    core_dir = tmp_path / "core-run"
    core_dir.mkdir()
    plugin_dir = tmp_path / "plugin-run"
    plugin_dir.mkdir()

    core_artifact = {
        "entry_id": "210b1171-0982-403e-849e-f19d45d2d3cc",
        "engine": "vllm-hust",
        "engine_version": "0.17.2rc1.dev450+g289b51ab2.d20260417",
        "config_type": "single_gpu",
        "model": {"name": "Qwen/Qwen2.5-7B-Instruct"},
        "workload": {"name": "sharegpt-online"},
        "hardware": {"chip_model": "910B3", "chip_count": 1},
        "metrics": {"ttft_ms": 1916.74, "throughput_tps": 154.58},
        "versions": {
            "protocol": "N/A",
            "backend": "0.1.dev2743+g7f888566e",
            "core": "N/A",
            "benchmark": "0.1.0",
        },
        "metadata": {
            "submitted_at": "2026-04-17T07:15:37Z",
            "github_repository": "vLLM-HUST/vllm-hust",
            "engine_version": "0.17.2rc1.dev450+g289b51ab2.d20260417",
        },
    }
    plugin_artifact = {
        "entry_id": "a423d806-4cd9-4ffd-87ef-48b690cc757f",
        "engine": "vllm-hust",
        "engine_version": "0.1.dev2743+g7f888566e",
        "config_type": "single_gpu",
        "model": {"name": "Qwen/Qwen2.5-7B-Instruct"},
        "workload": {"name": "sharegpt-online"},
        "hardware": {"chip_model": "910B3", "chip_count": 1},
        "metrics": {
            "ttft_ms": 1916.7414716590429,
            "throughput_tps": 154.57739501109546,
        },
        "versions": {
            "protocol": "N/A",
            "backend": "N/A",
            "core": "N/A",
            "benchmark": "0.1.0",
        },
        "metadata": {
            "submitted_at": "2026-04-17T14:35:27Z",
            "github_repository": "vLLM-HUST/vllm-ascend-hust",
            "engine_version": "0.1.dev2743+g7f888566e",
        },
    }

    (core_dir / "run_leaderboard.json").write_text(
        json.dumps(core_artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (plugin_dir / "run_leaderboard.json").write_text(
        json.dumps(plugin_artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    changed = normalize_submission_artifacts_in_tree(tmp_path)

    assert core_dir / "run_leaderboard.json" in changed
    assert plugin_dir / "run_leaderboard.json" in changed

    normalized_core = json.loads(
        (core_dir / "run_leaderboard.json").read_text(encoding="utf-8")
    )
    normalized_plugin = json.loads(
        (plugin_dir / "run_leaderboard.json").read_text(encoding="utf-8")
    )

    assert normalized_core["versions"]["core"] == "0.17.2rc1.dev450+g289b51ab2.d20260417"
    assert normalized_plugin["versions"]["backend"] == "0.1.dev2743+g7f888566e"
    assert normalized_plugin["versions"]["core"] == "0.17.2rc1.dev450+g289b51ab2.d20260417"


def test_normalize_submission_artifacts_in_tree_pairs_same_spec_plugin_submission(
    tmp_path: Path,
) -> None:
    core_dir = tmp_path / "core-run"
    core_dir.mkdir()
    plugin_dir = tmp_path / "plugin-run"
    plugin_dir.mkdir()

    core_artifact = {
        "entry_id": "core-same-spec",
        "engine": "vllm-hust",
        "engine_version": "v0.17.2.post1-1079-gd4a408c47",
        "config_type": "single_gpu",
        "model": {"name": "Qwen/Qwen2.5-14B-Instruct"},
        "workload": {"name": "random-online"},
        "hardware": {"chip_model": "910B3", "chip_count": 1},
        "metrics": {"ttft_ms": 393.87, "throughput_tps": 222.78},
        "versions": {
            "protocol": "N/A",
            "backend": "N/A",
            "core": "0.17.2.post1-1079-gd4a408c47",
            "benchmark": "0.1.0",
        },
        "metadata": {
            "submitted_at": "2026-05-12T10:31:50Z",
            "github_repository": "vLLM-HUST/vllm-hust",
            "engine_version": "v0.17.2.post1-1079-gd4a408c47",
            "data_source": "vllm-ascend-hust-ci-same-spec",
        },
    }
    plugin_artifact = {
        "entry_id": "plugin-same-spec",
        "engine": "vllm-hust",
        "engine_version": "bc9634de",
        "config_type": "single_gpu",
        "model": {"name": "Qwen/Qwen2.5-14B-Instruct"},
        "workload": {"name": "random-online"},
        "hardware": {"chip_model": "910B3", "chip_count": 1},
        "metrics": {"ttft_ms": 393.871, "throughput_tps": 222.779},
        "versions": {
            "protocol": "N/A",
            "backend": "N/A",
            "core": "N/A",
            "benchmark": "0.1.0",
        },
        "metadata": {
            "submitted_at": "2026-05-12T10:32:10Z",
            "github_repository": "vLLM-HUST/vllm-ascend-hust",
            "engine_version": "bc9634de",
            "data_source": "vllm-ascend-hust-ci-same-spec",
        },
    }

    (core_dir / "run_leaderboard.json").write_text(
        json.dumps(core_artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (plugin_dir / "run_leaderboard.json").write_text(
        json.dumps(plugin_artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    changed = normalize_submission_artifacts_in_tree(tmp_path)

    assert core_dir / "run_leaderboard.json" in changed
    assert plugin_dir / "run_leaderboard.json" in changed

    normalized_core = json.loads(
        (core_dir / "run_leaderboard.json").read_text(encoding="utf-8")
    )
    normalized_plugin = json.loads(
        (plugin_dir / "run_leaderboard.json").read_text(encoding="utf-8")
    )

    assert normalized_core["versions"]["backend"] == "0.1.0"
    assert normalized_plugin["versions"]["backend"] == "0.1.0"
    assert normalized_plugin["versions"]["core"] == "0.17.2.post1-1079-gd4a408c47"


def test_checked_in_submissions_are_already_normalized() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_paths = iter_submission_artifact_paths(repo_root / "submissions")

    assert artifact_paths

    for artifact_path in artifact_paths:
        payload = load_submission_artifact(artifact_path)
        normalized = normalize_submission_artifact_contract(deepcopy(payload))

        assert payload == normalized, artifact_path
