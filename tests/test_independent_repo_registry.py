"""Tests for the independent optimization repo result card registry (issue #89)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vllm_hust_benchmark.independent_repo_registry import (  # noqa: E402
    REQUIRED_REPOS,
    SCHEMA_PATH,
    check_required_repo_coverage,
    load_registry,
    load_schema,
    validate_registry_semantics,
)

REGISTRY_PATH = (
    REPO_ROOT
    / "leaderboard-data"
    / "independent-repos"
    / "independent_repo_result_cards.json"
)

ZERO_COMMIT = "0" * 40


def _valid_card(card_id: str = "card-1") -> dict:
    return {
        "card_id": card_id,
        "status": "formal-presentable",
        "workload": "random-online",
        "comparison_id": "cmp-1",
        "point_role": "baseline",
        "spec_summary": {
            "model": "Qwen2.5-14B-Instruct",
            "hardware": "1x910B2",
            "precision": "BF16",
        },
        "metrics": {
            "ttft_ms": 250.0,
            "tbt_ms": 35.0,
            "throughput_tps": 245.0,
            "peak_mem_mb": 30000,
            "error_rate": 0.0,
        },
        "evidence": {
            "repo_commit": ZERO_COMMIT,
            "artifact_url": "https://example.com/artifact.json",
            "repetitions": 3,
        },
    }


def _valid_repo(name: str = "vllm-hust-bidkv") -> dict:
    return {
        "repo_name": name,
        "repo_url": f"https://github.com/vLLM-HUST/{name}",
        "coverage_class": "independent-repo",
        "mechanism_summary": "test mechanism",
        "serving_prs": [161],
        "result_cards": [_valid_card()],
    }


def _valid_registry(repos: list[dict] | None = None) -> dict:
    if repos is None:
        repos = [
            {**_valid_repo(name), "result_cards": [_valid_card(f"card-{name}")]}
            for name in sorted(REQUIRED_REPOS)
        ]
    return {
        "schema_version": "independent-repo-result-card/v1",
        "generated_at": "2026-08-06T00:00:00Z",
        "issue_ref": 89,
        "repos": repos,
    }


class TestSchemaFile:
    def test_schema_is_valid_draft7(self) -> None:
        schema = load_schema()
        Draft7Validator.check_schema(schema)

    def test_schema_path_exists(self) -> None:
        assert SCHEMA_PATH.is_file()


class TestRegistryFile:
    def test_registry_file_exists(self) -> None:
        assert REGISTRY_PATH.is_file()

    def test_registry_passes_schema_and_semantics(self) -> None:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        validate_registry_semantics(registry, context="registry_file")

    def test_registry_covers_all_required_repos(self) -> None:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        check_required_repo_coverage(registry)

    def test_load_registry_round_trip(self) -> None:
        registry = load_registry(REGISTRY_PATH)
        assert registry["schema_version"] == "independent-repo-result-card/v1"


class TestSemanticValidation:
    def test_valid_registry_passes(self) -> None:
        validate_registry_semantics(_valid_registry())

    def test_blocked_status_requires_blocker(self) -> None:
        reg = _valid_registry()
        card = reg["repos"][0]["result_cards"][0]
        card["status"] = "blocked"
        card["blocker"] = None
        card["metrics"] = None
        with pytest.raises(ValueError, match="requires a non-empty blocker"):
            validate_registry_semantics(reg)

    def test_non_blocked_status_forbids_blocker(self) -> None:
        reg = _valid_registry()
        card = reg["repos"][0]["result_cards"][0]
        card["status"] = "formal-presentable"
        card["blocker"] = "should not be here"
        with pytest.raises(ValueError, match="must not carry a blocker"):
            validate_registry_semantics(reg)

    def test_formal_presentable_requires_metrics(self) -> None:
        reg = _valid_registry()
        card = reg["repos"][0]["result_cards"][0]
        card["status"] = "formal-presentable"
        card["metrics"] = None
        with pytest.raises(ValueError, match="requires a non-null metrics"):
            validate_registry_semantics(reg)

    def test_formal_presentable_requires_finite_metric(self) -> None:
        reg = _valid_registry()
        card = reg["repos"][0]["result_cards"][0]
        card["metrics"] = {
            "ttft_ms": None,
            "tbt_ms": None,
            "throughput_tps": None,
            "peak_mem_mb": None,
            "error_rate": None,
        }
        with pytest.raises(ValueError, match="requires a non-null metrics"):
            validate_registry_semantics(reg)

    def test_duplicate_repo_name_rejected(self) -> None:
        reg = _valid_registry(
            [_valid_repo("vllm-hust-bidkv"), _valid_repo("vllm-hust-bidkv")]
        )
        with pytest.raises(ValueError, match="duplicate repo_name"):
            validate_registry_semantics(reg)

    def test_duplicate_card_id_rejected(self) -> None:
        reg = _valid_registry(
            [
                {
                    **_valid_repo("vllm-hust-bidkv"),
                    "result_cards": [_valid_card("dup-id"), _valid_card("dup-id")],
                }
            ]
        )
        with pytest.raises(ValueError, match="duplicate card_id"):
            validate_registry_semantics(reg)

    def test_repo_without_cards_rejected(self) -> None:
        reg = _valid_registry([{**_valid_repo("vllm-hust-bidkv"), "result_cards": []}])
        with pytest.raises(ValueError, match="non-empty|>=1 result card"):
            validate_registry_semantics(reg)


class TestRequiredRepoCoverage:
    def test_all_required_repos_present(self) -> None:
        check_required_repo_coverage(_valid_registry())

    def test_missing_repo_fails(self) -> None:
        reg = _valid_registry(
            [_valid_repo(name) for name in sorted(REQUIRED_REPOS)[1:]]
        )
        with pytest.raises(ValueError, match="missing required independent repos"):
            check_required_repo_coverage(reg)

    def test_repo_present_but_no_cards_fails(self) -> None:
        reg = _valid_registry(
            [
                {**_valid_repo(name), "result_cards": []}
                if name == "vllm-hust-bidkv"
                else _valid_repo(name)
                for name in sorted(REQUIRED_REPOS)
            ]
        )
        with pytest.raises(ValueError, match="missing required independent repos"):
            check_required_repo_coverage(reg)
