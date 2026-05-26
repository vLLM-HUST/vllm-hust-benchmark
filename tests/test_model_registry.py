from __future__ import annotations

import pytest

from vllm_hust_benchmark.model_registry import load_model_identity_registry
from vllm_hust_benchmark.model_registry import normalize_model_identity_payload
from vllm_hust_benchmark.model_registry import resolve_model_identity
from vllm_hust_benchmark.model_registry import resolve_model_identity_from_payload
from vllm_hust_benchmark.model_registry import validate_model_identity_payload


def test_load_model_identity_registry_contains_seeded_qwen_models() -> None:
    records = load_model_identity_registry()
    canonical_ids = {record.canonical_id for record in records}

    assert "hf:Qwen/Qwen2.5-0.5B-Instruct" in canonical_ids
    assert "hf:Qwen/Qwen2.5-14B-Instruct" in canonical_ids
    assert "hf:Qwen/Qwen2.5-7B-Instruct" in canonical_ids
    assert "hf:meta-llama/Llama-3.1-8B-Instruct" in canonical_ids
    assert "hf:deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" in canonical_ids
    assert "hf:mistralai/Mistral-7B-Instruct-v0.3" in canonical_ids


def test_resolve_model_identity_by_short_alias() -> None:
    identity = resolve_model_identity("Qwen2.5-14B-Instruct")

    assert identity.canonical_id == "hf:Qwen/Qwen2.5-14B-Instruct"
    assert identity.repo_id == "Qwen/Qwen2.5-14B-Instruct"
    assert identity.short_name == "Qwen2.5-14B-Instruct"
    assert identity.display_name == "Qwen2.5-14B-Instruct"


def test_resolve_model_identity_by_hf_cache_path() -> None:
    identity = resolve_model_identity(
        "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/abc123"
    )

    assert identity.canonical_id == "hf:Qwen/Qwen2.5-14B-Instruct"
    assert identity.repo_id == "Qwen/Qwen2.5-14B-Instruct"


def test_resolve_model_identity_falls_back_for_unseeded_repo_id() -> None:
    identity = resolve_model_identity("meta-llama/Llama-3.2-3B-Instruct")

    assert identity.canonical_id == "hf:meta-llama/Llama-3.2-3B-Instruct"
    assert identity.repo_id == "meta-llama/Llama-3.2-3B-Instruct"
    assert identity.short_name == "Llama-3.2-3B-Instruct"
    assert identity.display_name == "Llama-3.2-3B-Instruct"


def test_resolve_model_identity_from_payload_uses_short_alias() -> None:
    identity = resolve_model_identity_from_payload({"name": "Qwen2.5-0.5B-Instruct"})

    assert identity.canonical_id == "hf:Qwen/Qwen2.5-0.5B-Instruct"
    assert identity.repo_id == "Qwen/Qwen2.5-0.5B-Instruct"


def test_normalize_model_identity_payload_sets_required_contract_fields() -> None:
    payload = normalize_model_identity_payload(
        {
            "name": "Llama-3.1-8B-Instruct",
            "parameters": "8B",
            "precision": "BF16",
        }
    )

    assert payload["canonical_id"] == "hf:meta-llama/Llama-3.1-8B-Instruct"
    assert payload["repo_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert payload["short_name"] == "Llama-3.1-8B-Instruct"
    assert payload["display_name"] == "Llama-3.1-8B-Instruct"
    assert payload["name"] == "meta-llama/Llama-3.1-8B-Instruct"


def test_validate_model_identity_payload_rejects_name_repo_id_mismatch() -> None:
    with pytest.raises(ValueError, match="model.name equal to model.repo_id"):
        validate_model_identity_payload(
            {
                "canonical_id": "hf:Qwen/Qwen2.5-14B-Instruct",
                "repo_id": "Qwen/Qwen2.5-14B-Instruct",
                "short_name": "Qwen2.5-14B-Instruct",
                "display_name": "Qwen2.5-14B-Instruct",
                "name": "Qwen2.5-14B-Instruct",
            }
        )


def test_resolve_model_identity_rejects_unknown_short_alias() -> None:
    with pytest.raises(ValueError, match="unknown short model alias"):
        resolve_model_identity("Llama-3.3-70B-Instruct")