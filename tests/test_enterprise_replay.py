from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from vllm_hust_benchmark.enterprise_replay import AUTHORIZATION_SCHEMA_VERSION
from vllm_hust_benchmark.enterprise_replay import EnterpriseReplayError
from vllm_hust_benchmark.enterprise_replay import enterprise_case_rows
from vllm_hust_benchmark.enterprise_replay import enterprise_dataset_rows
from vllm_hust_benchmark.enterprise_replay import load_enterprise_dataset_registry
from vllm_hust_benchmark.enterprise_replay import load_enterprise_replay_requests
from vllm_hust_benchmark.enterprise_replay import resolve_enterprise_data_root


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixture_registry(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    rendered = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(rendered, encoding="utf-8")
    asset_bytes = path.read_bytes()
    return {
        "schema_version": "enterprise-dataset-registry/v1",
        "registry_version": "test",
        "effective_from": "2026-08-13",
        "data_policy": {},
        "datasets": [
            {
                "dataset_id": "fixture_dataset",
                "relative_path": path.name,
                "source_format": "wrapped_openai_request_jsonl",
                "source_model_name": "qwen3",
                "record_count": len(rows),
                "sha256": hashlib.sha256(asset_bytes).hexdigest(),
                "workload_family_id": "fixture-family",
                "provenance_class": "enterprise_observed_request",
                "preserved_source_fields": [],
            }
        ],
        "cases": [
            {
                "case_id": "fixture_all",
                "dataset_id": "fixture_dataset",
                "filter_id": None,
                "sampling_unit": "request",
                "replay_order": "source_order",
            }
        ],
    }


def _authorization_file(root: Path, *dataset_ids: str) -> Path:
    path = root / "authorization.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": AUTHORIZATION_SCHEMA_VERSION,
                "authorized_dataset_ids": list(dataset_ids),
            }
        ),
        encoding="utf-8",
    )
    return path


def _wrapped_row(index: int) -> dict[str, object]:
    body = {
        "model": "qwen3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"question {index}"},
                    {"type": "text", "text": "return json"},
                ],
            }
        ],
        "stream": True,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer"}},
        "max_tokens": 32,
    }
    return {
        "model_name": "qwen3",
        "request_type": "ChatCompletionStream",
        "request_body": json.dumps(body),
    }


def test_enterprise_registry_matches_json_schema() -> None:
    schema = json.loads(
        (REPO_ROOT / "schemas" / "enterprise_dataset_registry_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        load_enterprise_dataset_registry()
    )


def test_enterprise_registry_covers_all_delivered_assets_and_migrated_cases() -> None:
    datasets = enterprise_dataset_rows()
    cases = {row["case_id"] for row in enterprise_case_rows()}

    assert len(datasets) == 8
    assert sum(row["record_count"] for row in datasets) == 26500
    assert {
        "enterprise_code_eval_all",
        "enterprise_json_all",
        "enterprise_normal_chat_all",
        "enterprise_long_text",
        "enterprise_prefix_shared",
        "enterprise_reuse_conversation",
        "enterprise_semantic_similar",
        "enterprise_long_prefill",
    } <= cases


def test_data_root_is_never_discovered_implicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_HUST_ENTERPRISE_DATA_ROOT", raising=False)

    with pytest.raises(EnterpriseReplayError, match="data root is required"):
        resolve_enterprise_data_root()


def test_replay_fails_closed_without_dataset_authorization(tmp_path: Path) -> None:
    asset = tmp_path / "fixture.jsonl"
    registry = _fixture_registry(asset, [_wrapped_row(0)])

    with pytest.raises(EnterpriseReplayError, match="authorization file"):
        load_enterprise_replay_requests(
            "fixture_all", data_root=tmp_path, registry=registry
        )


def test_replay_fails_closed_on_checksum_mismatch(tmp_path: Path) -> None:
    asset = tmp_path / "fixture.jsonl"
    registry = _fixture_registry(asset, [_wrapped_row(0)])
    authorization = _authorization_file(tmp_path, "fixture_dataset")
    asset.write_text(asset.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(EnterpriseReplayError, match="checksum mismatch"):
        load_enterprise_replay_requests(
            "fixture_all",
            data_root=tmp_path,
            authorization_file=authorization,
            registry=registry,
        )


def test_replay_is_deterministic_and_preserves_source_model_separately(tmp_path: Path) -> None:
    asset = tmp_path / "fixture.jsonl"
    registry = _fixture_registry(asset, [_wrapped_row(index) for index in range(20)])
    authorization = _authorization_file(tmp_path, "fixture_dataset")

    first = load_enterprise_replay_requests(
        "fixture_all",
        data_root=tmp_path,
        authorization_file=authorization,
        limit=5,
        seed=17,
        registry=registry,
    )
    second = load_enterprise_replay_requests(
        "fixture_all",
        data_root=tmp_path,
        authorization_file=authorization,
        limit=5,
        seed=17,
        registry=registry,
    )

    assert [row.request_id for row in first] == [row.request_id for row in second]
    assert len(first) == 5
    assert all(row.source_model_name == "qwen3" for row in first)
    assert all("question" not in row.request_id for row in first)
    assert all(row.to_openai_payload(served_model="Qwen/Qwen3-32B")["model"] == "Qwen/Qwen3-32B" for row in first)


def test_redacted_record_keeps_shape_but_not_request_body(tmp_path: Path) -> None:
    asset = tmp_path / "fixture.jsonl"
    registry = _fixture_registry(asset, [_wrapped_row(0)])
    authorization = _authorization_file(tmp_path, "fixture_dataset")
    request = load_enterprise_replay_requests(
        "fixture_all",
        data_root=tmp_path,
        authorization_file=authorization,
        registry=registry,
    )[0]

    record = request.redacted_record(served_model="Qwen/Qwen3-32B")

    assert "request_body" not in record
    assert "messages" not in record
    assert record["source_model_name"] == "qwen3"
    assert record["served_model"] == "Qwen/Qwen3-32B"
    assert record["tool_count"] == 1
    assert record["has_response_format"] is True
    assert len(record["request_sha256"]) == 64


def test_group_sampling_preserves_group_order_and_allowlisted_shape_metadata(
    tmp_path: Path,
) -> None:
    rows = []
    for group_id in ("group-a", "group-b", "group-c"):
        for turn in range(2):
            rows.append(
                {
                    "id": f"{group_id}-{turn}",
                    "messages": [{"role": "user", "content": f"question {turn}"}],
                    "group_id": group_id,
                    "session_id": f"session-{group_id}",
                    "created_at_ms": 100 + turn,
                    "prompt_hash": f"hash-{group_id}-{turn}",
                    "prompt_tokens_est": 1000 + turn,
                    "metadata": {
                        "length_bucket": "1k-2k",
                        "output_tokens_target": 64,
                        "private_note": "must-not-leak",
                    },
                }
            )
    asset = tmp_path / "fixture.jsonl"
    rendered = "".join(json.dumps(row) + "\n" for row in rows)
    asset.write_text(rendered, encoding="utf-8")
    registry = {
        "schema_version": "enterprise-dataset-registry/v1",
        "registry_version": "test",
        "effective_from": "2026-08-14",
        "data_policy": {},
        "datasets": [
            {
                "dataset_id": "grouped_fixture",
                "relative_path": asset.name,
                "source_format": "direct_messages_jsonl",
                "source_model_name": "source",
                "record_count": len(rows),
                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                "workload_family_id": "grouped-family",
                "provenance_class": "enterprise_provided_synthetic_or_hybrid",
                "preserved_source_fields": ["group_id", "session_id", "metadata"],
            }
        ],
        "cases": [
            {
                "case_id": "grouped_case",
                "dataset_id": "grouped_fixture",
                "filter_id": None,
                "sampling_unit": "group",
                "replay_order": "group_then_timestamp",
            }
        ],
    }
    authorization = _authorization_file(tmp_path, "grouped_fixture")

    selected = load_enterprise_replay_requests(
        "grouped_case",
        data_root=tmp_path,
        authorization_file=authorization,
        registry=registry,
        limit=4,
        seed=7,
    )

    assert len(selected) == 4
    assert all(selected[index].source_group_id == selected[index + 1].source_group_id for index in (0, 2))
    assert selected[0].source_created_at_ms < selected[1].source_created_at_ms
    record = selected[0].redacted_record(
        served_model="Qwen/Qwen3-32B", generation_overrides={"max_tokens": 64}
    )
    assert record["max_output_tokens"] == 64
    assert record["source_shape_metadata"]["length_bucket"] == "1k-2k"
    assert "private_note" not in record["source_shape_metadata"]
    assert record["source_group_key_sha256"] != selected[0].source_group_id
