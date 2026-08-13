from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "delivery-suite-registry/v1"
REQUIRED_WORKLOAD_IDS = frozenset(
    {
        "general_qa",
        "code",
        "reasoning",
        "multimodal",
        "kv_reuse",
        "structured_output",
        "long_context",
        "agent",
        "ai4science",
    }
)
VALID_DEPLOYMENT_TIERS = frozenset(
    {"single_node_main", "single_node_conditional", "customer_multinode_extension"}
)


@dataclass(frozen=True)
class DeliverySuiteEntry:
    workload_id: str
    workload_family_ids: tuple[str, ...]
    benchmark_id: str
    model_id: str
    deployment_tier: str
    enterprise_case_ids: tuple[str, ...]
    engine_pressure: tuple[str, ...]
    acceptance_metrics: tuple[str, ...]


def _registry_resource() -> Any:
    return resources.files("vllm_hust_benchmark.data").joinpath(
        "delivery_suite_registry.json"
    )


def _require_non_empty_string(value: object, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"delivery suite {field} must be a non-empty string")
    return normalized


def _require_string_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"delivery suite {field} must be a non-empty array")
    normalized = [_require_non_empty_string(item, field=field) for item in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"delivery suite {field} must not contain duplicates")
    return normalized


def validate_delivery_suite_registry(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"delivery suite schema_version must be {SCHEMA_VERSION!r}")
    _require_non_empty_string(payload.get("registry_version"), field="registry_version")
    academic_contract = payload.get("academic_workload_contract")
    if not isinstance(academic_contract, dict):
        raise ValueError("delivery suite academic_workload_contract must be an object")
    if academic_contract.get("descriptor_schema_version") != "workload-descriptor/v1":
        raise ValueError("delivery suite workload descriptor schema version mismatch")
    if academic_contract.get("family_generator_version") != "family-generator/v1":
        raise ValueError("delivery suite family generator version mismatch")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("delivery suite entries must be an array")

    workload_ids: list[str] = []
    benchmark_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("delivery suite entries must be objects")
        workload_id = _require_non_empty_string(
            entry.get("workload_id"), field="workload_id"
        )
        benchmark = entry.get("benchmark")
        model = entry.get("model")
        if not isinstance(benchmark, dict) or not isinstance(model, dict):
            raise ValueError(f"{workload_id}: benchmark and model must be objects")
        benchmark_id = _require_non_empty_string(
            benchmark.get("id"), field=f"{workload_id}.benchmark.id"
        )
        _require_string_list(
            entry.get("workload_family_ids"),
            field=f"{workload_id}.workload_family_ids",
        )
        _require_non_empty_string(
            benchmark.get("authority_url"),
            field=f"{workload_id}.benchmark.authority_url",
        )
        _require_non_empty_string(model.get("id"), field=f"{workload_id}.model.id")
        deployment_tier = _require_non_empty_string(
            model.get("deployment_tier"),
            field=f"{workload_id}.model.deployment_tier",
        )
        if deployment_tier not in VALID_DEPLOYMENT_TIERS:
            raise ValueError(
                f"{workload_id}: unsupported deployment tier {deployment_tier!r}"
            )
        if deployment_tier == "customer_multinode_extension":
            raise ValueError(
                f"{workload_id}: customer multi-node models cannot be fixed acceptance targets"
            )
        _require_string_list(
            entry.get("engine_pressure"), field=f"{workload_id}.engine_pressure"
        )
        _require_string_list(
            entry.get("acceptance_metrics"),
            field=f"{workload_id}.acceptance_metrics",
        )
        enterprise_case_ids = entry.get("enterprise_case_ids")
        if not isinstance(enterprise_case_ids, list):
            raise ValueError(f"{workload_id}.enterprise_case_ids must be an array")
        for case_id in enterprise_case_ids:
            _require_non_empty_string(
                case_id, field=f"{workload_id}.enterprise_case_ids"
            )
        workload_ids.append(workload_id)
        benchmark_ids.append(benchmark_id)

    if len(workload_ids) != len(set(workload_ids)):
        raise ValueError("delivery suite workload_id values must be unique")
    if set(workload_ids) != REQUIRED_WORKLOAD_IDS:
        missing = sorted(REQUIRED_WORKLOAD_IDS.difference(workload_ids))
        extra = sorted(set(workload_ids).difference(REQUIRED_WORKLOAD_IDS))
        raise ValueError(
            f"delivery suite must cover exactly nine workloads; missing={missing}, extra={extra}"
        )
    if len(benchmark_ids) != len(set(benchmark_ids)):
        raise ValueError(
            "each workload must own one distinct fixed benchmark; arrival traces and "
            "enterprise cases are workload inputs, not additional benchmark identities"
        )

    extensions = payload.get("customer_multinode_extensions")
    if not isinstance(extensions, list) or not extensions:
        raise ValueError("customer_multinode_extensions must be a non-empty array")
    for extension in extensions:
        if not isinstance(extension, dict):
            raise ValueError("customer multi-node extensions must be objects")
        _require_non_empty_string(extension.get("model_id"), field="extension.model_id")
        if extension.get("deployment_tier") != "customer_multinode_extension":
            raise ValueError("customer extension must use customer_multinode_extension tier")


@lru_cache(maxsize=1)
def load_delivery_suite_registry() -> dict[str, Any]:
    with _registry_resource().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("delivery suite registry must be a JSON object")
    validate_delivery_suite_registry(payload)
    return payload


def delivery_suite_entries() -> tuple[DeliverySuiteEntry, ...]:
    records: list[DeliverySuiteEntry] = []
    for entry in load_delivery_suite_registry()["entries"]:
        records.append(
            DeliverySuiteEntry(
                workload_id=entry["workload_id"],
                workload_family_ids=tuple(entry["workload_family_ids"]),
                benchmark_id=entry["benchmark"]["id"],
                model_id=entry["model"]["id"],
                deployment_tier=entry["model"]["deployment_tier"],
                enterprise_case_ids=tuple(entry["enterprise_case_ids"]),
                engine_pressure=tuple(entry["engine_pressure"]),
                acceptance_metrics=tuple(entry["acceptance_metrics"]),
            )
        )
    return tuple(records)


def resolve_delivery_workload(workload_id: str) -> dict[str, Any]:
    for entry in load_delivery_suite_registry()["entries"]:
        if entry["workload_id"] == workload_id:
            return dict(entry)
    raise ValueError(f"unknown delivery workload: {workload_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the fixed delivery suite")
    parser.add_argument("--workload-id", choices=sorted(REQUIRED_WORKLOAD_IDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = (
        resolve_delivery_workload(args.workload_id)
        if args.workload_id
        else load_delivery_suite_registry()
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
