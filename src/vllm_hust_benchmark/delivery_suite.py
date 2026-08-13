from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "delivery-suite-registry/v1"
REQUIRED_APPLICATION_SCENARIO_IDS = frozenset(
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
# Kept as an import-compatible alias for the unmerged v1 branch API.
REQUIRED_WORKLOAD_IDS = REQUIRED_APPLICATION_SCENARIO_IDS
REQUIRED_CAPABILITY_TARGET_IDS = frozenset(
    {
        "api_protocol",
        "artifact_startup",
        "open_loop_overload",
        "prefill_decode_balance",
        "scheduler_preemption",
        "kv_reuse_eviction",
        "memory_oom_boundary",
        "structured_decode",
        "multimodal_pipeline",
        "parallelism_topology",
        "fault_cancellation_restart",
        "fairness_soak",
    }
)
VALID_DEPLOYMENT_TIERS = frozenset(
    {"single_node_main", "single_node_conditional", "customer_multinode_extension"}
)
VALID_ASSET_TYPES = frozenset(
    {
        "quality_benchmark",
        "serving_trace",
        "serving_harness",
        "microbenchmark",
        "enterprise_replay",
        "design_reference",
    }
)
PINNED_ARTIFACT_STATUSES = frozenset({"pinned_candidate", "admitted"})


@dataclass(frozen=True)
class DeliverySuiteEntry:
    workload_id: str
    workload_family_ids: tuple[str, ...]
    public_asset_ids: tuple[str, ...]
    model_id: str
    model_revision: str
    deployment_tier: str
    enterprise_case_ids: tuple[str, ...]
    engine_capability_ids: tuple[str, ...]
    primary_metric: str
    minimum_effect_percent: float


def _registry_resource() -> Any:
    return resources.files("vllm_hust_benchmark.data").joinpath(
        "delivery_suite_registry.json"
    )


def _require_non_empty_string(value: object, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"delivery suite {field} must be a non-empty string")
    return normalized


def _require_string_list(
    value: object, *, field: str, allow_empty: bool = False
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise ValueError(f"delivery suite {field} must be {qualifier}")
    normalized = [_require_non_empty_string(item, field=field) for item in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"delivery suite {field} must not contain duplicates")
    return normalized


def _validate_model_artifact(model: dict[str, Any], *, scenario_id: str) -> None:
    model_id = _require_non_empty_string(model.get("repo_id"), field=f"{scenario_id}.model_artifact.repo_id")
    if "/" not in model_id:
        raise ValueError(f"{scenario_id}: model repo_id must be a namespaced repository ID")
    revision = _require_non_empty_string(
        model.get("revision"), field=f"{scenario_id}.model_artifact.revision"
    )
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{scenario_id}: model revision must be a full 40-character commit SHA")
    for field in ("tokenizer_revision", "processor_revision"):
        value = model.get(field)
        if value is not None and not re.fullmatch(r"[0-9a-f]{40}", str(value)):
            raise ValueError(f"{scenario_id}: {field} must be null or a full commit SHA")
    deployment_tier = _require_non_empty_string(
        model.get("deployment_tier"), field=f"{scenario_id}.model_artifact.deployment_tier"
    )
    if deployment_tier not in VALID_DEPLOYMENT_TIERS:
        raise ValueError(f"{scenario_id}: unsupported deployment tier {deployment_tier!r}")
    if deployment_tier == "customer_multinode_extension":
        raise ValueError(f"{scenario_id}: customer multi-node models cannot be main acceptance targets")
    status = _require_non_empty_string(
        model.get("artifact_status"), field=f"{scenario_id}.model_artifact.artifact_status"
    )
    if status not in PINNED_ARTIFACT_STATUSES:
        raise ValueError(f"{scenario_id}: main target artifact must be pinned")
    runtime = model.get("runtime_profile")
    if not isinstance(runtime, dict):
        raise ValueError(f"{scenario_id}: runtime_profile must be an object")
    for field in ("dtype", "tensor_parallel_size", "data_parallel_size", "max_model_len"):
        if field not in runtime:
            raise ValueError(f"{scenario_id}: runtime_profile is missing {field}")
    _require_non_empty_string(model.get("chat_template_source"), field=f"{scenario_id}.model_artifact.chat_template_source")


def _validate_acceptance_policy(policy: object) -> None:
    if not isinstance(policy, dict):
        raise ValueError("delivery suite acceptance_policy must be an object")
    design = policy.get("experimental_design")
    portfolio = policy.get("portfolio_gate")
    baselines = policy.get("causal_baselines")
    guardrails = policy.get("global_guardrails")
    if not all(isinstance(item, dict) for item in (design, portfolio, baselines, guardrails)):
        raise ValueError("acceptance policy sections must be objects")
    if int(design.get("paired_blocks") or 0) < 5:
        raise ValueError("formal acceptance requires at least five paired restart blocks")
    if int(design.get("bootstrap_resamples") or 0) < 1000:
        raise ValueError("formal acceptance requires at least 1000 bootstrap resamples")
    required_baselines = {"upstream_stock", "hust_feature_off", "hust_feature_on", "minus_one_ablation"}
    if set(baselines) != required_baselines:
        raise ValueError("causal_baselines must define stock, feature-off, feature-on and minus-one ablation")
    mandatory = set(
        _require_string_list(
            portfolio.get("mandatory_primary_pass_scenarios"),
            field="acceptance_policy.portfolio_gate.mandatory_primary_pass_scenarios",
        )
    )
    if not mandatory <= REQUIRED_APPLICATION_SCENARIO_IDS:
        raise ValueError("portfolio mandatory scenarios must be application scenarios")
    minimum = int(portfolio.get("minimum_primary_pass_count") or 0)
    if minimum < len(mandatory) or minimum > len(REQUIRED_APPLICATION_SCENARIO_IDS):
        raise ValueError("portfolio minimum_primary_pass_count is inconsistent")
    if float(guardrails.get("maximum_secondary_regression_percent") or 0) <= 0:
        raise ValueError("global guardrail regression allowance must be positive")


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

    _validate_acceptance_policy(payload.get("acceptance_policy"))

    assets = payload.get("evaluation_assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("delivery suite evaluation_assets must be a non-empty array")
    asset_ids: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("evaluation asset entries must be objects")
        asset_id = _require_non_empty_string(asset.get("asset_id"), field="evaluation_asset.asset_id")
        if asset_id in asset_ids:
            raise ValueError(f"duplicate evaluation asset: {asset_id}")
        if asset.get("asset_type") not in VALID_ASSET_TYPES:
            raise ValueError(f"{asset_id}: unsupported asset_type")
        _require_non_empty_string(asset.get("authority_url"), field=f"{asset_id}.authority_url")
        asset_ids.add(asset_id)

    capability_targets = payload.get("engine_capability_targets")
    if not isinstance(capability_targets, list):
        raise ValueError("engine_capability_targets must be an array")
    capability_ids = {
        _require_non_empty_string(item.get("capability_id"), field="capability_id")
        for item in capability_targets
        if isinstance(item, dict)
    }
    if capability_ids != REQUIRED_CAPABILITY_TARGET_IDS:
        missing = sorted(REQUIRED_CAPABILITY_TARGET_IDS - capability_ids)
        extra = sorted(capability_ids - REQUIRED_CAPABILITY_TARGET_IDS)
        raise ValueError(f"engine capability coverage mismatch; missing={missing}, extra={extra}")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("delivery suite entries must be an array")
    scenario_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("delivery suite entries must be objects")
        scenario_id = _require_non_empty_string(
            entry.get("application_scenario_id"), field="application_scenario_id"
        )
        _require_string_list(entry.get("workload_family_ids"), field=f"{scenario_id}.workload_family_ids")
        public_asset_ids = _require_string_list(
            entry.get("public_asset_ids"), field=f"{scenario_id}.public_asset_ids"
        )
        if not set(public_asset_ids) <= asset_ids:
            raise ValueError(f"{scenario_id}: unknown public asset ID")
        enterprise_case_ids = _require_string_list(
            entry.get("enterprise_case_ids"),
            field=f"{scenario_id}.enterprise_case_ids",
            allow_empty=True,
        )
        del enterprise_case_ids
        entry_capabilities = _require_string_list(
            entry.get("engine_capability_ids"), field=f"{scenario_id}.engine_capability_ids"
        )
        if not set(entry_capabilities) <= capability_ids:
            raise ValueError(f"{scenario_id}: unknown engine capability ID")
        model = entry.get("model_artifact")
        if not isinstance(model, dict):
            raise ValueError(f"{scenario_id}: model_artifact must be an object")
        _validate_model_artifact(model, scenario_id=scenario_id)
        primary = entry.get("primary_endpoint")
        quality = entry.get("quality_gate")
        if not isinstance(primary, dict) or not isinstance(quality, dict):
            raise ValueError(f"{scenario_id}: primary_endpoint and quality_gate must be objects")
        _require_non_empty_string(primary.get("metric"), field=f"{scenario_id}.primary_endpoint.metric")
        if float(primary.get("minimum_effect_percent") or 0) <= 0:
            raise ValueError(f"{scenario_id}: primary endpoint effect must be positive")
        margin = quality.get("non_inferiority_margin")
        if margin is None or float(margin) < 0:
            raise ValueError(f"{scenario_id}: quality margin must be non-negative")
        scenario_ids.append(scenario_id)

    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("delivery suite application_scenario_id values must be unique")
    if set(scenario_ids) != REQUIRED_APPLICATION_SCENARIO_IDS:
        missing = sorted(REQUIRED_APPLICATION_SCENARIO_IDS.difference(scenario_ids))
        extra = sorted(set(scenario_ids).difference(REQUIRED_APPLICATION_SCENARIO_IDS))
        raise ValueError(
            f"delivery suite must cover exactly nine application scenarios; missing={missing}, extra={extra}"
        )

    extensions = payload.get("customer_multinode_extensions")
    if not isinstance(extensions, list) or not extensions:
        raise ValueError("customer_multinode_extensions must be a non-empty array")
    for extension in extensions:
        if not isinstance(extension, dict):
            raise ValueError("customer multi-node extensions must be objects")
        _require_non_empty_string(extension.get("model_class"), field="extension.model_class")
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
        model = entry["model_artifact"]
        primary = entry["primary_endpoint"]
        records.append(
            DeliverySuiteEntry(
                workload_id=entry["application_scenario_id"],
                workload_family_ids=tuple(entry["workload_family_ids"]),
                public_asset_ids=tuple(entry["public_asset_ids"]),
                model_id=model["repo_id"],
                model_revision=model["revision"],
                deployment_tier=model["deployment_tier"],
                enterprise_case_ids=tuple(entry["enterprise_case_ids"]),
                engine_capability_ids=tuple(entry["engine_capability_ids"]),
                primary_metric=primary["metric"],
                minimum_effect_percent=float(primary["minimum_effect_percent"]),
            )
        )
    return tuple(records)


def resolve_delivery_workload(workload_id: str) -> dict[str, Any]:
    for entry in load_delivery_suite_registry()["entries"]:
        if entry["application_scenario_id"] == workload_id:
            return dict(entry)
    raise ValueError(f"unknown delivery workload: {workload_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the proposed future delivery suite")
    parser.add_argument("--workload-id", choices=sorted(REQUIRED_APPLICATION_SCENARIO_IDS))
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
