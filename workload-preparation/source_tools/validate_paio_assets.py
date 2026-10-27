"""Read-only, aggregate-only validation of the eight PAIO assets.

No request text, source IDs, group names, matches or per-record hashes are emitted.
This validates request inputs, not source representativeness or runtime outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


VALIDATOR_VERSION = "paio-validator/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
SENSITIVE_PATTERNS = {
    "email_like": re.compile(
        r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,24}"
    ),
    "ipv4_like": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "cn_mobile_like": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "credential_like": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})\b"),
}
STRUCTURE_FIELDS = ("session_id", "conversation_id", "group_id")
TIME_FIELDS = ("timestamp", "created_at_ms")
SAFE_FORMATS = {"json_object", "json_schema", "text"}
SAFE_SCHEMA_FIELDS = {
    "model_name",
    "request_body",
    "request_type",
    "id",
    "workload_type",
    "group_id",
    "topic",
    "messages",
    "prompt",
    "prompt_tokens_est",
    "prompt_hash",
    "created_at_ms",
    "session_id",
    "conversation_id",
    "metadata",
    "request_id",
    "query_text",
    "model",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stream",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "stream_options",
    "response_format",
    "tools",
    "tool_choice",
}


def reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON literal")


def parse_json(value: str) -> Any:
    return json.loads(value, parse_constant=reject_constant)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    """Deterministic JSON identity, not RFC 8785 or tokenizer identity."""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def distribution(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def quantile(p: float) -> int | float:
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 6),
        "quantile_method": "nearest_rank",
    }


def numeric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from strings(nested)


def validate_asset(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    path = (root / spec["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("asset path must stay under the explicit data root")
    report: dict[str, Any] = {
        **spec,
        "provider": "PAIO",
        "real_derived_or_synthetic": "enterprise_provided_real_derived_or_constructed_not_full_production_trace",
        "construction_method": "see asset known_biases; original provider generation procedure not independently reproduced",
        "authorization_and_license": {
            "local_analysis": "requested_by_user",
            "redistribution": "not_documented",
            "license": "not_supplied",
        },
        "model_or_tokenizer_dependency": "content token counts require a pinned tokenizer and explicit rendering convention",
        "prompt_token_distribution": {
            "status": "NOT_COMPUTED",
            "reason": "no frozen tokenizer selected for this structural audit",
        },
        "runtime_trace_availability": "not_present_as_independent_runtime_records",
        "quality_oracle_availability": "no_independent_gold_or_scorer_supplied_with_these_files",
    }
    if not path.is_file():
        return {**report, "admission_status": "MISSING_ASSET"}
    source_hash = hashlib.sha256()
    counts: Counter[str] = Counter()
    outer_types: dict[str, Counter[str]] = defaultdict(Counter)
    body_types: dict[str, Counter[str]] = defaultdict(Counter)
    sensitive = Counter({name: 0 for name in SENSITIVE_PATTERNS})
    prompt_chars: list[int] = []
    message_lengths: list[int] = []
    estimates: list[int | float] = []
    output_targets: list[int | float] = []
    time_values: dict[str, list[int | float]] = {name: [] for name in TIME_FIELDS}
    groups: dict[str, Counter[str]] = {name: Counter() for name in STRUCTURE_FIELDS}
    identities: set[str] = set()
    prompts: set[str] = set()
    formats: Counter[str] = Counter()
    required = (
        ["request_body", "model_name", "request_type"]
        if spec["format"] == "wrapped_request"
        else ["messages", "prompt", "prompt_hash"]
    )
    with path.open("rb") as stream:
        for line_index, raw in enumerate(stream):
            source_hash.update(raw)
            counts["physical_lines"] += 1
            counts["byte_size"] += len(raw)
            if not raw.strip():
                counts["blank_line_count"] += 1
                continue
            counts["record_count"] += 1
            try:
                text = raw.decode("utf-8-sig" if line_index == 0 else "utf-8")
                outer = parse_json(text)
            except (UnicodeDecodeError, ValueError):
                counts["outer_parse_error_count"] += 1
                continue
            if not isinstance(outer, dict):
                counts["schema_error_count"] += 1
                continue
            for key, value in outer.items():
                # Only schema field names are retained; values never leave the process.
                outer_types[key if key in SAFE_SCHEMA_FIELDS else "<other_field>"][
                    type(value).__name__
                ] += 1
            if any(key not in outer for key in required):
                counts["schema_error_count"] += 1
                continue
            body = outer
            if spec["format"] == "wrapped_request":
                try:
                    body = (
                        parse_json(outer["request_body"])
                        if isinstance(outer["request_body"], str)
                        else outer["request_body"]
                    )
                except (TypeError, ValueError):
                    counts["inner_parse_error_count"] += 1
                    continue
            if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
                counts["schema_error_count"] += 1
                continue
            messages = body["messages"]
            if not messages or any(
                not isinstance(m, dict)
                or "content" not in m
                or not isinstance(m.get("role"), str)
                for m in messages
            ):
                counts["schema_error_count"] += 1
                continue
            for key, value in body.items():
                body_types[key if key in SAFE_SCHEMA_FIELDS else "<other_field>"][
                    type(value).__name__
                ] += 1
            message_lengths.append(len(messages))
            counts["multi_message_request_count"] += len(messages) > 1
            counts["history_bearing_request_count"] += any(
                m.get("role") == "assistant" for m in messages
            )
            counts["tools_present_count"] += bool(body.get("tools"))
            counts["stream_true_count"] += body.get("stream") is True
            prompt = (
                outer.get("prompt") if spec["format"] != "wrapped_request" else None
            )
            if not isinstance(prompt, str):
                prompt = "\n\n".join(
                    f"{m.get('role', '')}: " + "\n".join(strings(m.get("content")))
                    for m in messages
                )
            prompt_chars.append(len(prompt))
            try:
                prompt_identity = digest(prompt.encode("utf-8"))
            except UnicodeEncodeError:
                counts["schema_error_count"] += 1
                continue
            counts["duplicate_prompt_count"] += prompt_identity in prompts
            prompts.add(prompt_identity)
            identity_view = (
                body
                if spec["format"] == "wrapped_request"
                else {"prompt": prompt, "messages": messages}
            )
            try:
                identity = digest(canonical_bytes(identity_view))
            except (ValueError, TypeError, UnicodeEncodeError):
                counts["schema_error_count"] += 1
                continue
            counts["duplicate_request_count"] += identity in identities
            identities.add(identity)
            if isinstance(outer.get("prompt_hash"), str):
                counts["prompt_hash_checked_count"] += 1
                counts["prompt_hash_mismatch_count"] += (
                    outer["prompt_hash"] != prompt_identity[:16]
                )
            estimate = outer.get("prompt_tokens_est")
            if numeric(estimate):
                estimates.append(estimate)
            metadata = outer.get("metadata") or {}
            metadata = metadata if isinstance(metadata, dict) else {}
            output = body.get(
                "max_tokens",
                body.get("max_completion_tokens", metadata.get("output_tokens_target")),
            )
            if numeric(output):
                output_targets.append(output)
            for name in STRUCTURE_FIELDS:
                if outer.get(name) is not None:
                    groups[name][digest(canonical_bytes(outer[name]))] += 1
            for name in TIME_FIELDS:
                if numeric(outer.get(name)):
                    time_values[name].append(outer[name])
            response_format = body.get("response_format")
            if isinstance(response_format, dict):
                kind = response_format.get("type")
                formats[kind if kind in SAFE_FORMATS else "other"] += 1
            scan_parts = list(
                strings(
                    body
                    if spec["format"] == "wrapped_request"
                    else {"prompt": prompt, "messages": messages}
                )
            )
            for name, pattern in SENSITIVE_PATTERNS.items():
                sensitive[name] += any(
                    pattern.search(part) is not None for part in scan_parts
                )
            counts["valid_record_count"] += 1
    for name in (
        "record_count",
        "valid_record_count",
        "physical_lines",
        "byte_size",
        "blank_line_count",
        "outer_parse_error_count",
        "inner_parse_error_count",
        "schema_error_count",
        "duplicate_request_count",
        "duplicate_prompt_count",
        "prompt_hash_checked_count",
        "prompt_hash_mismatch_count",
        "multi_message_request_count",
        "history_bearing_request_count",
        "tools_present_count",
        "stream_true_count",
    ):
        counts.setdefault(name, 0)
    errors = (
        counts["outer_parse_error_count"]
        + counts["inner_parse_error_count"]
        + counts["schema_error_count"]
    )
    return {
        **report,
        **dict(counts),
        "sha256": source_hash.hexdigest(),
        "parse_error_count": counts["outer_parse_error_count"]
        + counts["inner_parse_error_count"],
        "expected_count_matches": counts["record_count"] == spec["expected_count"],
        "schema": {"outer": dict(outer_types), "request_view": dict(body_types)},
        "required_fields": required,
        "duplicate_rate": counts["duplicate_request_count"]
        / max(counts["valid_record_count"], 1),
        "duplicate_definition": "duplicates after the first; canonical JSON of full inner request (v1) or prompt+messages (v2/longtext), without record IDs",
        "sensitive_information_scan": {
            "status": "HEURISTIC_REVIEW_SIGNALS_NOT_CONFIRMED_PII",
            "record_match_counts": dict(sensitive),
            "patterns": {name: p.pattern for name, p in SENSITIVE_PATTERNS.items()},
        },
        "prompt_character_distribution": distribution(prompt_chars),
        "message_count_distribution": distribution(message_lengths),
        "supplied_prompt_token_estimate_distribution": distribution(estimates),
        "output_target_distribution": distribution(output_targets),
        "response_format_counts": dict(formats),
        "session_or_group_structure": {
            name: {
                "records_with_field": sum(values.values()),
                "unique_groups": len(values),
                "group_size_distribution": distribution(list(values.values())),
            }
            for name, values in groups.items()
        },
        "timestamp_availability": {
            name: {
                "distribution": distribution(values),
                "adjacent_decrease_count": sum(
                    b < a for a, b in zip(values, values[1:])
                ),
                "interpretation": "construction_time_not_production_arrival"
                if name == "created_at_ms"
                else "not_established_as_production_trace",
            }
            for name, values in time_values.items()
        },
        "admission_status": "PARSE_VALID_LOCAL_ANALYSIS_WITH_LIMITS"
        if errors == 0
        and counts["prompt_hash_mismatch_count"] == 0
        and counts["record_count"] == spec["expected_count"]
        else "REVIEW_REQUIRED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "data/catalogs/dataset_assets_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("output exists; choose a new path or explicitly pass --overwrite")
    catalog_bytes = args.catalog.read_bytes()
    catalog = json.loads(catalog_bytes)
    ids = [a["dataset_id"] for a in catalog["assets"]]
    if len(ids) != len(set(ids)):
        parser.error("dataset IDs must be unique")
    reports = [validate_asset(args.data_root, asset) for asset in catalog["assets"]]
    result = {
        "validator_version": VALIDATOR_VERSION,
        "validator_sha256": digest(Path(__file__).read_bytes()),
        "catalog_sha256": digest(catalog_bytes),
        "scope": "input validation only; not runtime evidence, exact tokenization, privacy certification or redistribution authorization",
        "assets": reports,
        "asset_count": len(reports),
        "record_count": sum(r.get("record_count", 0) for r in reports),
        "all_assets_parse_valid": all(
            r["admission_status"] == "PARSE_VALID_LOCAL_ANALYSIS_WITH_LIMITS"
            for r in reports
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "asset_count": result["asset_count"],
                "record_count": result["record_count"],
                "all_assets_parse_valid": result["all_assets_parse_valid"],
            }
        )
    )
    if not result["all_assets_parse_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
