"""Schema and cross-entry admission checks for trend coverage data.

The JSON schema owns shape and single-entry conditional requirements.  This
module owns the checks that need the complete candidate set, such as pair
matching and repeat-series completeness.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft7Validator

from vllm_hust_benchmark.workload_config_contract import (
    requires_workload_config_contract,
    validate_explicit_workload_config,
)


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "leaderboard_trend_v1.schema.json"
FORMAL_COVERAGE = {"full-matrix", "targeted-pair"}
RETIRED_MODEL_TOKENS = {"qwen3-8b", "qwen/qwen3-8b"}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    entry_id: str = "<missing-entry-id>"
    severity: str = "error"


@dataclass(frozen=True)
class AdmissionDecision:
    entry_id: str
    status: str
    reason: str
    issues: tuple[ValidationIssue, ...] = ()


@dataclass
class ValidationReport:
    decisions: list[AdmissionDecision] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    sanitized_entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def _entry_id(entry: Mapping[str, Any]) -> str:
    return str(entry.get("entry_id") or "<missing-entry-id>")


def _nested(entry: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = entry.get(key)
    return value if isinstance(value, Mapping) else {}


def _is_retired(entry: Mapping[str, Any]) -> bool:
    model = _nested(entry, "model")
    names = {
        str(model.get("name") or "").lower(),
        str(model.get("repo_id") or "").lower(),
        str(model.get("short_name") or "").lower(),
    }
    precision = str(model.get("precision") or "").lower()
    return bool(names & RETIRED_MODEL_TOKENS) and precision == "bf16"


def _is_w8a8(entry: Mapping[str, Any]) -> bool:
    model = _nested(entry, "model")
    values = [model.get("name"), model.get("short_name"), model.get("precision")]
    return any("w8a8" in str(value or "").lower() or str(value or "").lower() == "int8" for value in values)


def check_invalid_metrics(entry: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], bool]:
    """Return a sanitized metric copy, reasons, and whether a critical metric is invalid."""
    metrics = dict(_nested(entry, "metrics"))
    workload_name = str(_nested(entry, "workload").get("name") or "").lower()
    reasons: list[str] = []
    critical = False

    throughput = metrics.get("throughput_tps")
    if throughput is None:
        reasons.append("throughput_tps is null; latency workloads may omit it, other workloads require it")
        critical = "latency" not in workload_name
    elif not isinstance(throughput, (int, float)) or throughput <= 0:
        metrics["throughput_tps"] = None
        reasons.append("throughput_tps must be > 0; set it to null and retain the raw artifact")
        critical = "latency" not in workload_name

    ttft = metrics.get("ttft_ms")
    if ttft is None and "throughput" not in workload_name:
        reasons.append("ttft_ms is required for latency and online workloads")
        critical = True
    elif ttft is not None and (not isinstance(ttft, (int, float)) or ttft <= 0):
        metrics["ttft_ms"] = None
        reasons.append("ttft_ms must be > 0 for this workload; set it to null")
        critical = True

    error_rate = metrics.get("error_rate")
    if error_rate is not None and (not isinstance(error_rate, (int, float)) or not 0 <= error_rate <= 1):
        metrics["error_rate"] = None
        reasons.append("error_rate must be within [0, 1]; set it to null")
        critical = True
    elif isinstance(error_rate, (int, float)) and error_rate > 0.5:
        reasons.append("error_rate is above 0.5; downgrade to experimental until investigated")

    for name in ("peak_mem_mb", "tbt_ms", "tpot_ms"):
        value = metrics.get(name)
        if isinstance(value, (int, float)) and value < 0:
            metrics[name] = None
            reasons.append(f"{name} must not be negative; set it to null")

    return metrics, reasons, critical


def _schema_issues(entry: Mapping[str, Any], validator: Draft7Validator) -> list[ValidationIssue]:
    return [
        ValidationIssue("SCHEMA_INVALID", error.message, _entry_id(entry))
        for error in sorted(validator.iter_errors(entry), key=lambda item: list(item.path))
    ]


def _local_decision(entry: dict[str, Any], validator: Draft7Validator) -> AdmissionDecision:
    entry_id = _entry_id(entry)
    schema_issues = _schema_issues(entry, validator)
    if schema_issues:
        return AdmissionDecision(entry_id, "invalid", "Schema validation failed; fix the listed fields", tuple(schema_issues))

    if not entry.get("trend_schema_version") or not entry.get("coverage_class"):
        issue = ValidationIssue("LEGACY_NOT_ADMITTED", "Legacy entry has no trend coverage fields; migrate it before formal admission", entry_id, "warning")
        return AdmissionDecision(entry_id, "excluded", "Legacy entry retained for provenance only", (issue,))

    contract_errors: list[str] = []
    metadata = _nested(entry, "metadata")
    if requires_workload_config_contract(entry) or metadata.get("workload_config_contract"):
        contract_errors = validate_explicit_workload_config(entry)
    if contract_errors:
        issues = tuple(ValidationIssue("EFFECTIVE_CONFIG_INVALID", message, entry_id, "warning") for message in contract_errors)
        return AdmissionDecision(entry_id, "blocked", "Effective workload config is incomplete or inconsistent", issues)

    metrics, metric_reasons, critical_invalid = check_invalid_metrics(entry)
    entry["metrics"] = metrics
    if critical_invalid:
        issues = tuple(ValidationIssue("INVALID_METRIC", reason, entry_id) for reason in metric_reasons)
        return AdmissionDecision(entry_id, "invalid", "; ".join(metric_reasons), issues)

    coverage = entry["coverage_class"]
    if _is_retired(entry):
        reason = "Retired Qwen3-8B/BF16 record; exclude from all published trend views"
        return AdmissionDecision(entry_id, "excluded", reason)
    if metric_reasons and any("throughput_tps" in reason for reason in metric_reasons) and "latency" in str(_nested(entry, "workload").get("name") or "").lower():
        issue = ValidationIssue("LATENCY_THROUGHPUT_NOT_APPLICABLE", "random-latency may omit throughput_tps; keep the sanitized metric null and admit only as blocked", entry_id, "warning")
        return AdmissionDecision(entry_id, "blocked", issue.message, (issue,))
    if coverage == "experimental":
        reason = "W8A8/INT8 is outside the formal support matrix" if _is_w8a8(entry) else (entry.get("trend_reason") or "Experimental entry")
        return AdmissionDecision(entry_id, "experimental", reason)
    if metric_reasons:
        return AdmissionDecision(entry_id, "experimental", "; ".join(metric_reasons))
    return AdmissionDecision(entry_id, "pending", "Awaiting cross-entry coverage checks")


def _pair_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    model = _nested(entry, "model")
    hardware = _nested(entry, "hardware")
    workload = _nested(entry, "workload")
    return (
        model.get("canonical_id") or model.get("repo_id") or model.get("name"),
        hardware.get("vendor"), hardware.get("chip_model"), hardware.get("chip_count"),
        model.get("precision"), workload.get("name"), workload.get("input_length"), workload.get("output_length"),
        entry.get("config_type"),
    )


def _cross_entry_decision(entry: dict[str, Any], decision: AdmissionDecision, entries: list[dict[str, Any]], local: dict[str, AdmissionDecision]) -> AdmissionDecision:
    if decision.status != "pending":
        return decision
    entry_id = _entry_id(entry)
    coverage = entry.get("coverage_class")
    if coverage == "full-matrix":
        if entry.get("point_role") != "checkpoint" or not entry.get("campaign_id"):
            issue = ValidationIssue("MATRIX_METADATA_MISSING", "full-matrix requires campaign_id and point_role=checkpoint", entry_id, "warning")
            return AdmissionDecision(entry_id, "blocked", issue.message, (issue,))
        group = entry.get("repeat_group")
        if not group:
            return AdmissionDecision(entry_id, "experimental", "Single full-matrix run has no repeat_group")
        series = [candidate for candidate in entries if candidate.get("repeat_group") == group]
        indices = [candidate.get("repeat_index") for candidate in series]
        if len(indices) != len(set(indices)) or sorted(indices) != list(range(len(indices))):
            issue = ValidationIssue("MATRIX_REPEAT_INDEX_INVALID", "repeat_index values must be unique and contiguous from 0", entry_id, "warning")
            return AdmissionDecision(entry_id, "blocked", issue.message, (issue,))
        aggregate = entry.get("canonical_aggregate") or {}
        if len(series) > 1 and aggregate.get("count") != len(series):
            issue = ValidationIssue("MATRIX_AGGREGATE_MISMATCH", f"canonical_aggregate.count={aggregate.get('count')!r} does not match {len(series)} raw repetitions", entry_id, "warning")
            return AdmissionDecision(entry_id, "blocked", issue.message, (issue,))
        if aggregate.get("count", 0) < 3:
            return AdmissionDecision(entry_id, "experimental", f"Insufficient repetitions: got {aggregate.get('count')}, need at least 3")
        return AdmissionDecision(entry_id, "default", "Formal full-matrix series passed admission")

    if coverage == "targeted-pair":
        if not entry.get("repeat_group") or not entry.get("canonical_aggregate"):
            issue = ValidationIssue("PAIR_AGGREGATE_MISSING", "targeted-pair requires repeat_group and canonical_aggregate before formal admission", entry_id, "warning")
            return AdmissionDecision(entry_id, "blocked", issue.message, (issue,))
        if (entry.get("canonical_aggregate") or {}).get("count", 0) < 3:
            return AdmissionDecision(entry_id, "experimental", "Insufficient repetitions: targeted-pair needs at least 3 samples per side")
        comparison_id = entry.get("comparison_id")
        role = entry.get("point_role")
        counterpart = next((candidate for candidate in entries if candidate.get("comparison_id") == comparison_id and candidate.get("point_role") != role), None)
        if counterpart is None:
            issue = ValidationIssue("PAIR_HALF_MISSING", f"No comparable {('head' if role == 'baseline' else 'baseline')} entry for comparison_id={comparison_id!r}", entry_id, "warning")
            return AdmissionDecision(entry_id, "blocked", issue.message, (issue,))
        if _pair_key(entry) != _pair_key(counterpart):
            issue = ValidationIssue("PAIR_NOT_COMPARABLE", "Pair dimensions differ; align model, hardware, precision, workload, and topology", entry_id, "warning")
            return AdmissionDecision(entry_id, "blocked", issue.message, (issue,))
        counterpart_decision = local.get(_entry_id(counterpart))
        if counterpart_decision is None or counterpart_decision.status != "pending":
            status = counterpart_decision.status if counterpart_decision else "unknown"
            issue = ValidationIssue("PAIR_COUNTERPART_NOT_ADMITTED", f"Counterpart status is {status}; resolve it before publishing the pair", entry_id, "warning")
            return AdmissionDecision(entry_id, "blocked", issue.message, (issue,))
        return AdmissionDecision(entry_id, "default", f"Formal targeted-pair {role} passed admission")
    return decision


def validate_entries(entries: Iterable[Mapping[str, Any]], *, schema_path: Path | None = None) -> ValidationReport:
    """Validate and classify a complete candidate set without mutating input entries."""
    copied = [copy.deepcopy(dict(entry)) for entry in entries]
    schema_file = schema_path or SCHEMA_PATH
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)
    report = ValidationReport(sanitized_entries=copied)
    local = {_entry_id(entry): _local_decision(entry, validator) for entry in copied}
    report.decisions = [_cross_entry_decision(entry, local[_entry_id(entry)], copied, local) for entry in copied]
    report.issues = [issue for decision in report.decisions for issue in decision.issues]
    return report


def load_json_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise ValueError(f"{path}: expected an object or an array of objects")
