"""Trend-coverage-compatible benchmark entry producer.

Wraps the legacy ``export_leaderboard_artifacts`` to produce entries that carry
all required trend-coverage fields (T06/T08) and pass the T09 admission
validator.  Every trend parameter is taken as an explicit keyword argument — the
producer never infers coverage class, campaign, comparison, repeat metadata, or
aggregate from filenames, environment variables, or frontend context.

Usage
-----
    from vllm_hust_benchmark.trend_producer import produce_trend_entry

    artifact_path = produce_trend_entry(
        scenario=scenario,
        …                  # same parameters as export_leaderboard_artifacts
        coverage_class="full-matrix",
        campaign_id="full-stack-jul-2026/v1",
        point_role="checkpoint",
        repeat_group="…",
        repeat_index=0,
        trend_status="default",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from vllm_hust_benchmark.leaderboard_export import export_leaderboard_artifacts
from vllm_hust_benchmark.models import ScenarioDefinition
from vllm_hust_benchmark.trend_validator import validate_entries
from vllm_hust_benchmark.workload_config_contract import (
    WORKLOAD_CONFIG_CONTRACT_VERSION,
    is_official_workload_contract_entry,
    validate_explicit_workload_config,
)

# ---------------------------------------------------------------------------
# Trend coverage field names
# ---------------------------------------------------------------------------
TREND_SCHEMA_VERSION = "trend-coverage/v1"
VALID_COVERAGE_CLASSES = frozenset({"full-matrix", "targeted-pair", "experimental"})
VALID_POINT_ROLES: frozenset = frozenset({"baseline", "head", "checkpoint", None})
VALID_TREND_STATUSES = frozenset(
    {"default", "experimental", "blocked", "invalid", "excluded"}
)


# ---------------------------------------------------------------------------
# Parameter validation (fast-fail, before building the entry)
# ---------------------------------------------------------------------------


def _validate_trend_params(
    coverage_class: str | None,
    campaign_id: str | None,
    comparison_id: str | None,
    point_role: str | None,
    repeat_group: str | None,
    repeat_index: int | None,
    canonical_aggregate: Mapping[str, Any] | None,
    trend_status: str,
    trend_reason: str | None,
) -> None:
    """Raise ``ValueError`` if the trend-parameter combination is invalid.

    This mirrors the JSON Schema conditional rules (``leaderboard_trend_v1.schema.json``)
    and the T09 validator's expectations.  It provides a fast-fail check before
    spending time building the base entry.
    """
    # -- If no trend fields are requested, nothing to validate ---------------
    if coverage_class is None:
        return

    if coverage_class not in VALID_COVERAGE_CLASSES:
        raise ValueError(
            f"coverage_class must be one of {sorted(VALID_COVERAGE_CLASSES)}, "
            f"got {coverage_class!r}"
        )
    if trend_status not in VALID_TREND_STATUSES:
        raise ValueError(
            f"trend_status must be one of {sorted(VALID_TREND_STATUSES)}, "
            f"got {trend_status!r}"
        )

    if point_role is not None and point_role not in VALID_POINT_ROLES:
        raise ValueError(
            f"point_role must be one of {sorted(VALID_POINT_ROLES - {None})} or None, "
            f"got {point_role!r}"
        )

    # -- full-matrix ---------------------------------------------------------
    if coverage_class == "full-matrix":
        if not campaign_id:
            raise ValueError("full-matrix requires campaign_id")
        if point_role != "checkpoint":
            raise ValueError(
                f"full-matrix requires point_role='checkpoint', got {point_role!r}"
            )

    # -- targeted-pair -------------------------------------------------------
    if coverage_class == "targeted-pair":
        if not campaign_id:
            raise ValueError("targeted-pair requires campaign_id")
        if not comparison_id:
            raise ValueError("targeted-pair requires comparison_id")
        if point_role not in ("baseline", "head"):
            raise ValueError(
                f"targeted-pair requires point_role='baseline' or 'head', "
                f"got {point_role!r}"
            )

    # -- experimental --------------------------------------------------------
    if coverage_class == "experimental":
        if point_role is not None:
            raise ValueError(
                "experimental entries must have point_role=null (None), "
                f"got {point_role!r}"
            )
        if comparison_id is not None:
            raise ValueError("experimental entries must not have a comparison_id")
        if trend_status not in ("experimental", "invalid", "excluded"):
            raise ValueError(
                f"experimental coverage requires trend_status in "
                f"{{'experimental', 'invalid', 'excluded'}}, got {trend_status!r}"
            )

    # -- repeat fields -------------------------------------------------------
    if repeat_group is not None and repeat_index is None:
        raise ValueError("repeat_index is required when repeat_group is provided")
    if repeat_index is not None and repeat_group is None:
        raise ValueError("repeat_group is required when repeat_index is provided")

    # -- canonical_aggregate required for non-experimental with repeats ------
    if (
        repeat_group is not None
        and coverage_class != "experimental"
        and canonical_aggregate is None
    ):
        raise ValueError(
            f"canonical_aggregate is required for {coverage_class} entries "
            "that carry a repeat_group"
        )

    # -- trend_reason for non-default/non-experimental statuses --------------
    if trend_status in ("blocked", "invalid", "excluded") and not trend_reason:
        raise ValueError(
            f"trend_reason is required when trend_status is {trend_status!r}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def produce_trend_entry(
    # --- original export_leaderboard_artifacts parameters (unchanged) -------
    scenario: ScenarioDefinition,
    metrics_file: Path | None,
    benchmark_result_file: Path | None,
    constraints_file: Path | None,
    same_spec_file: Path | None,
    output_dir: Path,
    artifact_name: str,
    run_id: str,
    engine: str,
    engine_version: str,
    model_name: str,
    model_parameters: str,
    model_precision: str,
    model_quantization: str | None = None,
    hardware_vendor: str = "Huawei",
    hardware_chip_model: str = "910B2",
    chip_count: int = 1,
    node_count: int = 1,
    memory_per_chip_gb: float | None = None,
    total_memory_gb: float | None = None,
    submitter: str = "vllm-hust-team",
    baseline_engine: str = "vllm",
    domestic_chip_class: str = "Ascend-class",
    representative_model_band: str = "7B-13B",
    data_source: str = "vllm-hust-benchmark",
    input_length: int | None = None,
    output_length: int | None = None,
    batch_size: int | None = None,
    concurrent_requests: int | None = None,
    protocol_version: str = "N/A",
    backend_version: str = "N/A",
    core_version: str = "N/A",
    peak_mem_mb: float | None = None,
    git_commit: str | None = None,
    github_user: str | None = None,
    github_commit_url: str | None = None,
    github_repository: str | None = None,
    github_ref: str | None = None,
    github_event_name: str | None = None,
    github_pr_number: int | None = None,
    github_pr_url: str | None = None,
    runtime_python: str | None = None,
    engine_source_repository: str | None = None,
    engine_source_ref: str | None = None,
    engine_source_commit: str | None = None,
    plugin_source_engine: str | None = None,
    plugin_source_repository: str | None = None,
    plugin_source_ref: str | None = None,
    plugin_source_commit: str | None = None,
    spec_path: Path | None = None,
    # --- trend coverage parameters (explicit, never inferred) ---------------
    coverage_class: str | None = None,
    campaign_id: str | None = None,
    comparison_id: str | None = None,
    point_role: str | None = None,
    repeat_group: str | None = None,
    repeat_index: int | None = None,
    canonical_aggregate: Mapping[str, Any] | None = None,
    trend_status: str = "default",
    trend_reason: str | None = None,
    # --- controls -----------------------------------------------------------
    validate: bool = True,
) -> Path:
    """Produce a complete benchmark entry with trend-coverage fields.

    Parameters
    ----------
    scenario, metrics_file, benchmark_result_file, ...
        Same as :func:`~vllm_hust_benchmark.leaderboard_export.export_leaderboard_artifacts`.
    coverage_class
        One of ``"full-matrix"``, ``"targeted-pair"``, ``"experimental"``.
        When ``None`` the entry is produced as a **legacy** entry (no trend
        fields) — this exists only for migration; new entries should always
        provide a value.
    campaign_id, comparison_id, point_role, repeat_group, repeat_index,
    canonical_aggregate, trend_status, trend_reason
        Trend coverage metadata.  Rules match the JSON Schema:
        ``full-matrix`` → requires ``campaign_id``, ``point_role="checkpoint"``.
        ``targeted-pair`` → requires ``campaign_id``, ``comparison_id``,
        ``point_role`` in ``{"baseline", "head"}``.
        ``experimental`` → ``point_role`` must be ``None``, no ``comparison_id``,
        ``trend_status`` in ``{"experimental", "invalid", "excluded"}``.
        ``repeat_group`` requires ``repeat_index``.
        ``non-experimental + repeat_group`` requires ``canonical_aggregate``.
    validate
        If ``True`` (default), validate the final entry through T09's
        :func:`~vllm_hust_benchmark.trend_validator.validate_entries` after
        writing.  Raises ``ValueError`` on validation failure.

    Returns
    -------
    Path
        The path to the written artifact JSON file.

    Raises
    ------
    ValueError
        If the trend-parameter combination is invalid, or if the produced entry
        fails T09 validation (when ``validate=True``).
    """
    # -- 1. Validate trend parameters early ----------------------------------
    _validate_trend_params(
        coverage_class=coverage_class,
        campaign_id=campaign_id,
        comparison_id=comparison_id,
        point_role=point_role,
        repeat_group=repeat_group,
        repeat_index=repeat_index,
        canonical_aggregate=canonical_aggregate,
        trend_status=trend_status,
        trend_reason=trend_reason,
    )

    # -- 2. Generate base entry ----------------------------------------------
    artifact_path, _manifest_path = export_leaderboard_artifacts(
        scenario=scenario,
        metrics_file=metrics_file,
        benchmark_result_file=benchmark_result_file,
        constraints_file=constraints_file,
        same_spec_file=same_spec_file,
        output_dir=output_dir,
        artifact_name=artifact_name,
        run_id=run_id,
        engine=engine,
        engine_version=engine_version,
        model_name=model_name,
        model_parameters=model_parameters,
        model_precision=model_precision,
        model_quantization=model_quantization,
        hardware_vendor=hardware_vendor,
        hardware_chip_model=hardware_chip_model,
        chip_count=chip_count,
        node_count=node_count,
        memory_per_chip_gb=memory_per_chip_gb,
        total_memory_gb=total_memory_gb,
        submitter=submitter,
        baseline_engine=baseline_engine,
        domestic_chip_class=domestic_chip_class,
        representative_model_band=representative_model_band,
        data_source=data_source,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        concurrent_requests=concurrent_requests,
        protocol_version=protocol_version,
        backend_version=backend_version,
        core_version=core_version,
        peak_mem_mb=peak_mem_mb,
        git_commit=git_commit,
        github_user=github_user,
        github_commit_url=github_commit_url,
        github_repository=github_repository,
        github_ref=github_ref,
        github_event_name=github_event_name,
        github_pr_number=github_pr_number,
        github_pr_url=github_pr_url,
        runtime_python=runtime_python,
        engine_source_repository=engine_source_repository,
        engine_source_ref=engine_source_ref,
        engine_source_commit=engine_source_commit,
        plugin_source_engine=plugin_source_engine,
        plugin_source_repository=plugin_source_repository,
        plugin_source_ref=plugin_source_ref,
        plugin_source_commit=plugin_source_commit,
        spec_path=spec_path,
    )

    # -- 3. Read back the written entry --------------------------------------
    entry = json.loads(artifact_path.read_text(encoding="utf-8"))

    # -- 4. Overlay trend coverage fields ------------------------------------
    if coverage_class is not None:
        entry["trend_schema_version"] = TREND_SCHEMA_VERSION
        entry["coverage_class"] = coverage_class
        entry["trend_status"] = trend_status
        if campaign_id is not None:
            entry["campaign_id"] = campaign_id
        if comparison_id is not None:
            entry["comparison_id"] = comparison_id
        if point_role is not None:
            entry["point_role"] = point_role
        if repeat_group is not None:
            entry["repeat_group"] = repeat_group
        if repeat_index is not None:
            entry["repeat_index"] = repeat_index
        if canonical_aggregate is not None:
            entry["canonical_aggregate"] = dict(canonical_aggregate)
        if trend_reason is not None:
            entry["trend_reason"] = trend_reason

    # -- 5. Ensure workload config contract is set for official entries ------
    if is_official_workload_contract_entry(entry):
        entry.setdefault("metadata", {})["workload_config_contract"] = (
            WORKLOAD_CONFIG_CONTRACT_VERSION
        )
        config_errors = validate_explicit_workload_config(entry)
        if config_errors:
            raise ValueError(
                "Workload config contract validation failed: "
                + "; ".join(config_errors)
            )

    # -- 6. Run T09 validation -----------------------------------------------
    if validate:
        report = validate_entries([entry])
        if not report.passed:
            messages = [
                f"[{issue.severity}] {issue.code}: {issue.message}"
                for issue in report.issues
            ]
            raise ValueError(
                "Produced entry failed T09 admission:\n" + "\n".join(messages)
            )

    # -- 7. Write final entry ------------------------------------------------
    artifact_path.write_text(
        json.dumps(entry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def add_trend_fields_to_existing_entry(
    artifact_path: Path,
    *,
    coverage_class: str | None = None,
    campaign_id: str | None = None,
    comparison_id: str | None = None,
    point_role: str | None = None,
    repeat_group: str | None = None,
    repeat_index: int | None = None,
    canonical_aggregate: Mapping[str, Any] | None = None,
    trend_status: str = "default",
    trend_reason: str | None = None,
    validate: bool = True,
) -> Path:
    """Add or update trend-coverage fields on an already-written entry file.

    This is intended for **migration** — converting a legacy entry that was
    produced by the old producer into a trend-compatible entry.  New entries
    should use :func:`produce_trend_entry` instead.

    The entry is read, patched, validated, and written back in-place.
    """
    # Validate params early
    _validate_trend_params(
        coverage_class=coverage_class,
        campaign_id=campaign_id,
        comparison_id=comparison_id,
        point_role=point_role,
        repeat_group=repeat_group,
        repeat_index=repeat_index,
        canonical_aggregate=canonical_aggregate,
        trend_status=trend_status,
        trend_reason=trend_reason,
    )

    entry = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(entry, dict):
        raise ValueError(
            f"{artifact_path}: expected a JSON object, got {type(entry).__name__}"
        )

    if coverage_class is not None:
        entry["trend_schema_version"] = TREND_SCHEMA_VERSION
        entry["coverage_class"] = coverage_class
        entry["trend_status"] = trend_status
        if campaign_id is not None:
            entry["campaign_id"] = campaign_id
        if comparison_id is not None:
            entry["comparison_id"] = comparison_id
        if point_role is not None:
            entry["point_role"] = point_role
        if repeat_group is not None:
            entry["repeat_group"] = repeat_group
        if repeat_index is not None:
            entry["repeat_index"] = repeat_index
        if canonical_aggregate is not None:
            entry["canonical_aggregate"] = dict(canonical_aggregate)
        if trend_reason is not None:
            entry["trend_reason"] = trend_reason

    # Ensure workload config contract marker for official entries
    if is_official_workload_contract_entry(entry):
        entry.setdefault("metadata", {})["workload_config_contract"] = (
            WORKLOAD_CONFIG_CONTRACT_VERSION
        )

    if validate:
        report = validate_entries([entry])
        if not report.passed:
            messages = [
                f"[{issue.severity}] {issue.code}: {issue.message}"
                for issue in report.issues
            ]
            raise ValueError(
                "Patched entry failed T09 admission:\n" + "\n".join(messages)
            )

    artifact_path.write_text(
        json.dumps(entry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return artifact_path
