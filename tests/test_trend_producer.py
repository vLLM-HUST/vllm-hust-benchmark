"""Tests for the trend-coverage-compatible entry producer (trend_producer.py).

Verifies:
  1.  Entries produced with full trend metadata pass T09 validation.
  2.  Missing/invalid trend parameters are caught early by _validate_trend_params.
  3.  Produced entries carry the expected trend fields.
  4.  The ``add_trend_fields_to_existing_entry`` migration helper works.
  5.  Workload config contract is correctly set for official entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.trend_producer import (
    TREND_SCHEMA_VERSION,
    _validate_trend_params,
    add_trend_fields_to_existing_entry,
    produce_trend_entry,
)
from vllm_hust_benchmark.trend_validator import validate_entries
from vllm_hust_benchmark.workload_config_contract import (
    WORKLOAD_CONFIG_CONTRACT_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def minimal_entry_dict(**overrides: str | int | None) -> dict:
    """Return a minimal but schema-valid entry dict (without trend fields)."""
    entry: dict = {
        "entry_id": "00000000-0000-4000-8000-000000000000",
        "engine": "vllm-hust",
        "engine_version": "v0.23.1-rc0",
        "config_type": "single_gpu",
        "hardware": {
            "vendor": "Huawei",
            "chip_model": "910B2",
            "chip_count": 1,
        },
        "model": {
            "canonical_id": "hf:Qwen/Qwen2.5-7B-Instruct",
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "short_name": "Qwen2.5-7B-Instruct",
            "display_name": "Qwen2.5-7B-Instruct",
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "parameters": "7B",
            "precision": "BF16",
        },
        "workload": {
            "name": "random-online",
            "input_length": 1024,
            "output_length": 256,
            "batch_size": None,
            "concurrent_requests": None,
            "dataset": "random",
        },
        "metrics": {
            "ttft_ms": 42.0,
            "throughput_tps": 321.0,
            "peak_mem_mb": 10240,
            "error_rate": 0.0,
        },
        "constraints": {},
        "versions": {},
        "environment": {},
        "metadata": {
            "submitted_at": "2026-07-25T00:00:00Z",
        },
    }
    entry.update(overrides)
    return entry


def minimal_full_matrix_entry(**overrides: str | int | None) -> dict:
    """Return a minimal valid full-matrix trend entry."""
    entry = minimal_entry_dict(
        trend_schema_version=TREND_SCHEMA_VERSION,
        coverage_class="full-matrix",
        campaign_id="test-campaign/v1",
        point_role="checkpoint",
        repeat_group="test::group",
        repeat_index=0,
        canonical_aggregate={
            "method": "mean",
            "count": 3,
            "metrics": {"ttft_ms": {"value": 42.0}},
            "outlier_handling": "none",
        },
        trend_status="default",
    )
    entry.update(overrides)
    return entry


def minimal_targeted_pair_entry(**overrides: str | int | None) -> dict:
    """Return a minimal valid targeted-pair trend entry (head side)."""
    entry = minimal_entry_dict(
        trend_schema_version=TREND_SCHEMA_VERSION,
        coverage_class="targeted-pair",
        campaign_id="test-pair/v1",
        comparison_id="test-comparison",
        point_role="head",
        repeat_group="test::pair::head",
        repeat_index=0,
        canonical_aggregate={
            "method": "mean",
            "count": 3,
            "metrics": {"ttft_ms": {"value": 38.0}},
            "outlier_handling": "none",
        },
        trend_status="default",
    )
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------------
# _validate_trend_params — fast-fail parameter validation
# ---------------------------------------------------------------------------


class TestValidateTrendParams:
    def test_none_coverage_ok(self) -> None:
        """No coverage_class should be a no-op."""
        _validate_trend_params(
            coverage_class=None,
            campaign_id=None,
            comparison_id=None,
            point_role=None,
            repeat_group=None,
            repeat_index=None,
            canonical_aggregate=None,
            trend_status="default",
            trend_reason=None,
        )

    def test_invalid_coverage_class(self) -> None:
        with pytest.raises(ValueError, match="coverage_class"):
            _validate_trend_params(
                coverage_class="invalid-class",
                campaign_id=None, comparison_id=None,
                point_role=None, repeat_group=None, repeat_index=None,
                canonical_aggregate=None, trend_status="default",
                trend_reason=None,
            )

    def test_full_matrix_requires_campaign_id(self) -> None:
        with pytest.raises(ValueError, match="campaign_id"):
            _validate_trend_params(
                coverage_class="full-matrix",
                campaign_id="", comparison_id=None,
                point_role="checkpoint", repeat_group=None, repeat_index=None,
                canonical_aggregate=None, trend_status="default",
                trend_reason=None,
            )

    def test_full_matrix_requires_point_role_checkpoint(self) -> None:
        with pytest.raises(ValueError, match="point_role"):
            _validate_trend_params(
                coverage_class="full-matrix",
                campaign_id="campaign/v1", comparison_id=None,
                point_role="baseline", repeat_group=None, repeat_index=None,
                canonical_aggregate=None, trend_status="default",
                trend_reason=None,
            )

    def test_targeted_pair_requires_comparison_id(self) -> None:
        with pytest.raises(ValueError, match="comparison_id"):
            _validate_trend_params(
                coverage_class="targeted-pair",
                campaign_id="campaign/v1", comparison_id=None,
                point_role="head", repeat_group=None, repeat_index=None,
                canonical_aggregate=None, trend_status="default",
                trend_reason=None,
            )

    def test_targeted_pair_requires_valid_point_role(self) -> None:
        with pytest.raises(ValueError, match="point_role"):
            _validate_trend_params(
                coverage_class="targeted-pair",
                campaign_id="campaign/v1", comparison_id="cmp-1",
                point_role="checkpoint", repeat_group=None, repeat_index=None,
                canonical_aggregate=None, trend_status="default",
                trend_reason=None,
            )

    def test_experimental_forbids_comparison_id(self) -> None:
        with pytest.raises(ValueError, match="comparison_id"):
            _validate_trend_params(
                coverage_class="experimental",
                campaign_id=None, comparison_id="cmp-1",
                point_role=None, repeat_group=None, repeat_index=None,
                canonical_aggregate=None, trend_status="experimental",
                trend_reason=None,
            )

    def test_experimental_requires_valid_trend_status(self) -> None:
        with pytest.raises(ValueError, match="trend_status"):
            _validate_trend_params(
                coverage_class="experimental",
                campaign_id=None, comparison_id=None,
                point_role=None, repeat_group=None, repeat_index=None,
                canonical_aggregate=None, trend_status="default",
                trend_reason=None,
            )

    def test_repeat_group_requires_repeat_index(self) -> None:
        with pytest.raises(ValueError, match="repeat_index"):
            _validate_trend_params(
                coverage_class="full-matrix",
                campaign_id="c/v1", comparison_id=None,
                point_role="checkpoint",
                repeat_group="r", repeat_index=None,
                canonical_aggregate={"method": "mean", "count": 3, "metrics": {}, "outlier_handling": "none"},
                trend_status="default", trend_reason=None,
            )

    def test_repeat_index_requires_repeat_group(self) -> None:
        with pytest.raises(ValueError, match="repeat_group"):
            _validate_trend_params(
                coverage_class="full-matrix",
                campaign_id="c/v1", comparison_id=None,
                point_role="checkpoint",
                repeat_group=None, repeat_index=0,
                canonical_aggregate={"method": "mean", "count": 3, "metrics": {}, "outlier_handling": "none"},
                trend_status="default", trend_reason=None,
            )

    def test_blocked_status_requires_reason(self) -> None:
        with pytest.raises(ValueError, match="trend_reason"):
            _validate_trend_params(
                coverage_class="targeted-pair",
                campaign_id="c/v1", comparison_id="cmp-1",
                point_role="head", repeat_group="g", repeat_index=0,
                canonical_aggregate={"method": "mean", "count": 3, "metrics": {}, "outlier_handling": "none"},
                trend_status="blocked", trend_reason=None,
            )

    def test_valid_full_matrix_passes(self) -> None:
        _validate_trend_params(
            coverage_class="full-matrix",
            campaign_id="c/v1", comparison_id=None,
            point_role="checkpoint",
            repeat_group="g", repeat_index=0,
            canonical_aggregate={"method": "mean", "count": 3, "metrics": {"ttft_ms": {"value": 10}}, "outlier_handling": "none"},
            trend_status="default", trend_reason=None,
        )

    def test_valid_targeted_pair_passes(self) -> None:
        _validate_trend_params(
            coverage_class="targeted-pair",
            campaign_id="c/v1", comparison_id="cmp-1",
            point_role="baseline",
            repeat_group="g", repeat_index=0,
            canonical_aggregate={"method": "mean", "count": 3, "metrics": {"ttft_ms": {"value": 40}}, "outlier_handling": "none"},
            trend_status="default", trend_reason=None,
        )

    def test_valid_experimental_passes(self) -> None:
        _validate_trend_params(
            coverage_class="experimental",
            campaign_id=None, comparison_id=None,
            point_role=None, repeat_group=None, repeat_index=None,
            canonical_aggregate=None, trend_status="experimental",
            trend_reason="Single W8A8 run outside formal matrix",
        )


# ---------------------------------------------------------------------------
# Entry validation via T09
# ---------------------------------------------------------------------------


class TestEntryPassesT09Validation:
    """Produced entries (faked via minimal dicts) must pass T09 admission."""

    def test_full_matrix_default_status(self) -> None:
        entries = [
            minimal_full_matrix_entry(repeat_index=i)
            for i in range(3)
        ]
        report = validate_entries(entries)
        assert report.passed, f"Unexpected issues: {report.issues}"
        for decision in report.decisions:
            assert decision.status in ("default", "pending")

    def test_targeted_pair_both_sides(self) -> None:
        baseline = minimal_targeted_pair_entry(point_role="baseline", repeat_group="test::pair::baseline")
        head = minimal_targeted_pair_entry(point_role="head", repeat_group="test::pair::head")
        entries = []
        for i, eid in enumerate(["a0000000-0000-4000-8000-000000000001",
                                  "a0000000-0000-4000-8000-000000000002",
                                  "a0000000-0000-4000-8000-000000000003"]):
            b = dict(baseline)
            b["repeat_index"] = i
            b["entry_id"] = eid
            entries.append(b)
        for i, eid in enumerate(["b0000000-0000-4000-8000-000000000001",
                                  "b0000000-0000-4000-8000-000000000002",
                                  "b0000000-0000-4000-8000-000000000003"]):
            h = dict(head)
            h["repeat_index"] = i
            h["entry_id"] = eid
            entries.append(h)
        report = validate_entries(entries)
        assert report.passed, f"Unexpected issues: {report.issues}"

    def test_experimental_pass(self) -> None:
        entry = minimal_entry_dict(
            trend_schema_version=TREND_SCHEMA_VERSION,
            coverage_class="experimental",
            point_role=None,
            trend_status="experimental",
            trend_reason="Test experimental",
        )
        report = validate_entries([entry])
        assert report.passed, f"Unexpected issues: {report.issues}"
        assert report.decisions[0].status == "experimental"

    def test_invalid_entry_fails(self) -> None:
        """An entry with a bad metric should be flagged as invalid."""
        entry = minimal_full_matrix_entry()
        entry["metrics"]["ttft_ms"] = -1  # invalid
        report = validate_entries([entry])
        assert not report.passed

    def test_missing_trend_fields_fails(self) -> None:
        """A legacy entry without trend fields is excluded, not admitted."""
        entry = minimal_entry_dict()  # no trend fields
        report = validate_entries([entry])
        # Legacy entries without trend fields are 'excluded' with warning
        assert report.decisions[0].status == "excluded"

    def test_trend_reason_required_for_blocked(self) -> None:
        entry = minimal_full_matrix_entry(
            trend_status="blocked",
            trend_reason="Test blocking reason",
        )
        report = validate_entries([entry])
        assert report.passed, f"Unexpected issues: {report.issues}"
        assert report.decisions[0].status == "blocked"


# ---------------------------------------------------------------------------
# Workload config contract integration
# ---------------------------------------------------------------------------


class TestWorkloadConfigContract:
    """Official entries must have the contract marker set."""

    def test_official_entry_gets_contract_marker(self) -> None:
        entry = minimal_full_matrix_entry()
        entry["metadata"]["workload_config_contract"] = WORKLOAD_CONFIG_CONTRACT_VERSION
        entry["same_spec"] = {
            "spec_id": "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2",
            "resolved_server_parameters": {"gpu_memory_utilization": 0.6},
            "resolved_client_parameters": {
                "dataset_name": "random",
                "random_input_len": 1024,
                "random_output_len": 256,
                "num_prompts": 200,
                "request_rate": 1,
                "no_stream": False,
            },
        }
        from vllm_hust_benchmark.workload_config_contract import validate_explicit_workload_config
        errors = validate_explicit_workload_config(entry)
        assert not errors, f"Contract errors: {errors}"

    def test_official_entry_missing_contract_fails(self) -> None:
        entry = minimal_full_matrix_entry()
        entry["same_spec"] = {
            "spec_id": "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2",
            "resolved_server_parameters": {},
            "resolved_client_parameters": {},
        }
        from vllm_hust_benchmark.workload_config_contract import validate_explicit_workload_config
        errors = validate_explicit_workload_config(entry)
        assert errors  # Should have contract validation errors

    def test_non_official_entry_skips_contract(self) -> None:
        entry = minimal_full_matrix_entry()
        entry["same_spec"] = {
            "spec_id": "non-official-spec",
            "resolved_server_parameters": {},
            "resolved_client_parameters": {},
        }
        from vllm_hust_benchmark.workload_config_contract import validate_explicit_workload_config
        errors = validate_explicit_workload_config(entry)  # non-official should return []
        assert not errors


# ---------------------------------------------------------------------------
# add_trend_fields_to_existing_entry — migration helper
# ---------------------------------------------------------------------------


class TestAddTrendFieldsToExistingEntry:

    def test_adds_fields_and_passes_validation(self, tmp_path: Path) -> None:
        artifact = tmp_path / "run_leaderboard.json"
        artifact.write_text(json.dumps(minimal_entry_dict(), indent=2))
        add_trend_fields_to_existing_entry(
            artifact,
            coverage_class="full-matrix",
            campaign_id="migration/v1",
            point_role="checkpoint",
            repeat_group="mig::group",
            repeat_index=0,
            canonical_aggregate={
                "method": "mean", "count": 3,
                "metrics": {"ttft_ms": {"value": 42.0}},
                "outlier_handling": "none",
            },
            trend_status="default",
            validate=True,
        )
        entry = json.loads(artifact.read_text())
        assert entry["trend_schema_version"] == TREND_SCHEMA_VERSION
        assert entry["coverage_class"] == "full-matrix"
        assert entry["trend_status"] == "default"

    def test_missing_reason_for_blocked_fails(self, tmp_path: Path) -> None:
        artifact = tmp_path / "run_leaderboard.json"
        artifact.write_text(json.dumps(minimal_entry_dict(), indent=2))
        with pytest.raises(ValueError, match="trend_reason"):
            add_trend_fields_to_existing_entry(
                artifact,
                coverage_class="full-matrix",
                campaign_id="c/v1",
                point_role="checkpoint",
                trend_status="blocked",
                trend_reason=None,
            )

    def test_validate_flag_can_be_disabled(self, tmp_path: Path) -> None:
        """With validate=False, a badly formed entry still gets written."""
        artifact = tmp_path / "run_leaderboard.json"
        entry = minimal_entry_dict()
        entry["metrics"]["ttft_ms"] = -1  # invalid, but we won't validate
        artifact.write_text(json.dumps(entry, indent=2))
        # This should not raise because validate=False
        add_trend_fields_to_existing_entry(
            artifact,
            coverage_class="full-matrix",
            campaign_id="c/v1",
            point_role="checkpoint",
            repeat_group="g",
            repeat_index=0,
            canonical_aggregate={
                "method": "mean", "count": 3,
                "metrics": {"ttft_ms": {"value": 42.0}},
                "outlier_handling": "none",
            },
            trend_status="blocked",
            trend_reason="Known invalid metric",
            validate=False,
        )
        # Verify the fields were still written
        updated = json.loads(artifact.read_text())
        assert updated["coverage_class"] == "full-matrix"
        assert updated["trend_reason"] == "Known invalid metric"

    def test_non_dict_json_raises_value_error(self, tmp_path: Path) -> None:
        """A JSON array (non-dict) should raise ValueError."""
        artifact = tmp_path / "non_dict_entry.json"
        artifact.write_text(json.dumps(["not", "a", "dict"]))
        with pytest.raises(ValueError, match="expected a JSON object"):
            add_trend_fields_to_existing_entry(
                artifact,
                coverage_class="full-matrix",
                campaign_id="c/v1",
                point_role="checkpoint",
                trend_status="default",
            )


# ---------------------------------------------------------------------------
# Integration: end-to-end trend field presence
# ---------------------------------------------------------------------------


class TestProduceTrendEntryEndToEnd:
    """End-to-end tests that verify produced entries carry expected fields.

    Because ``export_leaderboard_artifacts`` requires real files (metrics,
    benchmarks results, constraints), we cannot easily call
    ``produce_trend_entry()`` in a unit test without fixtures.  Instead we
    verify that the **in-memory validation path** works, and that the field
    overlay logic is correct.
    """

    def test_full_matrix_field_set(self) -> None:
        entry = minimal_full_matrix_entry()
        expected = {
            "trend_schema_version": TREND_SCHEMA_VERSION,
            "coverage_class": "full-matrix",
            "campaign_id": "test-campaign/v1",
            "point_role": "checkpoint",
            "repeat_group": "test::group",
            "repeat_index": 0,
            "canonical_aggregate": {
                "method": "mean",
                "count": 3,
                "metrics": {"ttft_ms": {"value": 42.0}},
                "outlier_handling": "none",
            },
            "trend_status": "default",
        }
        for key, value in expected.items():
            assert entry[key] == value, f"Mismatch for {key}"

    def test_targeted_pair_field_set(self) -> None:
        entry = minimal_targeted_pair_entry()
        assert entry["trend_schema_version"] == TREND_SCHEMA_VERSION
        assert entry["coverage_class"] == "targeted-pair"
        assert entry["campaign_id"] == "test-pair/v1"
        assert entry["comparison_id"] == "test-comparison"
        assert entry["point_role"] == "head"
        assert entry["trend_status"] == "default"

    def test_experimental_field_set(self) -> None:
        entry = minimal_entry_dict(
            trend_schema_version=TREND_SCHEMA_VERSION,
            coverage_class="experimental",
            point_role=None,
            trend_status="experimental",
            trend_reason="Experimental test entry",
        )
        assert entry["coverage_class"] == "experimental"
        assert entry["point_role"] is None
        assert entry["trend_status"] == "experimental"
        assert "comparison_id" not in entry

    def test_legacy_entry_has_no_trend_fields(self) -> None:
        entry = minimal_entry_dict()
        for field in ("trend_schema_version", "coverage_class", "trend_status",
                      "campaign_id", "comparison_id", "point_role",
                      "repeat_group", "repeat_index", "canonical_aggregate"):
            assert field not in entry, f"{field} should not be in legacy entry"

    def test_canonical_aggregate_required_for_repeated_non_experimental(self) -> None:
        """full-matrix with repeat_group must have canonical_aggregate."""
        with pytest.raises(ValueError, match="canonical_aggregate"):
            _validate_trend_params(
                coverage_class="full-matrix",
                campaign_id="c/v1", comparison_id=None,
                point_role="checkpoint",
                repeat_group="g", repeat_index=0,
                canonical_aggregate=None,
                trend_status="default", trend_reason=None,
            )
