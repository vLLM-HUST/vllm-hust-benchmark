"""Tests for aggregate_results.py — repeat-run aggregation."""

from __future__ import annotations

import copy
import json
import math
import statistics
import tempfile
from pathlib import Path

import pytest

from vllm_hust_benchmark.aggregate_results import (
    VALID_AGG_METHODS,
    VALID_OUTLIER_HANDLING,
    aggregate_entries,
    apply_aggregate_to_entry,
    build_series_signature,
    compute_canonical_aggregate,
    compute_metric_stats,
    detect_outliers_iqr,
    detect_outliers_3sigma,
    get_repeat_group,
    get_repeat_index,
    group_entries_by_repeat_group,
    load_entries_from_paths,
    validate_aggregate_method,
    validate_aggregate_structure,
    write_aggregated_entries,
    _trimmed_mean,
    _percentile,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ttft_values():
    return [280.0, 290.0, 310.0, 285.0, 295.0]


@pytest.fixture
def base_entry_dict():
    """A minimal but valid leaderboard entry skeleton."""
    return {
        "entry_id": "test-entry-0000-0000-000000000000",
        "engine": "vllm-hust",
        "engine_version": "v0.20.1rc0-464-g51621c35bc",
        "config_type": "multi_gpu",
        "hardware": {
            "vendor": "Huawei",
            "chip_model": "910B2",
            "chip_count": 2,
            "interconnect": "HCCS",
        },
        "model": {
            "canonical_id": "hf:Qwen/Qwen2.5-14B-Instruct",
            "repo_id": "Qwen/Qwen2.5-14B-Instruct",
            "short_name": "Qwen2.5-14B-Instruct",
            "display_name": "Qwen2.5-14B-Instruct",
            "name": "Qwen/Qwen2.5-14B-Instruct",
            "parameters": "14B",
            "precision": "BF16",
        },
        "workload": {
            "name": "agent-research-online",
            "input_length": 1024,
            "output_length": 256,
        },
        "metrics": {
            "ttft_ms": 290.0,
            "throughput_tps": 185.0,
            "peak_mem_mb": 20480,
            "error_rate": 0.0,
        },
    }


@pytest.fixture
def repeat_group_entries(base_entry_dict):
    """Three entries belonging to the same repeat_group."""
    rg = (
        "full-stack-jul-2026/v1::"
        "hf:Qwen/Qwen2.5-14B-Instruct::910B2::BF16::"
        "agent-research-online::2chip::multi_gpu::vllm-hust"
    )
    entries = []
    for idx, (ttft, tp, mem) in enumerate(
        [(280.0, 182.0, 20480), (290.0, 185.0, 20480), (310.0, 194.0, 20480)]
    ):
        e = copy.deepcopy(base_entry_dict)
        e["entry_id"] = f"test-entry-{idx:04d}-{rg[:8]}"
        e["repeat_group"] = rg
        e["repeat_index"] = idx
        e["metrics"]["ttft_ms"] = ttft
        e["metrics"]["throughput_tps"] = tp
        e["metrics"]["peak_mem_mb"] = mem
        entries.append(e)
    return entries


@pytest.fixture
def mixed_entries(base_entry_dict, repeat_group_entries):
    """Entries with and without repeat_group (to test pass-through)."""
    entries = list(repeat_group_entries)

    # Add a solo entry without repeat_group
    solo = copy.deepcopy(base_entry_dict)
    solo["entry_id"] = "solo-entry-0000-0000-000000000000"
    solo["repeat_group"] = None
    solo["repeat_index"] = None
    entries.append(solo)

    return entries


# ---------------------------------------------------------------------------
# Test: helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_get_repeat_group_present(self, base_entry_dict):
        e = dict(base_entry_dict, repeat_group="group/v1::series")
        assert get_repeat_group(e) == "group/v1::series"

    def test_get_repeat_group_none(self, base_entry_dict):
        e = dict(base_entry_dict, repeat_group=None)
        assert get_repeat_group(e) is None

    def test_get_repeat_group_empty_string(self, base_entry_dict):
        e = dict(base_entry_dict, repeat_group="")
        assert get_repeat_group(e) is None

    def test_get_repeat_index_present(self):
        e = {"repeat_index": 3}
        assert get_repeat_index(e) == 3

    def test_get_repeat_index_float(self):
        e = {"repeat_index": 2.0}
        assert get_repeat_index(e) == 2

    def test_get_repeat_index_float_non_integer(self):
        e = {"repeat_index": 2.5}
        assert get_repeat_index(e) is None

    def test_get_repeat_index_none(self):
        e = {"repeat_index": None}
        assert get_repeat_index(e) is None

    def test_build_series_signature(self, base_entry_dict):
        sig = build_series_signature(base_entry_dict)
        assert "hf:Qwen/Qwen2.5-14B-Instruct" in sig
        assert "910B2" in sig
        assert "BF16" in sig
        assert "agent-research-online" in sig
        assert "2" in sig  # chip_count
        assert "multi_gpu" in sig
        assert "vllm-hust" in sig
        assert "v0.20.1rc0-464-g51621c35bc" in sig

    def test_build_series_signature_empty(self):
        assert build_series_signature({}) == ""

    def test_percentile(self):
        vals = [1, 2, 3, 4, 5]
        assert _percentile(vals, 0) == 1.0
        assert _percentile(vals, 50) == 3.0
        assert _percentile(vals, 100) == 5.0

    def test_percentile_interpolation(self):
        vals = [1, 2, 3, 4]
        p50 = _percentile(vals, 50)
        # 50th percentile of 4 values: k = 0.5 * 3 = 1.5
        # Interpolate: vals[1] + 0.5 * (vals[2] - vals[1]) = 2 + 0.5 = 2.5
        assert p50 == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Test: outlier detection
# ---------------------------------------------------------------------------


class TestOutlierDetection:
    def test_iqr_no_outliers(self):
        vals = [280, 290, 295, 300, 310]
        indices, (lo, hi) = detect_outliers_iqr(vals)
        assert indices == []

    def test_iqr_with_outlier(self):
        vals = [280, 290, 295, 300, 310, 550]
        indices, (lo, hi) = detect_outliers_iqr(vals)
        assert 5 in indices

    def test_iqr_low_outlier(self):
        vals = [50, 280, 290, 295, 300, 310]
        indices, _ = detect_outliers_iqr(vals)
        assert 0 in indices

    def test_iqr_too_few(self):
        vals = [1, 2, 3]
        indices, _ = detect_outliers_iqr(vals)
        assert indices == []  # need >= 4

    def test_3sigma_no_outliers(self):
        vals = [280, 290, 295, 300, 310]
        indices, _ = detect_outliers_3sigma(vals)
        assert indices == []

    def test_3sigma_with_outlier(self):
        # 3σ requires n ≥ 12 to detect a single extreme outlier (the outlier
        # inflates σ in small samples).  With n=12 tight values + 1 outlier:
        # [10]*11 + [100]
        vals = [10.0] * 11 + [100.0]
        indices, _ = detect_outliers_3sigma(vals)
        assert 11 in indices

    def test_3sigma_too_few(self):
        vals = [1, 2]
        indices, _ = detect_outliers_3sigma(vals)
        assert indices == []  # need >= 3

    def test_3sigma_detects_moderate_outlier(self):
        # With moderate separation and larger n, 3σ works:
        # 6 values at ~100 + 1 at 200
        vals = [95.0, 98.0, 100.0, 102.0, 105.0, 98.0, 200.0]
        # μ ≈ 114, σ ≈ ~38. 3σ ≈ 114, upper ≈ 228
        # |200-114| = 86 < 114 — not detected with n=7!
        # Use a larger n: [100]*10 + [200]
        vals2 = [100.0] * 10 + [200.0]
        indices, _ = detect_outliers_3sigma(vals2)
        assert 10 in indices


# ---------------------------------------------------------------------------
# Test: per-metric statistics
# ---------------------------------------------------------------------------


class TestComputeMetricStats:
    def test_mean(self, sample_ttft_values):
        result = compute_metric_stats(sample_ttft_values, method="mean")
        assert result["value"] == pytest.approx(292.0)
        assert result["min"] == 280.0
        assert result["max"] == 310.0
        # stdev([280, 290, 310, 285, 295]) ≈ 11.51
        assert result["std"] == pytest.approx(11.51, abs=0.05)

    def test_median(self, sample_ttft_values):
        result = compute_metric_stats(sample_ttft_values, method="median")
        assert result["value"] == 290.0

    def test_min(self, sample_ttft_values):
        result = compute_metric_stats(sample_ttft_values, method="min")
        assert result["value"] == 280.0

    def test_max(self, sample_ttft_values):
        result = compute_metric_stats(sample_ttft_values, method="max")
        assert result["value"] == 310.0

    def test_trimmed_mean(self, sample_ttft_values):
        # [280, 290, 310, 285, 295] sorted: [280, 285, 290, 295, 310]
        # trim 10% => remove 0 from each end (5*0.1=0.5, floor=0)
        # So trimmed_mean(0.1) ≈ mean of all 5 = 292.0
        result = compute_metric_stats(sample_ttft_values, method="trimmed_mean", trim_percent=0.1)
        assert result["value"] == pytest.approx(292.0)

    def test_trimmed_mean_20pct(self):
        # [1, 2, 3, 4, 5] trim 0.2 => remove 1 from each end => [2, 3, 4] => mean=3.0
        result = compute_metric_stats([1, 2, 3, 4, 5], method="trimmed_mean", trim_percent=0.2)
        assert result["value"] == 3.0

    def test_single_value(self):
        result = compute_metric_stats([42.0], method="mean")
        assert result["value"] == 42.0
        assert result["min"] == 42.0
        assert result["max"] == 42.0
        assert "std" not in result  # omitted for single value

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_metric_stats([], method="mean")


# ---------------------------------------------------------------------------
# Test: validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_aggregate_method(self):
        for m in sorted(VALID_AGG_METHODS):
            count = 3 if m != "trimmed_mean" else 4
            errors = validate_aggregate_method(m, count)
            assert errors == [], f"method={m} should be valid with count={count}, got {errors}"

    def test_invalid_aggregate_method(self):
        errors = validate_aggregate_method("invalid_method", 3)
        assert len(errors) >= 1
        assert "Invalid method" in errors[0]

    def test_trimmed_mean_needs_count_4(self):
        errors = validate_aggregate_method("trimmed_mean", 3)
        assert len(errors) == 1
        assert "count >= 4" in errors[0]

    def test_trimmed_mean_ok_with_4(self):
        errors = validate_aggregate_method("trimmed_mean", 4)
        assert errors == []

    def test_validate_structure_valid(self):
        agg = {
            "method": "mean",
            "count": 3,
            "metrics": {
                "ttft_ms": {"value": 295.0, "min": 280.0, "max": 310.0, "std": 15.0}
            },
            "outlier_handling": "none",
            "outlier_details": None,
            "note": "test",
        }
        errors = validate_aggregate_structure(agg)
        assert errors == []

    def test_validate_structure_invalid_method(self):
        agg = {
            "method": "foobar",
            "count": 3,
            "metrics": {"ttft_ms": {"value": 295.0}},
            "outlier_handling": "none",
        }
        errors = validate_aggregate_structure(agg)
        assert any("Invalid method" in e for e in errors)

    def test_validate_structure_missing_count(self):
        agg = {
            "method": "mean",
            "metrics": {"ttft_ms": {"value": 295.0}},
            "outlier_handling": "none",
        }
        errors = validate_aggregate_structure(agg)
        assert any("count" in e for e in errors)

    def test_validate_structure_invalid_outlier_handling(self):
        agg = {
            "method": "mean",
            "count": 3,
            "metrics": {"ttft_ms": {"value": 295.0}},
            "outlier_handling": "bananas",
        }
        errors = validate_aggregate_structure(agg)
        assert any("Invalid outlier_handling" in e for e in errors)


# ---------------------------------------------------------------------------
# Test: grouping
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_group_entries(self, repeat_group_entries, base_entry_dict):
        # Add a second group
        entries = list(repeat_group_entries)
        solo = copy.deepcopy(base_entry_dict)
        solo["entry_id"] = "solo"
        solo["repeat_group"] = None
        solo["repeat_index"] = None
        entries.append(solo)

        groups = group_entries_by_repeat_group(entries)
        assert "" in groups  # entries without repeat_group
        assert repeat_group_entries[0]["repeat_group"] in groups

        group_key = repeat_group_entries[0]["repeat_group"]
        assert len(groups[group_key]) == 3

    def test_group_sort_by_index(self, base_entry_dict):
        rg = "test-group/v1::series"
        entries = []
        for idx in [2, 0, 1]:
            e = copy.deepcopy(base_entry_dict)
            e["repeat_group"] = rg
            e["repeat_index"] = idx
            e["entry_id"] = f"e{idx}"
            entries.append(e)

        groups = group_entries_by_repeat_group(entries)
        group = groups[rg]
        indices = [get_repeat_index(e) for e in group]
        assert indices == [0, 1, 2]

    def test_no_repeat_group_passthrough(self, base_entry_dict):
        e = copy.deepcopy(base_entry_dict)
        e["repeat_group"] = None
        groups = group_entries_by_repeat_group([e])
        assert "" in groups
        assert len(groups[""]) == 1


# ---------------------------------------------------------------------------
# Test: canonical_aggregate computation
# ---------------------------------------------------------------------------


class TestCanonicalAggregate:
    def test_basic_mean(self, repeat_group_entries):
        agg = compute_canonical_aggregate(repeat_group_entries, method="mean")
        assert agg["method"] == "mean"
        assert agg["count"] == 3
        assert agg["outlier_handling"] == "none"

        # TTFT: [280, 290, 310] => mean 293.33...
        assert agg["metrics"]["ttft_ms"]["value"] == pytest.approx(293.333, abs=0.01)
        assert agg["metrics"]["ttft_ms"]["min"] == 280.0
        assert agg["metrics"]["ttft_ms"]["max"] == 310.0
        assert agg["metrics"]["ttft_ms"]["std"] is not None

        # throughput: [182, 185, 194] => mean 187.0
        assert agg["metrics"]["throughput_tps"]["value"] == pytest.approx(187.0)

        # peak_mem: [20480, 20480, 20480] => mean 20480, std 0
        assert agg["metrics"]["peak_mem_mb"]["value"] == 20480.0
        assert agg["metrics"]["peak_mem_mb"]["std"] == 0.0

    def test_median(self, repeat_group_entries):
        agg = compute_canonical_aggregate(repeat_group_entries, method="median")
        # Sorted TTFT: [280, 290, 310] => median 290
        assert agg["metrics"]["ttft_ms"]["value"] == 290.0

    def test_min(self, repeat_group_entries):
        agg = compute_canonical_aggregate(repeat_group_entries, method="min")
        assert agg["metrics"]["ttft_ms"]["value"] == 280.0
        assert agg["metrics"]["throughput_tps"]["value"] == 182.0

    def test_max(self, repeat_group_entries):
        agg = compute_canonical_aggregate(repeat_group_entries, method="max")
        assert agg["metrics"]["ttft_ms"]["value"] == 310.0
        assert agg["metrics"]["throughput_tps"]["value"] == 194.0

    def test_single_entry(self, base_entry_dict):
        e = copy.deepcopy(base_entry_dict)
        e["repeat_group"] = "some-group/v1::series"
        e["repeat_index"] = 0
        agg = compute_canonical_aggregate([e], method="mean")
        assert agg["count"] == 1
        assert agg["metrics"]["ttft_ms"]["value"] == 290.0
        assert agg["metrics"]["ttft_ms"]["min"] == 290.0
        assert agg["metrics"]["ttft_ms"]["max"] == 290.0

    def test_empty_entries_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_canonical_aggregate([], method="mean")

    def test_invalid_method_raises(self, repeat_group_entries):
        with pytest.raises(ValueError, match="Invalid method"):
            compute_canonical_aggregate(repeat_group_entries, method="foobar")

    def test_trimmed_mean_requires_min_4(self, repeat_group_entries):
        with pytest.raises(ValueError, match="requires count >= 4"):
            compute_canonical_aggregate(repeat_group_entries, method="trimmed_mean")

    def test_trimmed_mean_works_with_4(self, base_entry_dict):
        rg = "group/v1::series"
        entries = []
        for idx in range(4):
            e = copy.deepcopy(base_entry_dict)
            e["repeat_group"] = rg
            e["repeat_index"] = idx
            e["metrics"]["ttft_ms"] = 290.0 + idx * 10  # [290, 300, 310, 320]
            entries.append(e)
        agg = compute_canonical_aggregate(entries, method="trimmed_mean", trim_percent=0.25)
        # Sorted: [290, 300, 310, 320], trim 25% each side → remove 1 each → [300, 310] → mean 305
        assert agg["metrics"]["ttft_ms"]["value"] == pytest.approx(305.0)

    def test_note_contains_count(self, repeat_group_entries):
        agg = compute_canonical_aggregate(repeat_group_entries, method="mean")
        assert "3" in agg["note"]
        assert "mean" in agg["note"]


# ---------------------------------------------------------------------------
# Test: outlier handling in aggregate
# ---------------------------------------------------------------------------


class TestOutlierHandling:
    def test_outlier_removed(self, base_entry_dict):
        rg = "group/v1::series"
        entries = []
        for idx, ttft in enumerate([280, 290, 295, 300, 310, 550]):
            e = copy.deepcopy(base_entry_dict)
            e["repeat_group"] = rg
            e["repeat_index"] = idx
            e["metrics"]["ttft_ms"] = float(ttft)
            entries.append(e)

        agg = compute_canonical_aggregate(
            entries, method="mean", outlier_handling="removed", outlier_detection="iqr"
        )
        # The 550 outlier should be removed
        # Remaining: [280, 290, 295, 300, 310] => mean 295.0
        assert agg["outlier_handling"] == "removed"
        assert agg["metrics"]["ttft_ms"]["value"] == pytest.approx(295.0)
        assert agg["outlier_details"] is not None
        assert "removed" in agg["outlier_details"]

    def test_outlier_capped(self, base_entry_dict):
        rg = "group/v1::series"
        entries = []
        for idx, ttft in enumerate([280, 290, 295, 300, 310, 550]):
            e = copy.deepcopy(base_entry_dict)
            e["repeat_group"] = rg
            e["repeat_index"] = idx
            e["metrics"]["ttft_ms"] = float(ttft)
            entries.append(e)

        agg = compute_canonical_aggregate(
            entries, method="mean", outlier_handling="capped", outlier_detection="iqr"
        )
        assert agg["outlier_handling"] == "capped"
        assert agg["outlier_details"] is not None
        assert "Capped" in agg["outlier_details"]

    def test_outlier_handling_none_ignores(self, repeat_group_entries):
        agg = compute_canonical_aggregate(
            repeat_group_entries, method="mean", outlier_handling="none"
        )
        assert agg["outlier_handling"] == "none"
        assert agg["outlier_details"] is None


# ---------------------------------------------------------------------------
# Test: applying aggregate to entry
# ---------------------------------------------------------------------------


class TestApplyAggregate:
    def test_apply_updates_metrics(self, base_entry_dict):
        entry = copy.deepcopy(base_entry_dict)
        agg = {
            "method": "mean",
            "count": 3,
            "metrics": {
                "ttft_ms": {"value": 295.0, "min": 280.0, "max": 310.0, "std": 15.0},
                "throughput_tps": {"value": 187.0, "min": 182.0, "max": 194.0, "std": 6.0},
            },
            "outlier_handling": "none",
        }
        result = apply_aggregate_to_entry(entry, agg)
        assert result["metrics"]["ttft_ms"] == 295.0
        assert result["metrics"]["throughput_tps"] == 187.0
        assert result["canonical_aggregate"] == agg
        # Original unchanged
        assert entry["metrics"]["ttft_ms"] == 290.0  # original value preserved

    def test_apply_does_not_mutate_original(self, base_entry_dict):
        entry = copy.deepcopy(base_entry_dict)
        original_ttft = entry["metrics"]["ttft_ms"]
        agg = {
            "method": "mean",
            "count": 3,
            "metrics": {"ttft_ms": {"value": 999.0}},
            "outlier_handling": "none",
        }
        result = apply_aggregate_to_entry(entry, agg)
        assert result["metrics"]["ttft_ms"] == 999.0
        assert entry["metrics"]["ttft_ms"] == original_ttft


# ---------------------------------------------------------------------------
# Test: determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_different_order_same_result(self, base_entry_dict):
        """Aggregation must produce the same result regardless of input order."""
        rg = "group/v1::series"
        entries = []
        for idx, vals in enumerate([(280, 182), (310, 194), (290, 185)]):
            e = copy.deepcopy(base_entry_dict)
            e["repeat_group"] = rg
            e["repeat_index"] = idx
            e["metrics"]["ttft_ms"] = float(vals[0])
            e["metrics"]["throughput_tps"] = float(vals[1])
            entries.append(e)

        # Order 1: as defined
        agg1 = compute_canonical_aggregate(entries, method="mean")

        # Order 2: reversed
        agg2 = compute_canonical_aggregate(list(reversed(entries)), method="mean")

        assert agg1["count"] == agg2["count"]
        assert agg1["method"] == agg2["method"]
        for metric in agg1["metrics"]:
            assert agg1["metrics"][metric]["value"] == pytest.approx(
                agg2["metrics"][metric]["value"]
            )
            assert agg1["metrics"][metric]["min"] == agg2["metrics"][metric]["min"]
            assert agg1["metrics"][metric]["max"] == agg2["metrics"][metric]["max"]
            assert agg1["metrics"][metric]["std"] == pytest.approx(
                agg2["metrics"][metric]["std"]
            )

    def test_different_batch_order_aggregate_entries(self, base_entry_dict):
        """The high-level aggregate_entries must also be deterministic."""

        rg = "group/v1::series"
        entries = []
        for idx, vals in enumerate([(280, 182), (310, 194), (290, 185)]):
            e = copy.deepcopy(base_entry_dict)
            e["repeat_group"] = rg
            e["repeat_index"] = idx
            e["metrics"]["ttft_ms"] = float(vals[0])
            e["metrics"]["throughput_tps"] = float(vals[1])
            entries.append(e)

        result1 = aggregate_entries(entries, method="mean")
        result2 = aggregate_entries(list(reversed(entries)), method="mean")

        assert len(result1) == 1
        assert len(result2) == 1
        for metric in result1[0]["canonical_aggregate"]["metrics"]:
            assert result1[0]["canonical_aggregate"]["metrics"][metric]["value"] == pytest.approx(
                result2[0]["canonical_aggregate"]["metrics"][metric]["value"]
            )


# ---------------------------------------------------------------------------
# Test: aggregate_entries high-level
# ---------------------------------------------------------------------------


class TestAggregateEntries:
    def test_basic(self, mixed_entries):

        result = aggregate_entries(mixed_entries, method="mean")
        # 1 aggregated group + 1 solo entry = 2 output entries
        assert len(result) == 2

        # Check the aggregated entry
        agg_entries = [e for e in result if e.get("canonical_aggregate")]
        solo_entries = [e for e in result if not e.get("canonical_aggregate")]
        assert len(agg_entries) == 1
        assert len(solo_entries) == 1
        assert solo_entries[0]["entry_id"] == "solo-entry-0000-0000-000000000000"

    def test_aggregated_entry_has_correct_metrics(self, repeat_group_entries):

        result = aggregate_entries(repeat_group_entries, method="mean")
        assert len(result) == 1
        e = result[0]
        agg = e["canonical_aggregate"]
        assert agg["metrics"]["ttft_ms"]["value"] == pytest.approx(
            statistics.mean([280, 290, 310])
        )
        # Top-level metrics should match aggregate value
        assert e["metrics"]["ttft_ms"] == agg["metrics"]["ttft_ms"]["value"]

    def test_no_repeat_group(self, base_entry_dict):

        entry = copy.deepcopy(base_entry_dict)
        result = aggregate_entries([entry], method="mean")
        assert len(result) == 1
        assert "canonical_aggregate" not in result[0]


# ---------------------------------------------------------------------------
# Test: file I/O
# ---------------------------------------------------------------------------


class TestFileIO:
    def test_load_entries_from_paths(self, repeat_group_entries):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i, e in enumerate(repeat_group_entries):
                p = Path(tmpdir) / f"entry_{i}.json"
                p.write_text(json.dumps(e), encoding="utf-8")
                paths.append(str(p))

            loaded = load_entries_from_paths(paths)
            assert len(loaded) == 3
            assert loaded[0]["entry_id"] == repeat_group_entries[0]["entry_id"]

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_entries_from_paths(["/tmp/nonexistent-file.json"])

    def test_write_aggregated_entries(self, repeat_group_entries):

        result = aggregate_entries(repeat_group_entries, method="mean")

        with tempfile.TemporaryDirectory() as tmpdir:
            written = write_aggregated_entries(result, tmpdir)
            assert len(written) == 1
            assert Path(written[0]).is_file()
            data = json.loads(Path(written[0]).read_text(encoding="utf-8"))
            assert "canonical_aggregate" in data
            assert data["canonical_aggregate"]["method"] == "mean"


# ---------------------------------------------------------------------------
# Test: trimmed_mean helper
# ---------------------------------------------------------------------------


class TestTrimmedMean:
    def test_basic(self):
        result = _trimmed_mean([1, 2, 3, 4, 5], 0.2)
        assert result == 3.0  # [1,2,3,4,5] → trim 0.2 → [2,3,4] → mean=3

    def test_no_trim(self):
        result = _trimmed_mean([1, 2, 3, 4, 5], 0.0)
        assert result == 3.0  # mean of all 5

    def test_trim_too_much_raises(self):
        # trim_percent >= 0.5 is caught by range check
        with pytest.raises(ValueError, match="trim_percent must be"):
            _trimmed_mean([1, 2, 3], 0.6)

    def test_invalid_trim_percent_raises(self):
        with pytest.raises(ValueError, match="trim_percent must be"):
            _trimmed_mean([1, 2, 3], -0.1)

    def test_trim_percent_too_high_raises(self):
        with pytest.raises(ValueError, match="trim_percent must be"):
            _trimmed_mean([1, 2, 3], 0.6)
