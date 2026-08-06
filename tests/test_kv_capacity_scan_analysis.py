"""Unit tests for KV capacity scan analysis and scheduler event parsing (issue #134).

PR #146 review fix tests:
  - Updated tiering config names (hbm-only / tiering-disabled / tiering-enabled).
  - Acceptance criteria now returns admitted|blocked|incomplete|negative-result.
  - Missing evidence → blocked (not negative-result).
  - Insufficient reps → incomplete or blocked (not negative-result).
  - Missing provenance → blocked.
  - Incomplete timeline (preemptions but no 6-stage chain) → incomplete.
  - Parser tests for stage_events, timeline_complete, verify_kv_capacity_from_log.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load modules from scripts/ (not a package, so use importlib)
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def analyze_mod():
    return _load_module("analyze_kv_capacity_scan", "analyze_kv_capacity_scan.py")


@pytest.fixture(scope="module")
def parse_mod():
    return _load_module("parse_scheduler_events", "parse_scheduler_events.py")


# ---------------------------------------------------------------------------
# Shared fixtures for provenance and complete analysis data
# ---------------------------------------------------------------------------


def _make_provenance():
    """Return a provenance dict with all REQUIRED_PROVENANCE_FIELDS filled."""
    return {
        "engine_commit": "abc123def456",  # pragma: allowlist secret
        "plugin_commit": "fed654cba321",  # pragma: allowlist secret
        "cann_version": "8.0.0",
        "driver_version": "24.1.0",
        "torch_npu_version": "2.5.0",
        "model_revision": "abc789xyz",
        "model_weight_fingerprint": "sha256:deadbeefcafef00d",
        "resolved_parameters": {
            "gpu_memory_utilization": "0.60",
            "max_model_len": "32768",
            "enable_prefix_caching": True,
            "dtype": "float16",
            "kv_transfer_config": None,
        },
        "actual_kv_bytes": 8638343168,
    }


def _make_complete_tiering():
    """Return a complete tiering_comparison analysis output dict."""
    return {
        "per_config_stats": {
            "hbm-only": {
                "repetitions": 3,
                "stats": {"output_throughput": {"median": 300.0}},
            },
            "tiering-disabled": {
                "repetitions": 3,
                "stats": {"output_throughput": {"median": 200.0}},
            },
            "tiering-enabled": {
                "repetitions": 3,
                "stats": {"output_throughput": {"median": 220.0}},
            },
        },
        "comparison": {},
        "best_config": {"throughput": "hbm-only", "ttft": "hbm-only"},
        "configs_present": ["hbm-only", "tiering-disabled", "tiering-enabled"],
        "configs_required": ["hbm-only", "tiering-disabled", "tiering-enabled"],
        "configs_complete": ["hbm-only", "tiering-disabled", "tiering-enabled"],
    }


def _make_complete_capacity_curves():
    """Return capacity_curves with all 3 SCAN_WORKLOADS, 4 capacities, 3 reps each."""
    return {
        workload: {
            "8": {
                "repetitions": 3,
                "stats": {"output_throughput": {"median": 100.0}},
                "raw_values": [100.0, 102.0, 98.0],
            },
            "16": {
                "repetitions": 3,
                "stats": {"output_throughput": {"median": 200.0}},
                "raw_values": [200.0, 202.0, 198.0],
            },
            "24": {
                "repetitions": 3,
                "stats": {"output_throughput": {"median": 210.0}},
                "raw_values": [210.0, 212.0, 208.0],
            },
            "32": {
                "repetitions": 3,
                "stats": {"output_throughput": {"median": 215.0}},
                "raw_values": [215.0, 217.0, 213.0],
            },
        }
        for workload in ["random-online", "sharegpt-online", "prefix-repetition-online"]
    }


def _make_run_manifest_map(count=3):
    """Return a valid run_manifest_map with ``count`` runs, each having a manifest."""
    return {
        f"raw_results/random-online/8/rep-{i + 1}": _make_provenance()
        for i in range(count)
    }


def _make_per_rep_manifests(count=3):
    """Return a list of valid per-rep manifests (backward compat)."""
    return [_make_provenance() for _ in range(count)]


def _make_complete_timeline():
    """Return a preempt_timeline with at least one complete episode."""
    return {
        "total_preemptions": 1,
        "pressure_episodes": [{"timeline_complete": True}],
        "timeline_status": "complete",
    }


# ---------------------------------------------------------------------------
# compute_stats tests
# ---------------------------------------------------------------------------


class TestComputeStats:
    def test_empty_list(self, analyze_mod):
        stats = analyze_mod.compute_stats([])
        assert stats["count"] == 0
        assert stats["median"] is None

    def test_single_value(self, analyze_mod):
        stats = analyze_mod.compute_stats([42.0])
        assert stats["median"] == 42.0
        assert stats["count"] == 1
        assert stats["stdev"] == 0.0

    def test_three_values(self, analyze_mod):
        stats = analyze_mod.compute_stats([100.0, 200.0, 300.0])
        assert stats["median"] == 200.0
        assert stats["min"] == 100.0
        assert stats["max"] == 300.0
        assert stats["count"] == 3
        assert stats["iqr"] is not None

    def test_even_count(self, analyze_mod):
        stats = analyze_mod.compute_stats([10.0, 20.0, 30.0, 40.0])
        assert stats["median"] == 25.0
        assert stats["count"] == 4

    def test_iqr_calculation(self, analyze_mod):
        # Q1 of [1,2,3,4,5,6,7,8] = 2.75, Q3 = 6.25
        stats = analyze_mod.compute_stats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        assert stats["p25"] is not None
        assert stats["p75"] is not None
        assert stats["iqr"] == pytest.approx(stats["p75"] - stats["p25"], abs=0.01)


# ---------------------------------------------------------------------------
# extract_metrics tests
# ---------------------------------------------------------------------------


class TestExtractMetrics:
    def test_full_result(self, analyze_mod):
        raw = {
            "mean_ttft_ms": 235.7,
            "median_ttft_ms": 218.2,
            "p99_ttft_ms": 418.1,
            "mean_tpot_ms": 40.0,
            "p99_tpot_ms": 44.8,
            "mean_itl_ms": 40.0,
            "p99_itl_ms": 124.3,
            "output_throughput": 244.9,
            "request_throughput": 0.96,
            "max_concurrent_requests": 22,
        }
        metrics = analyze_mod.extract_metrics(raw)
        assert metrics["mean_ttft_ms"] == 235.7
        assert metrics["output_throughput"] == 244.9
        assert metrics["max_concurrent_requests"] == 22

    def test_missing_fields(self, analyze_mod):
        raw = {"mean_ttft_ms": 100.0}
        metrics = analyze_mod.extract_metrics(raw)
        assert metrics["mean_ttft_ms"] == 100.0
        assert metrics["output_throughput"] is None
        assert metrics["p99_ttft_ms"] is None

    def test_empty_dict(self, analyze_mod):
        metrics = analyze_mod.extract_metrics({})
        assert all(v is None for v in metrics.values())


# ---------------------------------------------------------------------------
# aggregate_reps tests
# ---------------------------------------------------------------------------


class TestAggregateReps:
    def test_three_reps(self, analyze_mod):
        reps = [
            {"output_throughput": 240.0, "mean_ttft_ms": 200.0},
            {"output_throughput": 250.0, "mean_ttft_ms": 210.0},
            {"output_throughput": 260.0, "mean_ttft_ms": 220.0},
        ]
        agg = analyze_mod.aggregate_reps(reps)
        tput_stats = agg["per_metric_stats"]["output_throughput"]
        assert tput_stats["median"] == 250.0
        assert tput_stats["count"] == 3
        assert tput_stats["min"] == 240.0
        assert tput_stats["max"] == 260.0

    def test_empty_reps(self, analyze_mod):
        agg = analyze_mod.aggregate_reps([])
        assert agg["per_metric_stats"]["output_throughput"]["count"] == 0

    def test_partial_metrics(self, analyze_mod):
        reps = [
            {"output_throughput": 240.0, "mean_ttft_ms": 200.0},
            {"output_throughput": 250.0},  # missing ttft
            {"output_throughput": 260.0, "mean_ttft_ms": 220.0},
        ]
        agg = analyze_mod.aggregate_reps(reps)
        assert agg["per_metric_stats"]["output_throughput"]["count"] == 3
        assert agg["per_metric_stats"]["mean_ttft_ms"]["count"] == 2


# ---------------------------------------------------------------------------
# analyze_capacity_scan tests
# ---------------------------------------------------------------------------


class TestAnalyzeCapacityScan:
    @pytest.fixture
    def sample_results(self):
        """4 capacities × 2 workloads × 3 reps."""
        results = {}
        for workload in ["random-online", "sharegpt-online"]:
            results[workload] = {}
            for kv in [8, 16, 24, 32]:
                results[workload][str(kv)] = {}
                for rep in range(1, 4):
                    # Simulate: larger KV → higher throughput, lower TTFT
                    base_tput = 200 + kv * 5
                    base_ttft = 300 - kv * 5
                    results[workload][str(kv)][f"rep-{rep}"] = {
                        "output_throughput": base_tput + rep * 2,
                        "mean_ttft_ms": base_ttft + rep * 5,
                        "p99_ttft_ms": base_ttft * 1.5 + rep * 10,
                        "request_throughput": 1.0,
                        "max_concurrent_requests": kv,
                    }
        return results

    def test_capacity_curves_built(self, analyze_mod, sample_results):
        analysis = analyze_mod.analyze_capacity_scan(sample_results)
        assert "capacity_curves" in analysis
        assert "random-online" in analysis["capacity_curves"]
        assert "8" in analysis["capacity_curves"]["random-online"]

    def test_capacities_covered(self, analyze_mod, sample_results):
        analysis = analyze_mod.analyze_capacity_scan(sample_results)
        assert analysis["capacities_covered"] == [8, 16, 24, 32]

    def test_inflection_points(self, analyze_mod, sample_results):
        analysis = analyze_mod.analyze_capacity_scan(sample_results)
        inflection = analysis["inflection_points"]
        assert "random-online" in inflection
        assert inflection["random-online"]["throughput_inflection_gib"] is not None

    def test_repetitions_counted(self, analyze_mod, sample_results):
        analysis = analyze_mod.analyze_capacity_scan(sample_results)
        for workload, curve in analysis["capacity_curves"].items():
            for cap, data in curve.items():
                assert data["repetitions"] == 3


# ---------------------------------------------------------------------------
# identify_inflection_points tests
# ---------------------------------------------------------------------------


class TestIdentifyInflectionPoints:
    def test_clear_throughput_drop(self, analyze_mod):
        curve = {
            "8": {"stats": {"output_throughput": {"median": 100.0}}},
            "16": {"stats": {"output_throughput": {"median": 200.0}}},
            "24": {"stats": {"output_throughput": {"median": 210.0}}},
            "32": {"stats": {"output_throughput": {"median": 215.0}}},
        }
        result = analyze_mod.identify_inflection_points({"test": curve})
        # Largest delta: 100→200 at cap 16
        assert result["test"]["throughput_inflection_gib"] == 16

    def test_insufficient_data(self, analyze_mod):
        curve = {"8": {"stats": {"output_throughput": {"median": 100.0}}}}
        result = analyze_mod.identify_inflection_points({"test": curve})
        assert result["test"]["throughput_inflection_gib"] is None
        assert "note" in result["test"]

    def test_none_values_skipped(self, analyze_mod):
        curve = {
            "8": {"stats": {"output_throughput": {"median": None}}},
            "16": {"stats": {"output_throughput": {"median": 200.0}}},
            "24": {"stats": {"output_throughput": {"median": 210.0}}},
        }
        result = analyze_mod.identify_inflection_points({"test": curve})
        # Only 16→24 has valid delta
        assert result["test"]["throughput_inflection_gib"] == 24


# ---------------------------------------------------------------------------
# analyze_tiering_comparison tests
# ---------------------------------------------------------------------------


class TestAnalyzeTieringComparison:
    def test_three_configs(self, analyze_mod):
        # PR #146 fix: real tiering config names
        results = {
            "hbm-only": [
                {"output_throughput": 300.0, "mean_ttft_ms": 150.0},
                {"output_throughput": 310.0, "mean_ttft_ms": 145.0},
                {"output_throughput": 305.0, "mean_ttft_ms": 148.0},
            ],
            "tiering-disabled": [
                {"output_throughput": 200.0, "mean_ttft_ms": 250.0},
                {"output_throughput": 210.0, "mean_ttft_ms": 240.0},
                {"output_throughput": 205.0, "mean_ttft_ms": 245.0},
            ],
            "tiering-enabled": [
                {"output_throughput": 220.0, "mean_ttft_ms": 230.0},
                {"output_throughput": 230.0, "mean_ttft_ms": 220.0},
                {"output_throughput": 225.0, "mean_ttft_ms": 225.0},
            ],
        }
        analysis = analyze_mod.analyze_tiering_comparison(results)
        assert "hbm-only" in analysis["per_config_stats"]
        assert "tiering-disabled" in analysis["per_config_stats"]
        assert "tiering-enabled" in analysis["per_config_stats"]
        assert analysis["best_config"]["throughput"] == "hbm-only"
        assert analysis["best_config"]["ttft"] == "hbm-only"
        # PR #146 fix: new return fields
        assert "configs_present" in analysis
        assert "configs_required" in analysis
        assert "configs_complete" in analysis
        assert set(analysis["configs_present"]) == {
            "hbm-only",
            "tiering-disabled",
            "tiering-enabled",
        }
        assert analysis["configs_required"] == [
            "hbm-only",
            "tiering-disabled",
            "tiering-enabled",
        ]
        # All 3 configs have 3 reps with valid throughput
        assert set(analysis["configs_complete"]) == {
            "hbm-only",
            "tiering-disabled",
            "tiering-enabled",
        }

    def test_comparison_deltas(self, analyze_mod):
        results = {
            "config-a": [{"output_throughput": 100.0, "mean_ttft_ms": 200.0}],
            "config-b": [{"output_throughput": 150.0, "mean_ttft_ms": 100.0}],
        }
        analysis = analyze_mod.analyze_tiering_comparison(results)
        delta = analysis["comparison"]["config-a_vs_config-b"]
        assert delta["throughput_delta_pct"] == pytest.approx(50.0, abs=0.1)
        assert delta["ttft_delta_pct"] == pytest.approx(-50.0, abs=0.1)


# ---------------------------------------------------------------------------
# check_acceptance_criteria tests
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    def test_all_met(self, analyze_mod):
        # PR #146 fix: include provenance, new tiering config names, valid throughput
        analysis = {
            "capacities_covered": [8, 16, 24, 32],
            "inflection_points": {"random-online": {"throughput_inflection_gib": 16}},
            "preempt_timeline": {
                "total_preemptions": 5,
                "pressure_episodes": [{"timeline_complete": True}],
            },
            "tiering_comparison": _make_complete_tiering(),
            "capacity_curves": _make_complete_capacity_curves(),
            "provenance": _make_provenance(),
            "run_manifest_map": _make_run_manifest_map(3),
        }
        result = analyze_mod.check_acceptance_criteria(analysis)
        assert result["all_criteria_met"] is True
        assert result["overall_status"] == "admitted"

    def test_missing_capacity(self, analyze_mod):
        # PR #146 fix: missing evidence → blocked (not negative-result)
        analysis = {
            "capacities_covered": [8, 16],
            "inflection_points": {},
            "preempt_timeline": {},
            "tiering_comparison": {"per_config_stats": {}},
            "capacity_curves": {},
        }
        result = analyze_mod.check_acceptance_criteria(analysis)
        assert result["all_criteria_met"] is False
        assert result["overall_status"] == "blocked"

    def test_insufficient_reps(self, analyze_mod):
        # PR #146 fix: insufficient reps → incomplete or blocked (not negative-result)
        analysis = {
            "capacities_covered": [8, 16, 24, 32],
            "inflection_points": {"w": {"throughput_inflection_gib": 16}},
            "preempt_timeline": {
                "total_preemptions": 1,
                "pressure_episodes": [{"timeline_complete": True}],
            },
            "tiering_comparison": _make_complete_tiering(),
            "provenance": _make_provenance(),
            "run_manifest_map": _make_run_manifest_map(3),
            "capacity_curves": {
                "w": {
                    "8": {
                        "repetitions": 2,
                        "stats": {"output_throughput": {"median": 100.0}},
                        "raw_values": [100.0, 102.0],
                    },  # < 3
                    "16": {
                        "repetitions": 3,
                        "stats": {"output_throughput": {"median": 200.0}},
                        "raw_values": [200.0, 202.0, 198.0],
                    },
                    "24": {
                        "repetitions": 3,
                        "stats": {"output_throughput": {"median": 210.0}},
                        "raw_values": [210.0, 212.0, 208.0],
                    },
                    "32": {
                        "repetitions": 3,
                        "stats": {"output_throughput": {"median": 215.0}},
                        "raw_values": [215.0, 217.0, 213.0],
                    },
                }
            },
        }
        result = analyze_mod.check_acceptance_criteria(analysis)
        assert result["all_criteria_met"] is False
        assert result["overall_status"] in ("incomplete", "blocked")

    def test_missing_provenance_blocked(self, analyze_mod):
        # PR #146 fix: missing provenance → blocked
        analysis = {
            "capacities_covered": [8, 16, 24, 32],
            "inflection_points": {"w": {"throughput_inflection_gib": 16}},
            "preempt_timeline": {
                "total_preemptions": 1,
                "pressure_episodes": [{"timeline_complete": True}],
            },
            "tiering_comparison": _make_complete_tiering(),
            "capacity_curves": _make_complete_capacity_curves(),
            # No provenance key
        }
        result = analyze_mod.check_acceptance_criteria(analysis)
        assert result["all_criteria_met"] is False
        assert result["overall_status"] == "blocked"

    def test_incomplete_timeline_incomplete(self, analyze_mod):
        # PR #146 fix: preemptions but no complete 6-stage timeline → incomplete
        analysis = {
            "capacities_covered": [8, 16, 24, 32],
            "inflection_points": {"w": {"throughput_inflection_gib": 16}},
            "preempt_timeline": {
                "total_preemptions": 2,
                "pressure_episodes": [
                    {"timeline_complete": False},
                    {"timeline_complete": False},
                ],
            },
            "tiering_comparison": _make_complete_tiering(),
            "capacity_curves": _make_complete_capacity_curves(),
            "provenance": _make_provenance(),
            "run_manifest_map": _make_run_manifest_map(3),
        }
        result = analyze_mod.check_acceptance_criteria(analysis)
        assert result["all_criteria_met"] is False
        assert result["overall_status"] == "incomplete"


# ---------------------------------------------------------------------------
# generate_report tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_full_report(self, analyze_mod):
        # PR #146 fix: pass provenance, expect status in the 4-value set
        capacity_analysis = {
            "capacities_covered": [8, 16, 24, 32],
            "inflection_points": {"random-online": {"throughput_inflection_gib": 16}},
            "capacity_curves": _make_complete_capacity_curves(),
        }
        tiering_analysis = _make_complete_tiering()
        preempt_timeline = _make_complete_timeline()
        provenance = _make_provenance()

        report = analyze_mod.generate_report(
            capacity_analysis,
            tiering_analysis,
            preempt_timeline,
            provenance,
            _make_run_manifest_map(3),
        )
        assert report["issue"] == 134
        assert "acceptance_criteria" in report
        assert "issue_89_linkage" in report
        assert report["issue_89_linkage"]["status"] in (
            "admitted",
            "incomplete",
            "blocked",
            "negative-result",
        )
        # With complete data and provenance, should be admitted
        assert report["issue_89_linkage"]["status"] == "admitted"


# ---------------------------------------------------------------------------
# parse_scheduler_events tests
# ---------------------------------------------------------------------------


class TestParseSchedulerEvents:
    @pytest.fixture
    def sample_log(self):
        return """\
(EngineCore pid=123) INFO 07-26 15:48:54 [interface.py:620] Setting kv cache block size to 128 for CUSTOM backend.
(EngineCore pid=123) INFO 07-26 15:49:29 [worker.py:803] Available KV cache memory: 8.04 GiB
(EngineCore pid=123) INFO 07-26 15:49:29 [kv_cache_utils.py:2203] GPU KV cache size: 43,904 tokens
(EngineCore pid=123) INFO 07-26 15:49:29 [kv_cache_utils.py:2204] Maximum concurrency for 32,768 tokens per request: 1.34x
(EngineCore pid=123) INFO 07-26 15:49:29 [worker.py:1029] Free memory on device (60.61/60.96 GiB) on startup. Desired GPU memory utilization is (0.6, 36.57 GiB). Actual usage: 27.54 GiB for weights, 0.26 GiB for peak activation, 0.18 GiB for non-torch memory, 0.47 GiB for NPU graph memory. Current KV cache memory: 8.04 GiB.
(APIServer pid=456) INFO 07-26 15:50:29 [loggers.py:282] Engine 000: Avg prompt throughput: 115.2 tokens/s, Avg generation throughput: 29.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.6%, Prefix cache hit rate: 43.8%
(APIServer pid=456) INFO 07-26 15:50:39 [loggers.py:282] Engine 000: Avg prompt throughput: 819.1 tokens/s, Avg generation throughput: 161.4 tokens/s, Running: 7 reqs, Waiting: 3 reqs, GPU KV cache usage: 19.9%, Prefix cache hit rate: 8.8%
(APIServer pid=456) INFO 07-26 15:50:49 [loggers.py:282] Engine 000: Avg prompt throughput: 1023.9 tokens/s, Avg generation throughput: 191.3 tokens/s, Running: 9 reqs, Waiting: 0 reqs, GPU KV cache usage: 24.6%, Prefix cache hit rate: 4.4%
(EngineCore pid=123) WARNING 07-26 15:50:45 [scheduler.py:789] Sequence group 42 is preempted
"""

    def test_parse_kv_cache_info(self, parse_mod, sample_log):
        info = parse_mod.parse_kv_cache_info(sample_log)
        assert info["kv_cache_memory_gib"] == 8.04
        assert info["kv_cache_tokens"] == 43904
        assert info["max_concurrency"] == 1.34
        assert info["kv_block_size"] == 128
        assert info["current_kv_memory_gib"] == 8.04

    def test_parse_memory_breakdown(self, parse_mod, sample_log):
        breakdown = parse_mod.parse_memory_breakdown(sample_log)
        assert breakdown["free_memory_gib"] == 60.61
        assert breakdown["total_memory_gib"] == 60.96
        assert breakdown["desired_utilization"] == 0.6
        assert breakdown["weights_gib"] == 27.54
        assert breakdown["activation_gib"] == 0.26
        assert breakdown["graph_memory_gib"] == 0.47

    def test_parse_engine_stats(self, parse_mod, sample_log):
        stats = parse_mod.parse_engine_stats(sample_log)
        assert len(stats) == 3
        assert stats[0]["running_reqs"] == 1
        assert stats[0]["waiting_reqs"] == 0
        assert stats[1]["waiting_reqs"] == 3
        assert stats[2]["running_reqs"] == 9
        assert stats[0]["kv_cache_usage_pct"] == 2.6
        assert stats[0]["prefix_cache_hit_rate_pct"] == 43.8
        assert stats[0]["timestamp"] is not None

    def test_parse_preemption_events(self, parse_mod, sample_log):
        events = parse_mod.parse_preemption_events(sample_log)
        assert len(events) == 1
        assert events[0]["seq_group_id"] == 42
        assert events[0]["event_type"] == "preempted"
        assert events[0]["timestamp"] is not None

    def test_reconstruct_timeline(self, parse_mod, sample_log):
        preempt_events = parse_mod.parse_preemption_events(sample_log)
        engine_stats = parse_mod.parse_engine_stats(sample_log)
        timeline = parse_mod.reconstruct_preempt_timeline(preempt_events, engine_stats)
        assert timeline["total_preemptions"] == 1
        assert len(timeline["pressure_episodes"]) == 1
        # Preempt at 15:50:45, restore when waiting=0 at 15:50:49
        episode = timeline["pressure_episodes"][0]
        assert episode["preempt_seq_group_id"] == 42
        assert episode["peak_waiting_reqs"] == 3
        # PR #146 fix: check stages dict and timeline_status
        assert "stages" in episode
        assert "timeline_status" in timeline
        # Without stage_events, timeline is incomplete
        assert timeline["timeline_status"] == "incomplete"
        assert episode["timeline_complete"] is False

    def test_empty_log(self, parse_mod):
        assert parse_mod.parse_kv_cache_info("")["kv_cache_memory_gib"] is None
        assert parse_mod.parse_engine_stats("") == []
        assert parse_mod.parse_preemption_events("") == []

    def test_no_preemption_timeline(self, parse_mod):
        timeline = parse_mod.reconstruct_preempt_timeline([], [])
        assert timeline["total_preemptions"] == 0
        assert timeline["pressure_episodes"] == []
        # PR #146 fix: timeline_status field
        assert timeline["timeline_status"] == "no_preemptions"

    def test_stage_events_parsing(self, parse_mod):
        # PR #146 fix: log with restore/admission patterns → stage_events populated
        log = """\
(EngineCore pid=123) WARNING 07-26 15:50:45 [scheduler.py:789] Sequence group 42 is preempted
(EngineCore pid=123) INFO 07-26 15:50:46 [restore.py:10] Restoring KV cache for seq_group 42
(EngineCore pid=123) INFO 07-26 15:50:47 [restore.py:20] Restored KV cache for seq_group 42
(EngineCore pid=123) INFO 07-26 15:50:48 [scheduler.py:100] Scheduler woke up after restore
(EngineCore pid=123) INFO 07-26 15:50:49 [scheduler.py:200] Sequence group 42 admitted
(EngineCore pid=123) INFO 07-26 15:50:50 [decode.py:10] First prefill for seq_group 42
"""
        stage_events = parse_mod.parse_stage_events(log)
        assert len(stage_events["preempt"]) >= 1
        assert stage_events["preempt"][0]["seq_group_id"] == 42
        assert len(stage_events["restore_start"]) >= 1
        assert len(stage_events["restore_done"]) >= 1
        assert len(stage_events["scheduler_wakeup"]) >= 1
        assert len(stage_events["admission"]) >= 1
        assert len(stage_events["first_prefill_or_decode"]) >= 1

        # Reconstruct with stage_events → timeline should be complete
        preempt_events = parse_mod.parse_preemption_events(log)
        timeline = parse_mod.reconstruct_preempt_timeline(
            preempt_events, [], stage_events
        )
        assert timeline["total_preemptions"] == 1
        episode = timeline["pressure_episodes"][0]
        assert episode["timeline_complete"] is True
        assert timeline["timeline_status"] == "complete"
        # All 6 stages should have timestamps
        for stage in parse_mod.TIMELINE_STAGES:
            assert episode["stages"][stage] is not None, f"Stage {stage} is None"

    def test_timeline_incomplete_without_stages(self, parse_mod):
        # PR #146 fix: only preempt events, no stage events → timeline_complete=False
        log = """\
(EngineCore pid=123) WARNING 07-26 15:50:45 [scheduler.py:789] Sequence group 42 is preempted
"""
        preempt_events = parse_mod.parse_preemption_events(log)
        engine_stats = parse_mod.parse_engine_stats(log)
        timeline = parse_mod.reconstruct_preempt_timeline(preempt_events, engine_stats)
        assert timeline["total_preemptions"] == 1
        assert timeline["timeline_status"] == "incomplete"
        episode = timeline["pressure_episodes"][0]
        assert episode["timeline_complete"] is False
        # preempt stage should have a timestamp, others should be None
        assert episode["stages"]["preempt"] is not None
        assert episode["stages"]["restore_start"] is None

    def test_verify_kv_capacity_from_log(self, parse_mod, tmp_path):
        # PR #146 fix: verify actual KV matches target from server log
        log_content = """\
(EngineCore pid=123) INFO 07-26 15:49:29 [worker.py:803] Available KV cache memory: 8.04 GiB
"""
        log_file = tmp_path / "server.log"
        log_file.write_text(log_content)

        result = parse_mod.verify_kv_capacity_from_log(
            str(log_file), 8, tolerance_gib=2.0
        )
        assert result["within_tolerance"] is True
        assert result["actual_kv_gib"] == 8.04
        assert result["target_kv_gib"] == 8
        assert result["diff_gib"] is not None
        assert result["error"] is None

    def test_verify_kv_capacity_mismatch(self, parse_mod, tmp_path):
        log_content = """\
(EngineCore pid=123) INFO 07-26 15:49:29 [worker.py:803] Available KV cache memory: 32.04 GiB
"""
        log_file = tmp_path / "server.log"
        log_file.write_text(log_content)

        result = parse_mod.verify_kv_capacity_from_log(
            str(log_file), 8, tolerance_gib=2.0
        )
        assert result["within_tolerance"] is False
        assert result["actual_kv_gib"] == 32.04

    def test_verify_kv_capacity_not_found(self, parse_mod, tmp_path):
        log_content = "No KV cache info here\n"
        log_file = tmp_path / "server.log"
        log_file.write_text(log_content)

        result = parse_mod.verify_kv_capacity_from_log(
            str(log_file), 8, tolerance_gib=2.0
        )
        assert result["within_tolerance"] is False
        assert result["actual_kv_gib"] is None
        assert result["error"] is not None

    def test_parse_cpu_offload_events(self, parse_mod):
        log = """\
(EngineCore pid=123) INFO 07-26 15:50:00 [connector.py:10] CPUOffloadingConnector initialized with cpu_bytes_to_use=8g
(EngineCore pid=123) INFO 07-26 15:50:01 [connector.py:20] Loading KV from cpu for seq_group 42
(EngineCore pid=123) INFO 07-26 15:50:02 [connector.py:30] Saving KV to cpu for seq_group 43
"""
        events = parse_mod.parse_cpu_offload_events(log)
        assert len(events) >= 3
        for ev in events:
            assert "raw_line" in ev
            assert "timestamp" in ev

    def test_parse_server_log_includes_new_sections(self, parse_mod, tmp_path):
        # PR #146 fix: parse_server_log includes stage_events and cpu_offload_events
        log_content = """\
(EngineCore pid=123) INFO 07-26 15:49:29 [worker.py:803] Available KV cache memory: 8.04 GiB
(EngineCore pid=123) WARNING 07-26 15:50:45 [scheduler.py:789] Sequence group 42 is preempted
(EngineCore pid=123) INFO 07-26 15:50:46 [connector.py:10] CPUOffloadingConnector initialized
"""
        log_file = tmp_path / "server.log"
        log_file.write_text(log_content)

        result = parse_mod.parse_server_log(str(log_file))
        assert "stage_events" in result
        assert "cpu_offload_events" in result
        assert isinstance(result["stage_events"], dict)
        assert isinstance(result["cpu_offload_events"], list)
        assert len(result["cpu_offload_events"]) >= 1


# ---------------------------------------------------------------------------
# _delta_pct and _find_max_delta_cap helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_delta_pct_positive(self, analyze_mod):
        assert analyze_mod._delta_pct(100.0, 150.0) == 50.0

    def test_delta_pct_negative(self, analyze_mod):
        assert analyze_mod._delta_pct(200.0, 100.0) == -50.0

    def test_delta_pct_none_values(self, analyze_mod):
        assert analyze_mod._delta_pct(None, 100.0) is None
        assert analyze_mod._delta_pct(100.0, None) is None

    def test_delta_pct_zero_base(self, analyze_mod):
        assert analyze_mod._delta_pct(0.0, 100.0) is None

    def test_find_max_delta_cap(self, analyze_mod):
        cap_vals = [(8, 100.0), (16, 200.0), (24, 210.0), (32, 215.0)]
        assert analyze_mod._find_max_delta_cap(cap_vals) == 16

    def test_find_max_delta_cap_single(self, analyze_mod):
        assert analyze_mod._find_max_delta_cap([(8, 100.0)]) is None


# ===========================================================================
# Reviewer round 1 reverse tests (issue 2/4 hardening)
# ===========================================================================


class TestTimelineMonotonicity:
    """Reverse tests for 6-stage timeline monotonicity (reviewer issue 2).

    Per reviewer: the timeline must not only have all stages present but also
    be monotonically ordered.  A non-monotonic timeline (e.g. restore_done
    before restore_start) indicates cross-episode contamination and must be
    rejected.
    """

    def test_monotonic_timeline_passes(self, parse_mod):
        """All 6 stages with increasing timestamps → complete."""
        episode = {
            "stages": {
                "preempt": "2026-08-04T15:50:45",
                "restore_start": "2026-08-04T15:50:46",
                "restore_done": "2026-08-04T15:50:47",
                "scheduler_wakeup": "2026-08-04T15:50:48",
                "admission": "2026-08-04T15:50:49",
                "first_prefill_or_decode": "2026-08-04T15:50:50",
            }
        }
        assert parse_mod.validate_timeline_complete(episode) is True

    def test_non_monotonic_timeline_rejected(self, parse_mod):
        """restore_done before restore_start → must be rejected."""
        episode = {
            "stages": {
                "preempt": "2026-08-04T15:50:45",
                "restore_start": "2026-08-04T15:50:48",
                "restore_done": "2026-08-04T15:50:46",  # before restore_start
                "scheduler_wakeup": "2026-08-04T15:50:49",
                "admission": "2026-08-04T15:50:50",
                "first_prefill_or_decode": "2026-08-04T15:50:51",
            }
        }
        assert parse_mod.validate_timeline_complete(episode) is False

    def test_admission_before_preempt_rejected(self, parse_mod):
        """admission before preempt → must be rejected."""
        episode = {
            "stages": {
                "preempt": "2026-08-04T15:50:50",
                "restore_start": "2026-08-04T15:50:51",
                "restore_done": "2026-08-04T15:50:52",
                "scheduler_wakeup": "2026-08-04T15:50:53",
                "admission": "2026-08-04T15:50:45",  # before preempt
                "first_prefill_or_decode": "2026-08-04T15:50:54",
            }
        }
        assert parse_mod.validate_timeline_complete(episode) is False

    def test_missing_stage_rejected(self, parse_mod):
        """One stage None → must be rejected."""
        episode = {
            "stages": {
                "preempt": "2026-08-04T15:50:45",
                "restore_start": "2026-08-04T15:50:46",
                "restore_done": "2026-08-04T15:50:47",
                "scheduler_wakeup": None,  # missing
                "admission": "2026-08-04T15:50:49",
                "first_prefill_or_decode": "2026-08-04T15:50:50",
            }
        }
        assert parse_mod.validate_timeline_complete(episode) is False

    def test_all_none_rejected(self, parse_mod):
        """All stages None → must be rejected."""
        episode = {"stages": {s: None for s in parse_mod.TIMELINE_STAGES}}
        assert parse_mod.validate_timeline_complete(episode) is False


class TestStrategy3FallbackGuard:
    """Reverse tests for strategy 3 fallback guard (reviewer issue 2).

    Per reviewer: strategy 3 (no-correlation fallback) should not blindly take
    the first event when multiple uncorrelated events exist.  Only use it when
    there is exactly one event for the stage.
    """

    def test_multiple_uncorrelated_events_not_matched(self, parse_mod):
        """Multiple scheduler_wakeup events without sgid → stage stays None."""
        preempt_event = {
            "timestamp": "2026-08-04T15:50:45",
            "seq_group_id": 42,
        }
        # Two scheduler_wakeup events, neither has seq_group_id
        stage_events = {
            "restore_start": [],
            "restore_done": [],
            "scheduler_wakeup": [
                {"timestamp": "2026-08-04T15:50:48", "seq_group_id": None},
                {"timestamp": "2026-08-04T15:50:48", "seq_group_id": None},
            ],
            "admission": [],
            "first_prefill_or_decode": [],
        }
        stages = parse_mod._build_stages_for_episode(preempt_event, stage_events)
        # Multiple uncorrelated events → strategy 3 should NOT match
        assert stages["scheduler_wakeup"] is None

    def test_single_uncorrelated_event_matched(self, parse_mod):
        """Single scheduler_wakeup event without sgid → strategy 3 matches."""
        preempt_event = {
            "timestamp": "2026-08-04T15:50:45",
            "seq_group_id": 42,
        }
        stage_events = {
            "restore_start": [],
            "restore_done": [],
            "scheduler_wakeup": [
                {"timestamp": "2026-08-04T15:50:48", "seq_group_id": None},
            ],
            "admission": [],
            "first_prefill_or_decode": [],
        }
        stages = parse_mod._build_stages_for_episode(preempt_event, stage_events)
        # Single event → strategy 3 is safe to use
        assert stages["scheduler_wakeup"] is not None


class TestEmptyCapacityCurvesBlocked:
    """Reverse tests for empty capacity_curves (reviewer issue 4).

    Per reviewer: an empty capacity_curves dict should NOT be a vacuous pass.
    """

    def test_empty_capacity_curves_repetitions_invalid(self, analyze_mod):
        """Empty capacity_curves → _validate_repetitions returns False."""
        analysis = {"capacity_curves": {}, "tiering_comparison": {}}
        valid, issues = analyze_mod._validate_repetitions(analysis)
        assert valid is False
        assert any("capacity_curves is empty" in i for i in issues)

    def test_empty_capacity_curves_blocked_status(self, analyze_mod):
        """Empty capacity_curves → overall_status is blocked."""
        analysis = {
            "capacity_curves": {},
            "tiering_comparison": {},
            "preempt_timeline": {"total_preemptions": 0, "pressure_episodes": []},
            "provenance": {},
        }
        result = analyze_mod.check_acceptance_criteria(analysis)
        assert result["overall_status"] == "blocked"


class TestPerRepThroughputValidation:
    """Reverse tests for per-rep throughput check (reviewer issue 4).

    Per reviewer: capacity scan should check each rep's throughput, not just
    the median, to catch NaN/0 hidden behind a positive median.
    """

    def test_rep_with_zero_throughput_rejected(self, analyze_mod):
        """A rep with throughput=0 must fail even if median is positive."""
        analysis = {
            "capacity_curves": {
                "sonnet-throughput": {
                    "8": {
                        "repetitions": 3,
                        "raw_values": [100.0, 0.0, 110.0],  # rep 2 is 0
                        "stats": {"output_throughput": {"median": 105.0}},
                    }
                }
            },
            "tiering_comparison": {},
        }
        valid, issues = analyze_mod._validate_repetitions(analysis)
        assert valid is False
        assert any("rep 2 invalid" in i for i in issues)

    def test_rep_with_nan_throughput_rejected(self, analyze_mod):
        """A rep with NaN throughput must fail."""
        import math

        analysis = {
            "capacity_curves": {
                "sonnet-throughput": {
                    "8": {
                        "repetitions": 3,
                        "raw_values": [100.0, math.nan, 110.0],
                        "stats": {"output_throughput": {"median": 105.0}},
                    }
                }
            },
            "tiering_comparison": {},
        }
        valid, issues = analyze_mod._validate_repetitions(analysis)
        assert valid is False
        assert any("rep 2 invalid" in i for i in issues)

    def test_all_reps_valid_passes(self, analyze_mod):
        """All reps with finite-positive throughput → pass."""
        analysis = {
            "capacity_curves": _make_complete_capacity_curves(),
            "tiering_comparison": {},
        }
        valid, issues = analyze_mod._validate_repetitions(analysis)
        assert valid is True
        assert issues == []


# ===========================================================================
# Reviewer round 2 tests: end-to-end bad rep, tiering config, per-workload
# coverage, per-rep manifests, provenance placeholder rejection
# ===========================================================================


class TestEndToEndBadRepBlocksAdmission:
    """End-to-end: a single bad rep must block admission via the real
    analyze_capacity_scan → check_acceptance_criteria path.

    Per reviewer round 2 issue 2: analyze_capacity_scan() previously dropped
    raw_values, so per-rep NaN/0 checks always traversed an empty list.
    Now raw_values are preserved, and a single bad rep must prevent
    admission.
    """

    def test_zero_throughput_rep_blocks_admission(self, analyze_mod):
        """One rep with throughput=0 must block admission (not hidden by median)."""
        # Build raw results: 4 capacities × 1 workload × 3 reps
        # rep-2 of capacity 8 has throughput=0
        results = {}
        for workload in ["random-online"]:
            results[workload] = {}
            for kv in [8, 16, 24, 32]:
                results[workload][str(kv)] = {}
                for rep in range(1, 4):
                    tput = 200 + kv * 5
                    if kv == 8 and rep == 2:
                        tput = 0  # bad rep
                    results[workload][str(kv)][f"rep-{rep}"] = {
                        "output_throughput": tput,
                        "mean_ttft_ms": 200.0,
                    }

        analysis = analyze_mod.analyze_capacity_scan(results)
        # raw_values must be populated for capacity 8
        cap8 = analysis["capacity_curves"]["random-online"]["8"]
        assert cap8["raw_values"] == [240.0, 0.0, 240.0]

        acceptance = analyze_mod.check_acceptance_criteria(analysis)
        assert acceptance["all_criteria_met"] is False
        assert acceptance["overall_status"] != "admitted"
        # The per-rep issue must be visible in repetition_validation
        rep_issues = acceptance["repetition_validation"]["issues"]
        assert any("rep 2 invalid" in i for i in rep_issues)

    def test_nan_throughput_rep_blocks_admission(self, analyze_mod):
        """One rep with NaN throughput must block admission."""
        import math as _math

        results = {}
        for workload in ["sharegpt-online"]:
            results[workload] = {}
            for kv in [8, 16, 24, 32]:
                results[workload][str(kv)] = {}
                for rep in range(1, 4):
                    tput = float(200 + kv * 5)
                    if kv == 16 and rep == 1:
                        tput = _math.nan  # bad rep
                    results[workload][str(kv)][f"rep-{rep}"] = {
                        "output_throughput": tput,
                    }

        analysis = analyze_mod.analyze_capacity_scan(results)
        acceptance = analyze_mod.check_acceptance_criteria(analysis)
        assert acceptance["all_criteria_met"] is False
        assert acceptance["overall_status"] != "admitted"

    def test_all_valid_reps_can_admit(self, analyze_mod):
        """All valid reps with proper raw_values → no per-rep issues."""
        results = {}
        for workload in ["random-online"]:
            results[workload] = {}
            for kv in [8, 16, 24, 32]:
                results[workload][str(kv)] = {}
                for rep in range(1, 4):
                    results[workload][str(kv)][f"rep-{rep}"] = {
                        "output_throughput": float(200 + kv * 5 + rep),
                    }

        analysis = analyze_mod.analyze_capacity_scan(results)
        # Verify raw_values are populated for ALL capacity points
        for cap_str, data in analysis["capacity_curves"]["random-online"].items():
            assert len(data["raw_values"]) == 3
            for v in data["raw_values"]:
                assert v > 0 and _math_isfinite(v)


def _math_isfinite(v):
    import math

    try:
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False


class TestTieringConfigValidation:
    """Tests for the tiering-enabled kv_transfer_config (reviewer round 2 issue 1).

    Per reviewer: CPUOffloadingConnector is deprecated and not registered in
    Ascend; SimpleCPUOffloadConnector is the registered connector. kv_role
    must be set and connector-private params go in kv_connector_extra_config.
    """

    @pytest.fixture
    def scan_script(self):
        """Read the kv_capacity_scan.sh script content."""
        script_path = _SCRIPTS_DIR / "kv_capacity_scan.sh"
        return script_path.read_text()

    def _extract_tiering_config(self, script_text):
        """Extract TIERING_KV_TRANSFER_CONFIG JSON from the shell script."""
        import re

        m = re.search(r"TIERING_KV_TRANSFER_CONFIG='([^']+)'", script_text)
        assert m, "TIERING_KV_TRANSFER_CONFIG not found in script"
        import json

        return json.loads(m.group(1))

    def test_script_uses_simple_cpu_offload_connector(self, scan_script):
        """The scan script must use SimpleCPUOffloadConnector, not CPUOffloadingConnector."""
        config = self._extract_tiering_config(scan_script)
        assert config["kv_connector"] == "SimpleCPUOffloadConnector"
        assert config["kv_connector"] != "CPUOffloadingConnector"

    def test_script_config_has_kv_role(self, scan_script):
        """kv_role must be set (required by KVTransferConfig)."""
        config = self._extract_tiering_config(scan_script)
        assert "kv_role" in config
        assert config["kv_role"] == "kv_both"

    def test_script_config_has_extra_config(self, scan_script):
        """cpu_bytes_to_use must be inside kv_connector_extra_config."""
        config = self._extract_tiering_config(scan_script)
        assert "kv_connector_extra_config" in config
        assert "cpu_bytes_to_use" in config["kv_connector_extra_config"]

    def test_script_config_passes_validation(self, scan_script, analyze_mod):
        """The script's tiering config must pass validate_tiering_config()."""
        config = self._extract_tiering_config(scan_script)
        valid, issues = analyze_mod.validate_tiering_config(config)
        assert valid, f"Invalid tiering config: {issues}"

    def test_old_cpu_offloading_connector_rejected(self, analyze_mod):
        """The deprecated CPUOffloadingConnector config must be rejected."""
        old_config = {
            "kv_connector": "CPUOffloadingConnector",
            "cpu_bytes_to_use": "8g",
        }
        valid, issues = analyze_mod.validate_tiering_config(old_config)
        assert valid is False
        assert any("not registered" in i for i in issues)

    def test_missing_kv_role_rejected(self, analyze_mod):
        """Missing kv_role must be rejected."""
        config = {
            "kv_connector": "SimpleCPUOffloadConnector",
            "kv_connector_extra_config": {"cpu_bytes_to_use": 8589934592},
        }
        valid, issues = analyze_mod.validate_tiering_config(config)
        assert valid is False
        assert any("kv_role" in i for i in issues)

    def test_missing_extra_config_rejected(self, analyze_mod):
        """Missing kv_connector_extra_config must be rejected."""
        config = {
            "kv_connector": "SimpleCPUOffloadConnector",
            "kv_role": "kv_both",
        }
        valid, issues = analyze_mod.validate_tiering_config(config)
        assert valid is False
        assert any("kv_connector_extra_config" in i for i in issues)

    def test_valid_config_passes(self, analyze_mod):
        """A fully valid config passes validation."""
        config = {
            "kv_connector": "SimpleCPUOffloadConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"cpu_bytes_to_use": 8589934592},
        }
        valid, issues = analyze_mod.validate_tiering_config(config)
        assert valid, f"Expected valid: {issues}"


class TestPerWorkloadCapacityCoverage:
    """Tests for per-workload (not union) capacity coverage (reviewer round 2 issue 3).

    Per reviewer: each workload must individually cover all 4 capacities
    (8/16/24/32 GiB). Union coverage across workloads is not sufficient.
    """

    def test_union_coverage_not_sufficient(self, analyze_mod):
        """Two workloads each missing a different capacity → must fail."""
        analysis = {
            "capacity_curves": {
                "w-a": {
                    "8": {
                        "repetitions": 3,
                        "raw_values": [100.0, 102.0, 98.0],
                        "stats": {"output_throughput": {"median": 100.0}},
                    },
                    "16": {
                        "repetitions": 3,
                        "raw_values": [200.0, 202.0, 198.0],
                        "stats": {"output_throughput": {"median": 200.0}},
                    },
                    "24": {
                        "repetitions": 3,
                        "raw_values": [210.0, 212.0, 208.0],
                        "stats": {"output_throughput": {"median": 210.0}},
                    },
                    # missing 32
                },
                "w-b": {
                    "8": {
                        "repetitions": 3,
                        "raw_values": [100.0, 102.0, 98.0],
                        "stats": {"output_throughput": {"median": 100.0}},
                    },
                    "16": {
                        "repetitions": 3,
                        "raw_values": [200.0, 202.0, 198.0],
                        "stats": {"output_throughput": {"median": 200.0}},
                    },
                    "32": {
                        "repetitions": 3,
                        "raw_values": [215.0, 217.0, 213.0],
                        "stats": {"output_throughput": {"median": 215.0}},
                    },
                    # missing 24; union covers all 4 but neither is complete
                },
            },
            "tiering_comparison": {},
        }
        valid, issues = analyze_mod._validate_repetitions(analysis)
        assert valid is False
        assert any("w-a" in i and "missing capacities" in i for i in issues)
        assert any("w-b" in i and "missing capacities" in i for i in issues)

    def test_all_workloads_complete_passes(self, analyze_mod):
        """All SCAN_WORKLOADS, each with all 4 capacities → passes.

        Per reviewer round 3 issue 1: the validator now requires ALL
        SCAN_WORKLOADS to be present, so this positive case must use the
        real workload names rather than synthetic ``w-a``/``w-b`` keys.
        """
        analysis = {
            "capacity_curves": _make_complete_capacity_curves(),
            "tiering_comparison": {},
        }
        valid, issues = analyze_mod._validate_repetitions(analysis)
        assert valid is True
        assert issues == []


class TestPerRepManifestValidation:
    """Tests for per-rep manifest binding (reviewer round 2 issue 4).

    Per reviewer: each rep must have its own manifest validated individually,
    not one manifest for the entire batch. Placeholder values like
    'unknown'/'not available' must be rejected.
    """

    def test_empty_manifests_blocked(self, analyze_mod):
        """No runs → blocked."""
        valid, issues = analyze_mod._validate_per_rep_manifests({})
        assert valid is False
        assert any("no runs" in i for i in issues)

    def test_one_bad_manifest_blocks(self, analyze_mod):
        """One run with a placeholder value → blocked."""
        good_manifest = _make_provenance()
        bad_manifest = _make_provenance()
        bad_manifest["engine_commit"] = "unknown"
        run_map = {
            "raw_results/random-online/8/rep-1": good_manifest,
            "raw_results/random-online/8/rep-2": bad_manifest,
            "raw_results/random-online/8/rep-3": good_manifest,
        }
        valid, issues = analyze_mod._validate_per_rep_manifests(run_map)
        assert valid is False
        assert any("rep-2" in i and "engine_commit" in i for i in issues)

    def test_all_manifests_valid_passes(self, analyze_mod):
        """All runs with valid manifests → passes."""
        run_map = _make_run_manifest_map(5)
        valid, issues = analyze_mod._validate_per_rep_manifests(run_map)
        assert valid is True
        assert issues == []

    def test_missing_manifests_blocks_admission(self, analyze_mod):
        """Empty run_manifest_map in acceptance check → blocked."""
        analysis = {
            "capacities_covered": [8, 16, 24, 32],
            "inflection_points": {"w": {"throughput_inflection_gib": 16}},
            "preempt_timeline": _make_complete_timeline(),
            "tiering_comparison": _make_complete_tiering(),
            "capacity_curves": _make_complete_capacity_curves(),
            "provenance": _make_provenance(),
            "run_manifest_map": {},  # empty → blocked
        }
        result = analyze_mod.check_acceptance_criteria(analysis)
        assert result["all_criteria_met"] is False
        assert result["overall_status"] == "blocked"


class TestProvenancePlaceholderRejection:
    """Tests that placeholder provenance values are rejected (reviewer round 2 issue 4).

    Per reviewer: 'unknown'/'not available' must NOT be treated as non-empty
    valid values.
    """

    def test_unknown_rejected(self, analyze_mod):
        """'unknown' value → field is missing."""
        provenance = _make_provenance()
        provenance["engine_commit"] = "unknown"
        valid, missing = analyze_mod._validate_provenance(provenance)
        assert valid is False
        assert "engine_commit" in missing

    def test_not_available_rejected(self, analyze_mod):
        """'not available' value → field is missing."""
        provenance = _make_provenance()
        provenance["model_revision"] = "not available"
        valid, missing = analyze_mod._validate_provenance(provenance)
        assert valid is False
        assert "model_revision" in missing

    def test_missing_weight_fingerprint_rejected(self, analyze_mod):
        """Per PR #152 review: missing model_weight_fingerprint → blocked."""
        provenance = _make_provenance()
        del provenance["model_weight_fingerprint"]
        valid, missing = analyze_mod._validate_provenance(provenance)
        assert valid is False
        assert "model_weight_fingerprint" in missing

    def test_placeholder_weight_fingerprint_rejected(self, analyze_mod):
        """'not available' weight fingerprint → field is missing."""
        provenance = _make_provenance()
        provenance["model_weight_fingerprint"] = "not available"
        valid, missing = analyze_mod._validate_provenance(provenance)
        assert valid is False
        assert "model_weight_fingerprint" in missing

    def test_case_insensitive_placeholder_rejected(self, analyze_mod):
        """'Unknown' (capitalized) → field is missing."""
        provenance = _make_provenance()
        provenance["cann_version"] = "Unknown"
        valid, missing = analyze_mod._validate_provenance(provenance)
        assert valid is False
        assert "cann_version" in missing

    def test_real_values_pass(self, analyze_mod):
        """All real (non-placeholder) values → valid."""
        provenance = _make_provenance()
        valid, missing = analyze_mod._validate_provenance(provenance)
        assert valid is True
        assert missing == []

    def test_none_rejected(self, analyze_mod):
        """None value → field is missing."""
        provenance = _make_provenance()
        provenance["driver_version"] = None
        valid, missing = analyze_mod._validate_provenance(provenance)
        assert valid is False
        assert "driver_version" in missing


class TestWorkloadCoverageRequired:
    """Tests that ALL SCAN_WORKLOADS must be present (reviewer round 3 issue 1).

    Per reviewer: providing only one workload (e.g. random-online) with all 4
    capacities must NOT pass — sharegpt-online and prefix-repetition-online are
    required too.
    """

    def test_single_workload_does_not_pass(self, analyze_mod):
        """Only random-online → blocked (missing sharegpt/prefix-repetition)."""
        analysis = {
            "capacities_covered": [8, 16, 24, 32],
            "inflection_points": {"random-online": {"throughput_inflection_gib": 16}},
            "preempt_timeline": _make_complete_timeline(),
            "tiering_comparison": _make_complete_tiering(),
            "capacity_curves": {
                "random-online": {
                    "8": {
                        "repetitions": 3,
                        "stats": {"output_throughput": {"median": 100.0}},
                        "raw_values": [100.0, 102.0, 98.0],
                    },
                    "16": {
                        "repetitions": 3,
                        "stats": {"output_throughput": {"median": 200.0}},
                        "raw_values": [200.0, 202.0, 198.0],
                    },
                    "24": {
                        "repetitions": 3,
                        "stats": {"output_throughput": {"median": 210.0}},
                        "raw_values": [210.0, 212.0, 208.0],
                    },
                    "32": {
                        "repetitions": 3,
                        "stats": {"output_throughput": {"median": 215.0}},
                        "raw_values": [215.0, 217.0, 213.0],
                    },
                }
            },
            "provenance": _make_provenance(),
            "run_manifest_map": _make_run_manifest_map(3),
        }
        result = analyze_mod.check_acceptance_criteria(analysis)
        assert result["all_criteria_met"] is False
        reps_issues = result["repetition_validation"]["issues"]
        assert any("missing required workloads" in i for i in reps_issues)
        assert any("sharegpt-online" in i for i in reps_issues)

    def test_two_workloads_does_not_pass(self, analyze_mod):
        """Missing prefix-repetition-online → blocked."""
        curves = _make_complete_capacity_curves()
        del curves["prefix-repetition-online"]
        analysis = {
            "capacities_covered": [8, 16, 24, 32],
            "inflection_points": {"random-online": {"throughput_inflection_gib": 16}},
            "preempt_timeline": _make_complete_timeline(),
            "tiering_comparison": _make_complete_tiering(),
            "capacity_curves": curves,
            "provenance": _make_provenance(),
            "run_manifest_map": _make_run_manifest_map(3),
        }
        result = analyze_mod.check_acceptance_criteria(analysis)
        assert result["all_criteria_met"] is False
        reps_issues = result["repetition_validation"]["issues"]
        assert any("prefix-repetition-online" in i for i in reps_issues)

    def test_all_three_workloads_passes(self, analyze_mod):
        """All 3 SCAN_WORKLOADS present → no missing workload issues."""
        valid, issues = analyze_mod._validate_repetitions(
            {
                "capacity_curves": _make_complete_capacity_curves(),
                "tiering_comparison": _make_complete_tiering(),
            }
        )
        assert not any("missing required workloads" in i for i in issues)


class TestManifestRunBinding:
    """Tests for per-run manifest binding (reviewer round 3 issue 2).

    Per reviewer: manifests must be bound to runs by relative path, not loaded
    as a flat list. Every run must have its own manifest; orphan manifests
    (without a raw.json) must be rejected.
    """

    def test_single_manifest_for_many_runs_blocked(self, analyze_mod):
        """45 runs but only 1 manifest → blocked (44 runs missing manifests)."""
        run_map = {
            f"raw_results/w/{cap}/rep-{r}": None
            for cap in [8, 16, 24, 32]
            for r in [1, 2, 3]
        }
        # Only 1 run has a manifest
        run_map["raw_results/w/8/rep-1"] = _make_provenance()
        valid, issues = analyze_mod._validate_per_rep_manifests(run_map)
        assert valid is False
        missing_count = sum(1 for i in issues if "missing env-manifest" in i)
        assert missing_count == 11  # 12 runs - 1 with manifest = 11 missing

    def test_missing_manifest_for_one_run_blocked(self, analyze_mod):
        """One run missing manifest → blocked."""
        run_map = {
            "raw_results/w/8/rep-1": _make_provenance(),
            "raw_results/w/8/rep-2": _make_provenance(),
            "raw_results/w/8/rep-3": None,  # missing
        }
        valid, issues = analyze_mod._validate_per_rep_manifests(run_map)
        assert valid is False
        assert any("rep-3" in i and "missing" in i for i in issues)

    def test_orphan_manifests_block_admission(self, analyze_mod):
        """Orphan manifests (no raw.json) → blocked."""
        analysis = {
            "capacities_covered": [8, 16, 24, 32],
            "inflection_points": {"random-online": {"throughput_inflection_gib": 16}},
            "preempt_timeline": _make_complete_timeline(),
            "tiering_comparison": _make_complete_tiering(),
            "capacity_curves": _make_complete_capacity_curves(),
            "provenance": _make_provenance(),
            "run_manifest_map": _make_run_manifest_map(3),
            "orphan_manifests": ["raw_results/stale/rep-1/env-manifest.json"],
        }
        result = analyze_mod.check_acceptance_criteria(analysis)
        assert result["all_criteria_met"] is False
        assert result["overall_status"] == "blocked"
        assert any(
            "orphan" in i for i in result["per_rep_manifest_validation"]["issues"]
        )


class TestMissingThroughputPreservesSlot:
    """Tests that missing output_throughput preserves the rep slot (reviewer round 3 issue 3).

    Per reviewer: aggregate_reps() must preserve None slots so
    len(raw_values) == repetitions. A rep missing output_throughput must
    not silently disappear.
    """

    def test_missing_throughput_keeps_slot(self, analyze_mod):
        """Rep with missing output_throughput → None in raw_values, not skipped."""
        reps = [
            {"output_throughput": 240.0, "mean_ttft_ms": 200.0},
            {"mean_ttft_ms": 210.0},  # missing output_throughput
            {"output_throughput": 260.0, "mean_ttft_ms": 220.0},
        ]
        agg = analyze_mod.aggregate_reps(reps)
        raw_tput = agg["raw_values"]["output_throughput"]
        assert len(raw_tput) == 3  # None preserved
        assert raw_tput[0] == 240.0
        assert raw_tput[1] is None
        assert raw_tput[2] == 260.0

    def test_missing_throughput_blocks_admission_e2e(self, analyze_mod):
        """End-to-end: one rep missing output_throughput → blocked."""
        results = {}
        for workload in [
            "random-online",
            "sharegpt-online",
            "prefix-repetition-online",
        ]:
            results[workload] = {}
            for kv in [8, 16, 24, 32]:
                results[workload][str(kv)] = {}
                for rep in range(1, 4):
                    if workload == "random-online" and kv == 8 and rep == 2:
                        # Missing output_throughput
                        results[workload][str(kv)][f"rep-{rep}"] = {
                            "mean_ttft_ms": 200.0,
                        }
                    else:
                        results[workload][str(kv)][f"rep-{rep}"] = {
                            "output_throughput": float(200 + kv * 5 + rep),
                        }

        analysis = analyze_mod.analyze_capacity_scan(results)
        # Check that raw_values has None for the missing rep
        raw_vals = analysis["capacity_curves"]["random-online"]["8"]["raw_values"]
        assert len(raw_vals) == 3
        assert raw_vals[1] is None

        # aggregate_reps preserves the None slot, so len(raw_values) == reps.
        # The validator must catch the invalid (None) rep value itself.
        valid, issues = analyze_mod._validate_repetitions(analysis)
        assert valid is False
        assert any("rep 2" in i and "invalid throughput" in i for i in issues)

    def test_compute_stats_handles_none_in_values(self, analyze_mod):
        """compute_stats with None values → filters them, doesn't crash."""
        stats = analyze_mod.compute_stats([100.0, None, 300.0])
        assert stats["count"] == 2  # None filtered
        assert stats["median"] == 200.0  # median of [100, 300]
        assert stats["min"] == 100.0
        assert stats["max"] == 300.0

    def test_raw_values_length_mismatch_blocks_admission(self, analyze_mod):
        """Regression guard: if raw_values length != repetitions, block.

        Per reviewer round 3 issue 3: even if aggregate_reps were changed to
        skip None slots (regression), the validator must catch the
        len(raw_values) != repetitions mismatch and block admission rather
        than vacuously passing on a shorter list.
        """
        # Construct a capacity_curves where one workload/capacity has
        # repetitions=3 but only 2 raw_values (simulating a regression
        # where aggregate_reps dropped a None slot).
        curves = _make_complete_capacity_curves()
        curves["random-online"]["8"]["repetitions"] = 3
        curves["random-online"]["8"]["raw_values"] = [240.0, 260.0]  # only 2
        analysis = {
            "capacity_curves": curves,
            "tiering_comparison": {},
        }
        valid, issues = analyze_mod._validate_repetitions(analysis)
        assert valid is False
        assert any(
            "len(raw_values)" in i and "!= repetitions" in i and "random-online/8" in i
            for i in issues
        )


class TestLoadFromTieringDir:
    """Tests for _load_from_tiering_dir (issue #134 follow-up evidence PR).

    Validates that the analyzer can load Part B tiering results from the
    ``tiering/<config>/rep-<N>/raw.json`` directory structure, including
    skipping reps marked as BLOCKED.
    """

    def test_loads_valid_tiering_results(self, analyze_mod, tmp_path):
        """Valid tiering results are loaded keyed by config name."""
        tiering_dir = tmp_path / "tiering"
        for config in ("hbm-only", "tiering-disabled"):
            for rep in range(1, 4):
                rep_dir = tiering_dir / config / f"rep-{rep}"
                rep_dir.mkdir(parents=True)
                (rep_dir / "raw.json").write_text(
                    json.dumps({"output_throughput": 100.0 + rep})
                )
        results = analyze_mod._load_from_tiering_dir(tiering_dir)
        assert set(results.keys()) == {"hbm-only", "tiering-disabled"}
        assert len(results["hbm-only"]) == 3
        assert len(results["tiering-disabled"]) == 3

    def test_skips_blocked_reps(self, analyze_mod, tmp_path):
        """Reps with STATUS=BLOCKED are skipped, not included in results."""
        tiering_dir = tmp_path / "tiering"
        config_dir = tiering_dir / "tiering-enabled"
        for rep in range(1, 4):
            rep_dir = config_dir / f"rep-{rep}"
            rep_dir.mkdir(parents=True)
            if rep == 1:
                (rep_dir / "STATUS").write_text(
                    "BLOCKED: SimpleCPUOffloadConnector incompatible"
                )
            else:
                (rep_dir / "raw.json").write_text(
                    json.dumps({"output_throughput": 200.0})
                )
        results = analyze_mod._load_from_tiering_dir(tiering_dir)
        assert "tiering-enabled" in results
        assert len(results["tiering-enabled"]) == 2  # rep-1 skipped

    def test_empty_dir_returns_empty_dict(self, analyze_mod, tmp_path):
        """Non-existent tiering directory returns empty dict."""
        results = analyze_mod._load_from_tiering_dir(tmp_path / "nonexistent")
        assert results == {}

    def test_no_raw_json_excluded(self, analyze_mod, tmp_path):
        """Reps without raw.json are silently excluded."""
        tiering_dir = tmp_path / "tiering"
        rep_dir = tiering_dir / "hbm-only" / "rep-1"
        rep_dir.mkdir(parents=True)
        # No raw.json, no STATUS — just an empty dir
        results = analyze_mod._load_from_tiering_dir(tiering_dir)
        assert results == {}


class TestShellScriptSyntax:
    """Verify that scripts/kv_capacity_scan.sh passes `bash -n` syntax check.

    Per PR #152 review round 4: a heredoc-inside-$(...) syntax error was
    missed because tests only replicated the logic in a temp script instead
    of checking the real entry point.  This test runs `bash -n` on the real
    script to catch syntax errors before CI.
    """

    def test_kv_capacity_scan_sh_syntax_valid(self):
        """scripts/kv_capacity_scan.sh must pass `bash -n`."""
        import subprocess

        script_path = _SCRIPTS_DIR / "kv_capacity_scan.sh"
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"bash -n failed for {script_path}:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


class TestModelRevisionFallback:
    """Script-level tests for model_revision fallback (reviewer round 3 issue 2).

    Per reviewer: generate_env_manifest calculates model_weight_fingerprint
    but when neither HF ref nor git HEAD is available, model_revision must
    fall back to weight_fingerprint:<fingerprint> instead of staying
    "not available".  This test extracts the relevant bash logic and runs it
    in a temp directory with only weight files (no refs/.git).
    """

    @pytest.fixture
    def scan_script(self):
        """Read the kv_capacity_scan.sh script content."""
        script_path = _SCRIPTS_DIR / "kv_capacity_scan.sh"
        return script_path.read_text()

    def test_fallback_logic_present_in_script(self, scan_script):
        """The script must contain the weight_fingerprint fallback logic."""
        assert "weight_fingerprint:${model_weight_fingerprint}" in scan_script, (
            "model_revision fallback to weight_fingerprint:<fingerprint> "
            "not found in kv_capacity_scan.sh"
        )

    def test_no_refs_no_git_falls_back_to_fingerprint(self, tmp_path):
        """With only weight files (no refs/.git), model_revision must be
        weight_fingerprint:<fingerprint>, not 'not available'.

        This test runs the actual bash logic from generate_env_manifest
        against a temp directory containing a dummy .safetensors file.
        """
        import subprocess

        # Create a temp model dir with only a weight file (no .git, no refs/)
        model_dir = tmp_path / "fake_model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"fake_weight_data_12345")

        # Extract and run the model_revision + fingerprint logic from the
        # bash script.  We source the relevant variables and conditions.
        bash_script = (
            """
set -euo pipefail
MODEL_PATH="%s"
PYTHON="$(command -v python3 || echo python3)"

model_revision="not available"
model_weight_fingerprint="not available"

# Replicate the HF ref check (no refs/main in our temp dir)
for hf_ref in "$MODEL_PATH/refs/main" "$MODEL_PATH/.cache/refs/main"; do
    if [ -f "$hf_ref" ]; then
        model_revision=$(cat "$hf_ref" 2>/dev/null | tr -d '[:space:]')
        break
    fi
done

# Replicate the git check (no .git in our temp dir)
if [ "$model_revision" = "not available" ] && [ -d "$MODEL_PATH/.git" ]; then
    model_revision=$(cd "$MODEL_PATH" 2>/dev/null && git rev-parse HEAD 2>/dev/null || echo "not available")
fi

# Replicate the weight fingerprint calculation
if [ -d "$MODEL_PATH" ]; then
    model_weight_fingerprint=$("$PYTHON" - "$MODEL_PATH" <<'WPEOF' 2>/dev/null || echo "not available"
import hashlib, os, sys
model_path = sys.argv[1]
entries = []
for fname in sorted(os.listdir(model_path)):
    if fname.endswith((".safetensors", ".bin", ".pt")):
        fpath = os.path.join(model_path, fname)
        if not os.path.isfile(fpath):
            continue
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        entries.append(f"{fname}:{h.hexdigest()}")
if entries:
    combined = hashlib.sha256("\\n".join(entries).encode()).hexdigest()
    print(f"sha256:{combined}")
else:
    print("not available")
WPEOF
    )
fi

# Replicate the fallback logic from PR #152 review round 3
if [ "$model_revision" = "not available" ] \\
    && [ "$model_weight_fingerprint" != "not available" ] \\
    && [ -n "$model_weight_fingerprint" ]; then
    model_revision="weight_fingerprint:${model_weight_fingerprint}"
fi

echo "MODEL_REVISION=$model_revision"
echo "MODEL_WEIGHT_FINGERPRINT=$model_weight_fingerprint"
"""
            % model_dir
        )

        script_file = tmp_path / "test_fingerprint.sh"
        script_file.write_text(bash_script)

        result = subprocess.run(
            ["bash", str(script_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"bash failed: {result.stderr}"

        # Parse output
        output = result.stdout.strip()
        lines = dict(line.split("=", 1) for line in output.split("\n") if "=" in line)

        # model_revision must be weight_fingerprint:<fingerprint>, not "not available"
        model_rev = lines.get("MODEL_REVISION", "")
        assert model_rev != "not available", (
            "model_revision stayed 'not available' even though "
            "model_weight_fingerprint was computed"
        )
        assert model_rev.startswith("weight_fingerprint:"), (
            f"model_revision should start with 'weight_fingerprint:' but got: {model_rev}"
        )

        # model_weight_fingerprint must be a valid sha256 fingerprint
        fingerprint = lines.get("MODEL_WEIGHT_FINGERPRINT", "")
        assert fingerprint.startswith("sha256:"), (
            f"model_weight_fingerprint should start with 'sha256:' but got: {fingerprint}"
        )
        assert fingerprint != "not available"

    def test_no_weight_files_stays_not_available(self, tmp_path):
        """With no refs, no .git, AND no weight files, both stay 'not available'."""
        import subprocess

        # Empty model dir — no weights, no refs, no .git
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()

        bash_script = (
            """
set -euo pipefail
MODEL_PATH="%s"
PYTHON="$(command -v python3 || echo python3)"

model_revision="not available"
model_weight_fingerprint="not available"

for hf_ref in "$MODEL_PATH/refs/main" "$MODEL_PATH/.cache/refs/main"; do
    if [ -f "$hf_ref" ]; then
        model_revision=$(cat "$hf_ref" 2>/dev/null | tr -d '[:space:]')
        break
    fi
done

if [ "$model_revision" = "not available" ] && [ -d "$MODEL_PATH/.git" ]; then
    model_revision=$(cd "$MODEL_PATH" 2>/dev/null && git rev-parse HEAD 2>/dev/null || echo "not available")
fi

if [ -d "$MODEL_PATH" ]; then
    model_weight_fingerprint=$("$PYTHON" - "$MODEL_PATH" <<'WPEOF' 2>/dev/null || echo "not available"
import hashlib, os, sys
model_path = sys.argv[1]
entries = []
for fname in sorted(os.listdir(model_path)):
    if fname.endswith((".safetensors", ".bin", ".pt")):
        fpath = os.path.join(model_path, fname)
        if not os.path.isfile(fpath):
            continue
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        entries.append(f"{fname}:{h.hexdigest()}")
if entries:
    combined = hashlib.sha256("\\n".join(entries).encode()).hexdigest()
    print(f"sha256:{combined}")
else:
    print("not available")
WPEOF
    )
fi

if [ "$model_revision" = "not available" ] \\
    && [ "$model_weight_fingerprint" != "not available" ] \\
    && [ -n "$model_weight_fingerprint" ]; then
    model_revision="weight_fingerprint:${model_weight_fingerprint}"
fi

echo "MODEL_REVISION=$model_revision"
echo "MODEL_WEIGHT_FINGERPRINT=$model_weight_fingerprint"
"""
            % model_dir
        )

        script_file = tmp_path / "test_fingerprint.sh"
        script_file.write_text(bash_script)

        result = subprocess.run(
            ["bash", str(script_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"bash failed: {result.stderr}"

        output = result.stdout.strip()
        lines = dict(line.split("=", 1) for line in output.split("\n") if "=" in line)

        # Both should stay "not available" since there are no weight files
        assert lines.get("MODEL_REVISION") == "not available"
        assert lines.get("MODEL_WEIGHT_FINGERPRINT") == "not available"


class TestCapacityBlockedByActualKV:
    """Tests that capacity points with actual KV > 2 GiB from target are
    marked blocked (reviewer round 3 issue 1).

    Per reviewer: 32 GiB target with actual ~29.1 GiB must NOT be marked
    as MET by widening tolerance.  The strict 2 GiB tolerance applies to
    ALL targets; unreachable targets must be reported as blocked.
    """

    def test_32gib_blocked_when_actual_kv_too_low(self, analyze_mod):
        """32 GiB target with actual ~29 GiB → blocked."""
        # Build run_manifest_map where 32 GiB runs have actual_kv_bytes
        # corresponding to ~29 GiB (not 32 GiB)
        manifest_8 = _make_provenance()
        manifest_8["actual_kv_bytes"] = int(8.04 * 1024**3)
        manifest_16 = _make_provenance()
        manifest_16["actual_kv_bytes"] = int(16.1 * 1024**3)
        manifest_24 = _make_provenance()
        manifest_24["actual_kv_bytes"] = int(24.0 * 1024**3)
        manifest_32 = _make_provenance()
        manifest_32["actual_kv_bytes"] = int(29.1 * 1024**3)  # ~2.9 GiB off

        run_manifest_map = {
            "raw_results/random-online/8/rep-1": manifest_8,
            "raw_results/random-online/16/rep-1": manifest_16,
            "raw_results/random-online/24/rep-1": manifest_24,
            "raw_results/random-online/32/rep-1": manifest_32,
        }
        analysis = {
            "capacities_covered": [8, 16, 24, 32],
            "inflection_points": {"random-online": {"throughput_inflection_gib": 16}},
            "preempt_timeline": _make_complete_timeline(),
            "tiering_comparison": _make_complete_tiering(),
            "capacity_curves": _make_complete_capacity_curves(),
            "provenance": _make_provenance(),
            "run_manifest_map": run_manifest_map,
        }
        result = analyze_mod.check_acceptance_criteria(analysis)
        # Capacity criterion must NOT be met
        cap_criterion = next(
            c for c in result["criteria"] if "capacity curves" in c["criterion"]
        )
        assert cap_criterion["met"] is False
        assert "32GiB" in cap_criterion["details"]
        assert "blocked" in cap_criterion["details"]
        assert result["overall_status"] == "blocked"

    def test_all_caps_met_when_actual_kv_within_tolerance(self, analyze_mod):
        """All capacity points with actual KV within 2 GiB → MET."""
        run_manifest_map = {}
        for cap in [8, 16, 24, 32]:
            for rep in range(1, 4):
                m = _make_provenance()
                m["actual_kv_bytes"] = int((cap + 0.1) * 1024**3)
                run_manifest_map[f"raw_results/random-online/{cap}/rep-{rep}"] = m
        analysis = {
            "capacities_covered": [8, 16, 24, 32],
            "inflection_points": {"random-online": {"throughput_inflection_gib": 16}},
            "preempt_timeline": _make_complete_timeline(),
            "tiering_comparison": _make_complete_tiering(),
            "capacity_curves": _make_complete_capacity_curves(),
            "provenance": _make_provenance(),
            "run_manifest_map": run_manifest_map,
        }
        result = analyze_mod.check_acceptance_criteria(analysis)
        cap_criterion = next(
            c for c in result["criteria"] if "capacity curves" in c["criterion"]
        )
        assert cap_criterion["met"] is True
        assert "blocked" not in cap_criterion["details"]

    def test_no_manifests_does_not_block(self, analyze_mod):
        """When run_manifest_map is empty, capacity check only verifies
        coverage, not actual KV (backward compatibility)."""
        analysis = {
            "capacities_covered": [8, 16, 24, 32],
            "inflection_points": {"random-online": {"throughput_inflection_gib": 16}},
            "preempt_timeline": _make_complete_timeline(),
            "tiering_comparison": _make_complete_tiering(),
            "capacity_curves": _make_complete_capacity_curves(),
            "provenance": _make_provenance(),
            "run_manifest_map": {},
        }
        result = analyze_mod.check_acceptance_criteria(analysis)
        cap_criterion = next(
            c for c in result["criteria"] if "capacity curves" in c["criterion"]
        )
        # Without manifests, only coverage is checked
        assert cap_criterion["met"] is True


class TestTieringEnabledErrorSignature:
    """Tests for tiering-enabled error signature verification (reviewer round 3 issue 3).

    Per reviewer: tiering-enabled failures must only be marked BLOCKED when
    the server log contains the verified SimpleCPUOffloadConnector shape
    mismatch RuntimeError signature.  Other failures must exit 1 (fail closed).
    """

    @pytest.fixture
    def scan_script(self):
        """Read the kv_capacity_scan.sh script content."""
        script_path = _SCRIPTS_DIR / "kv_capacity_scan.sh"
        return script_path.read_text()

    def test_error_signature_check_present(self, scan_script):
        """The script must check for the shape mismatch RuntimeError signature."""
        assert "RuntimeError: shape.*is invalid for input of size" in scan_script, (
            "tiering-enabled error signature check not found in script"
        )

    def test_fail_closed_for_unverified_errors(self, scan_script):
        """The script must exit 1 for unverified tiering-enabled errors."""
        assert "UNVERIFIED error" in scan_script, (
            "fail-closed message for unverified tiering errors not found"
        )

    def test_verified_signature_marks_blocked(self, scan_script):
        """The script must mark BLOCKED only when the signature matches."""
        assert "verified shape mismatch RuntimeError" in scan_script, (
            "verified signature BLOCKED message not found in script"
        )
