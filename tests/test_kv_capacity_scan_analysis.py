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
    """Return capacity_curves with all 4 capacities, 3 reps, valid throughput."""
    return {
        "w": {
            "8": {"repetitions": 3, "stats": {"output_throughput": {"median": 100.0}}},
            "16": {"repetitions": 3, "stats": {"output_throughput": {"median": 200.0}}},
            "24": {"repetitions": 3, "stats": {"output_throughput": {"median": 210.0}}},
            "32": {"repetitions": 3, "stats": {"output_throughput": {"median": 215.0}}},
        }
    }


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
            "capacity_curves": {
                "w": {
                    "8": {
                        "repetitions": 2,
                        "stats": {"output_throughput": {"median": 100.0}},
                    },  # < 3
                    "16": {
                        "repetitions": 3,
                        "stats": {"output_throughput": {"median": 200.0}},
                    },
                    "24": {
                        "repetitions": 3,
                        "stats": {"output_throughput": {"median": 210.0}},
                    },
                    "32": {
                        "repetitions": 3,
                        "stats": {"output_throughput": {"median": 215.0}},
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
