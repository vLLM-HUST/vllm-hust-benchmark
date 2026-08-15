"""Tests for issue #177 remaining-jumps tracking report."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_issue_151_regression.py"
REPORT_PATH = REPO_ROOT / "reports" / "issue_177_remaining_jumps_report.json"

# (workload, base_commit, head_commit, model_name)
EXPECTED_INTERVALS = [
    ("agent-research-online", "f273f9c5e2", "51621c35bc", "Qwen2.5-14B-Instruct"),
    ("instructcoder-online", "51621c35bc", "7a63f81e86", "Qwen2.5-Coder-14B-Instruct"),
    ("random-online", "7a63f81e86", "ec4847981f", "Qwen2.5-14B-Instruct"),
    ("visionarena-online", "ec4847981f", "ceec19abb0", "Qwen2.5-VL-7B-Instruct"),
]

EXPECTED_TRACKING_ISSUES = {
    "agent-research-online": (
        "https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/188"
    ),
    "instructcoder-online": (
        "https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/189"
    ),
    "random-online": ("https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/190"),
    "visionarena-online": (
        "https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/191"
    ),
}

REQUIRED_INTERVAL_FIELDS = [
    "interval_id",
    "workload",
    "base_commit",
    "head_commit",
    "retest_status",
    "reported_jump",
    "original_leaderboard",
    "metric_definitions",
    "hardware",
    "model",
    "same_spec_identity",
    "server_config",
    "client_config",
    "provenance",
    "reps_completed",
    "reps_required",
    "verdict",
    "disposition",
    "disposition_reason",
    "tracking_issue",
    "related_prs",
]


@pytest.fixture(scope="module")
def analyze_mod():
    spec = importlib.util.spec_from_file_location(
        "analyze_issue_151_regression_177", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# REMAINING_INTERVALS configuration
# ---------------------------------------------------------------------------


class TestRemainingIntervalsConfig:
    def test_remaining_intervals_count(self, analyze_mod):
        assert len(analyze_mod.REMAINING_INTERVALS) == 4

    def test_remaining_intervals_status(self, analyze_mod):
        # random-online was retested (issue #190) and superseded; the other 3
        # intervals have not been retested yet and remain pending.
        for iv in analyze_mod.REMAINING_INTERVALS:
            if iv["workload"] == "random-online":
                assert iv["retest_status"] == "completed"
            else:
                assert iv["retest_status"] == "pending"

    def test_intervals_list_unchanged(self, analyze_mod):
        # The 2 retested intervals must remain untouched (compare logic intact).
        assert len(analyze_mod.INTERVALS) == 2

    def test_commits_and_models(self, analyze_mod):
        for workload, base, head, model in EXPECTED_INTERVALS:
            iv = next(
                i for i in analyze_mod.REMAINING_INTERVALS if i["workload"] == workload
            )
            assert iv["base_commit"] == base
            assert iv["head_commit"] == head
            assert iv["model"]["name"] == model

    def test_hardware_is_single_card_910b2(self, analyze_mod):
        for iv in analyze_mod.REMAINING_INTERVALS:
            assert iv["hardware"]["chip_model"] == "910B2"
            assert iv["hardware"]["chip_count"] == 1
            assert iv["hardware"]["node_count"] == 1

    def test_no_910b3_introduced(self, analyze_mod):
        for iv in analyze_mod.REMAINING_INTERVALS:
            assert iv["hardware"]["chip_model"] != "910B3"


# ---------------------------------------------------------------------------
# generate_remaining_jumps_report()
# ---------------------------------------------------------------------------


class TestGenerateRemainingJumpsReport:
    def test_report_top_level(self, analyze_mod):
        report = analyze_mod.generate_remaining_jumps_report()
        assert report["issue"] == "#177"
        assert report["follow_up_for"] == "#165"
        assert report["parent_issue"] == "vLLM-HUST/vllm-hust#151"
        assert len(report["intervals"]) == 4

    def test_required_fields_present(self, analyze_mod):
        report = analyze_mod.generate_remaining_jumps_report()
        for iv in report["intervals"]:
            for field in REQUIRED_INTERVAL_FIELDS:
                assert field in iv, f"missing field {field}"

    def test_commits_match_expected(self, analyze_mod):
        report = analyze_mod.generate_remaining_jumps_report()
        for iv, (workload, base, head, _model) in zip(
            report["intervals"], EXPECTED_INTERVALS
        ):
            assert iv["workload"] == workload
            assert iv["base_commit"] == base
            assert iv["head_commit"] == head

    def test_verdicts_by_status(self, analyze_mod):
        report = analyze_mod.generate_remaining_jumps_report()
        for iv in report["intervals"]:
            if iv["workload"] == "random-online":
                assert iv["verdict"] == "not_reproducible"
            else:
                assert iv["verdict"] == "incomplete_evidence"

    def test_dispositions_by_status(self, analyze_mod):
        report = analyze_mod.generate_remaining_jumps_report()
        for iv in report["intervals"]:
            if iv["workload"] == "random-online":
                assert iv["disposition"] == "supersede"
            else:
                assert iv["disposition"] == "rerun"

    def test_metric_definitions_complete(self, analyze_mod):
        report = analyze_mod.generate_remaining_jumps_report()
        for iv in report["intervals"]:
            md = iv["metric_definitions"]
            assert set(md) == {"ttft_ms", "tpot_ms", "throughput_tps"}
            for value in md.values():
                assert isinstance(value, str) and value

    def test_provenance_has_repos(self, analyze_mod):
        report = analyze_mod.generate_remaining_jumps_report()
        for iv in report["intervals"]:
            prov = iv["provenance"]
            assert prov["engine_repo"] == "vLLM-HUST/vllm-hust"
            assert prov["plugin_repo"] == "vLLM-HUST/vllm-ascend-hust"

    def test_reps_counts(self, analyze_mod):
        report = analyze_mod.generate_remaining_jumps_report()
        for iv in report["intervals"]:
            assert iv["reps_required"] == 3
            if iv["workload"] == "random-online":
                assert iv["reps_completed"] == 3
            else:
                assert iv["reps_completed"] == 0

    def test_summary_stats(self, analyze_mod):
        report = analyze_mod.generate_remaining_jumps_report()
        s = report["summary"]
        assert s["total_remaining_intervals"] == 4
        assert s["reproducible_regressions"] == 0
        assert s["not_reproducible"] == 1
        assert s["incomplete_evidence"] == 3
        assert s["overall_verdict"] == "one_interval_superseded_three_pending"
        assert s["disposition_summary"]["rerun"] == 3
        assert s["disposition_summary"]["retain"] == 0
        assert s["disposition_summary"]["quarantine"] == 0
        assert s["disposition_summary"]["supersede"] == 1

    def test_reported_jump_first_interval(self, analyze_mod):
        report = analyze_mod.generate_remaining_jumps_report()
        first = report["intervals"][0]
        rj = first["reported_jump"]
        assert rj["ttft_ms"] == {"base": 281, "head": 434, "change_pct": 54.2}
        assert rj["tpot_ms"] == {"base": 49.1, "head": 55.7, "change_pct": 13.4}
        assert rj["throughput_tps"] == {
            "base": 187.9,
            "head": 180.8,
            "change_pct": -3.7,
        }

    def test_original_leaderboard_matches_reported_jump(self, analyze_mod):
        report = analyze_mod.generate_remaining_jumps_report()
        for iv in report["intervals"]:
            rj = iv["reported_jump"]
            ol = iv["original_leaderboard"]
            for metric in ("ttft_ms", "tpot_ms", "throughput_tps"):
                assert ol["base"][metric] == rj[metric]["base"]
                assert ol["head"][metric] == rj[metric]["head"]

    def test_write_to_disk(self, analyze_mod, tmp_path):
        out = tmp_path / "report.json"
        analyze_mod.generate_remaining_jumps_report(str(out))
        assert out.is_file()
        data = json.loads(out.read_text())
        assert data["issue"] == "#177"
        assert len(data["intervals"]) == 4


# ---------------------------------------------------------------------------
# Committed report file on disk
# ---------------------------------------------------------------------------


class TestCommittedReportFile:
    def test_file_exists_and_valid(self):
        assert REPORT_PATH.is_file()
        data = json.loads(REPORT_PATH.read_text())
        assert data["issue"] == "#177"
        assert data["follow_up_for"] == "#165"
        assert len(data["intervals"]) == 4

    def test_file_verdicts_and_dispositions(self):
        data = json.loads(REPORT_PATH.read_text())
        for iv in data["intervals"]:
            assert iv["reps_required"] == 3
            if iv["workload"] == "random-online":
                assert iv["verdict"] == "not_reproducible"
                assert iv["disposition"] == "supersede"
                assert iv["reps_completed"] == 3
            else:
                assert iv["verdict"] == "incomplete_evidence"
                assert iv["disposition"] == "rerun"
                assert iv["reps_completed"] == 0

    def test_file_summary(self):
        data = json.loads(REPORT_PATH.read_text())
        s = data["summary"]
        assert s["incomplete_evidence"] == 3
        assert s["not_reproducible"] == 1
        assert s["disposition_summary"]["rerun"] == 3
        assert s["disposition_summary"]["supersede"] == 1

    def test_file_matches_generator(self, analyze_mod):
        on_disk = json.loads(REPORT_PATH.read_text())
        generated = analyze_mod.generate_remaining_jumps_report()
        assert len(on_disk["intervals"]) == len(generated["intervals"])
        for a, b in zip(on_disk["intervals"], generated["intervals"]):
            assert a["base_commit"] == b["base_commit"]
            assert a["head_commit"] == b["head_commit"]
            assert a["verdict"] == b["verdict"]
            assert a["disposition"] == b["disposition"]
            assert a["reported_jump"] == b["reported_jump"]


# ---------------------------------------------------------------------------
# server_config: expected vs historical captured + real tracking issues
# (PR #180 review comments)
# ---------------------------------------------------------------------------


class TestServerConfigAndTracking:
    def test_server_config_distinguishes_expected_from_historical(
        self, analyze_mod
    ) -> None:
        for iv in analyze_mod.REMAINING_INTERVALS:
            sc = iv["server_config"]
            assert "official_target_expected" in sc
            assert sc["historical_captured"] == "unknown/config-unverified"
            expected = sc["official_target_expected"]
            assert expected["max_model_len"] == 32768
            assert expected["gpu_memory_utilization"] == 0.6
            assert expected["dtype"] == "float16"
            assert expected["enforce_eager"] is False

    def test_tracking_issue_is_real_issue_link(self, analyze_mod) -> None:
        for iv in analyze_mod.REMAINING_INTERVALS:
            link = iv["tracking_issue"]
            assert link.startswith(
                "https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/"
            )

    def test_report_tracking_issue_matches_expected(self, analyze_mod) -> None:
        report = analyze_mod.generate_remaining_jumps_report()
        for iv in report["intervals"]:
            assert iv["tracking_issue"] == EXPECTED_TRACKING_ISSUES[iv["workload"]]

    def test_committed_report_server_config_and_tracking(self) -> None:
        data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        for iv in data["intervals"]:
            sc = iv["server_config"]
            assert "official_target_expected" in sc
            assert sc["historical_captured"] == "unknown/config-unverified"
            assert iv["tracking_issue"] == EXPECTED_TRACKING_ISSUES[iv["workload"]]


# ---------------------------------------------------------------------------
# compare_interval fail-closed rep contract (PR #196 review)
# ---------------------------------------------------------------------------


class TestCompareIntervalFailClosed:
    """compare_interval must not emit a verdict with fewer than reps_required
    valid reps per side (regression test for PR #196 review)."""

    @staticmethod
    def _interval() -> dict:
        return {
            "name": "test-interval",
            "base_commit": "a1b2c3d4e5",  # pragma: allowlist secret
            "head_commit": "f6a7b8c9d0",  # pragma: allowlist secret
            "workload": "random-online",
            "reported_jump": "test jump",
            "reps_required": 3,
            "absolute_value_drift_note": "",
            "original_leaderboard": {},
        }

    @staticmethod
    def _write_rep(
        rep_dir: Path,
        *,
        completed: bool = True,
        valid_manifest: bool = True,
    ) -> None:
        rep_dir.mkdir(parents=True, exist_ok=True)
        if completed:
            (rep_dir / ".completed").write_text("ok\n")
        manifest = {
            "engine_commit_observed": "e" * 40,
            "plugin_commit_observed": "f" * 40,
        }
        (rep_dir / "env-manifest.json").write_text(
            json.dumps(manifest) if valid_manifest else "{not json\n"
        )
        (rep_dir / "raw.json").write_text(
            json.dumps(
                {
                    "mean_ttft_ms": 100.0,
                    "mean_tpot_ms": 10.0,
                    "output_throughput": 100.0,
                }
            )
            + "\n"
        )

    @classmethod
    def _write_side(
        cls,
        result_dir: Path,
        commit: str,
        workload: str,
        n: int,
        **kwargs,
    ) -> None:
        for rep in range(1, n + 1):
            cls._write_rep(result_dir / commit / workload / f"rep-{rep}", **kwargs)

    def test_one_rep_per_side_fails_closed(self, analyze_mod, tmp_path) -> None:
        iv = self._interval()
        self._write_side(tmp_path, iv["base_commit"], iv["workload"], 1)
        self._write_side(tmp_path, iv["head_commit"], iv["workload"], 1)
        result = analyze_mod.compare_interval(iv, tmp_path, None, None)
        assert result["verdict"] == "incomplete_evidence"
        assert result["base_reps"] == 1
        assert result["head_reps"] == 1
        assert result["reps_required"] == 3

    def test_missing_side_fails_closed(self, analyze_mod, tmp_path) -> None:
        iv = self._interval()
        self._write_side(tmp_path, iv["base_commit"], iv["workload"], 3)
        result = analyze_mod.compare_interval(iv, tmp_path, None, None)
        assert result["verdict"] == "incomplete_evidence"
        assert result["base_reps"] == 3
        assert result["head_reps"] == 0

    def test_invalid_rep_does_not_count_toward_contract(
        self, analyze_mod, tmp_path
    ) -> None:
        iv = self._interval()
        # rep-3 lacks the .completed marker -> only 2 valid base reps.
        self._write_side(tmp_path, iv["base_commit"], iv["workload"], 2)
        self._write_rep(
            tmp_path / iv["base_commit"] / iv["workload"] / "rep-3",
            completed=False,
        )
        self._write_side(tmp_path, iv["head_commit"], iv["workload"], 3)
        result = analyze_mod.compare_interval(iv, tmp_path, None, None)
        assert result["verdict"] == "incomplete_evidence"
        assert result["base_reps"] == 2
        assert result["head_reps"] == 3

    def test_full_three_reps_per_side_emits_verdict(
        self, analyze_mod, tmp_path
    ) -> None:
        iv = self._interval()
        self._write_side(tmp_path, iv["base_commit"], iv["workload"], 3)
        self._write_side(tmp_path, iv["head_commit"], iv["workload"], 3)
        result = analyze_mod.compare_interval(iv, tmp_path, None, None)
        assert result["verdict"] in ("not_reproducible", "reproducible_regression")
        assert result["base_reps"] == 3
        assert result["head_reps"] == 3
