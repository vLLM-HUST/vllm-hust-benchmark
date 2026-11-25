"""The outline counts total context and separates quality from stability."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stability", ROOT / "scripts/stability_assessment.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_output_can_cross_16k_without_a_16k_input():
    rows = [
        {"usage": {"prompt_tokens": 16350, "completion_tokens": 40}},
        {"usage": {"prompt_tokens": 100, "completion_tokens": 10}},
    ]
    assert MODULE.context_coverage(rows) == {
        "input_at_least16384": 0,
        "input_plus_output_at_least16384": 1,
    }


def test_empty_cohort_is_not_a_zero_latency_measurement():
    empty = MODULE.request_metrics([])
    assert empty["requests"] == empty["tpot_requests"] == 0
    assert empty["ttft_p99_ms"] is None
    assert empty["tpot_p99_ms"] is None


def test_correction_preserves_v1_and_does_not_rewrite_any_measured_metric():
    report = ROOT / "reports/five-datasets-standard-20261113"
    old = json.loads((report / "summary-public-v1.json").read_bytes())
    new = json.loads((report / "summary-public-v2.json").read_bytes())
    assert old["datasets"] == new["datasets"]
    assert old["continuous"] == new["continuous"]
    assert old["project_final_assessment"] == new["project_final_assessment"]
    assert new["revision"]["supersedes_sha256"] == MODULE.digest(
        report / "summary-public-v1.json"
    )
    audit = new["stability_assessment"]
    assert audit == json.loads((report / "stability-audit-v1.json").read_bytes())
    assert audit["total_context_includes_output"]
    assert audit["quality_is_a_separate_assessment"]
    assert audit["each_dataset_input_16k_is_not_required"]
    for name in ("swe", "terminal"):
        for arm in audit["profiles"][name].values():
            assert arm["input_at_least16384"] == 0
            assert arm["input_plus_output_at_least16384"] > 0
    for profile in audit["profiles"].values():
        for arm in profile.values():
            assert arm["failed"] == arm["unreturned"] == 0
            assert arm["length_stop_budget_mismatches"] == 0
    assert all(
        not v["error_lines_before_shutdown"] and v["completed"] and v["quiescent"]
        for v in audit["stage_logs"].values()
    )
    for role in ("B0", "B1"):
        groups = audit["continuous"][role]["groups"]
        for group in groups.values():
            assert len(group["submission_cohorts"]) == 5
            assert (
                sum(w["requests"] for w in group["submission_cohorts"])
                == group["requests"]
            )
    target = new["formal_targets"]["3.4.3"]
    assert target == MODULE.milestone_target(audit)
    assert target["quality_is_a_separate_assessment"]
    assert "裁判未决与准确率另列" in target["reason"]
    assert not target["accepted"]
