"""Verify r9 reference coverage independently of the plan generator."""

from collections import Counter
import hashlib
import json
import statistics

from test_midterm_vocab_prefill import BASE, archive_members


def shape(case):
    return (
        case["capacity"],
        case["requests"],
        tuple(case["lengths"]),
        tuple(case["initial_flags"]),
    )


def test_r9_reference_matches_both_raw_ranks():
    root = BASE / "gdn-reference-r3"
    members = archive_members(root)
    raw = archive_members(BASE / "device-routing-r9")
    plan_bytes = members["source/gdn-reference-plan-r3.json"]
    plan = json.loads(plan_bytes)
    receipt = json.loads(raw["microflow-token-accounting-r9/receipt.json"])
    catalog = json.loads((BASE / "device-routing-r9/same-run-catalog.json").read_text())
    assert plan["request_phase"] == catalog["request_phase"]
    expected = {shape(c): c["observed_contexts"] * 30 for c in plan["cases"]}
    assert len(expected) == len(plan["cases"]) == 126
    assert sum(expected.values()) == 132 * 30
    assert len(plan["source_events_sha256"]) == 2
    for name, digest in plan["source_events_sha256"].items():
        data = raw["microflow-token-accounting-r9/" + name]
        assert hashlib.sha256(data).hexdigest() == digest
        counts = Counter()
        for line in data.splitlines():
            event = json.loads(line)
            if event["event"] != "forward_enter" or event["role"] != "target":
                continue
            if (
                not plan["request_phase"]["start_wall"]
                <= event["wall"]
                <= receipt["bench_ended_wall"]
            ):
                continue
            for group in event["owned_gdn"]:
                if group["decode_only"]:
                    continue
                lengths = tuple(
                    b - a for a, b in zip(group["prefill_cu"], group["prefill_cu"][1:])
                )
                flags = tuple(
                    group["initial"][i]
                    for i in range(group["live"])
                    if i not in group["verify_ids"]
                )
                counts[(group["capacity"], group["requests"], lengths, flags)] += len(
                    group["layers"]
                )
        assert counts == expected
    old_bytes = members["source/gdn-reference-plan-r2.json"]
    coverage = plan["prior_reference_coverage"]
    assert hashlib.sha256(old_bytes).hexdigest() == coverage["prior_plan_sha256"]
    prior = {shape(c) for c in json.loads(old_bytes)["cases"]}
    assert len(set(expected) & prior) == coverage["matched_unique_shapes"] == 41
    assert len(set(expected) - prior) == coverage["new_unique_shapes"] == 85


def test_r9_reference_results_and_cleanup():
    root = BASE / "gdn-reference-r3"
    members = archive_members(root)
    plan_bytes = members["source/gdn-reference-plan-r3.json"]
    plan = json.loads(plan_bytes)
    summary = json.loads((root / "reference.json").read_text())
    control = json.loads(members["gdn-reference-control-r3/receipt.json"])
    assert control["probe_exit_code"] == control["cleanup_exit_code"] == 0
    assert control["all_devices_released"] and not control["sampler_errors"]
    for name, digest in control["input_hashes"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    assert summary["cases_per_rank"] == 126 and summary["observed_contexts"] == 132
    for rank in (0, 1):
        result = json.loads(members[f"gdn-reference-r3/rank-{rank}.json"])
        assert result["status"] == "qualified_component_timings"
        assert result["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
        assert (
            result["library_sha256"]
            == "1c522f04a701d3423798b4b7f157737ef41956be25cf46842b6ba33c7ec41bea"  # pragma: allowlist secret
        )
        assert len(result["cases"]) == len(plan["cases"])
        for actual, expected, derived in zip(
            result["cases"], plan["cases"], summary["ranks"][rank]["cases"]
        ):
            assert actual["shape"] == expected
            assert set(actual["checks"]) == {"output", "state"}
            assert all(c["finite"] and c["close"] for c in actual["checks"].values())
            assert len(actual["timings"]) == plan["repetitions"] == 3
            assert all(
                t["replays"] == 30 and t["device_ms"] > 0 for t in actual["timings"]
            )
            values = [t["device_ms"] / t["replays"] for t in actual["timings"]]
            assert derived["replay_ms"] == values
            assert derived["median_ms"] == statistics.median(values)
    assert summary["status"] == "qualified_component_reference"
    assert "attainment" not in summary
