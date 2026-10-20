"""Recompute component qualification and coverage from archived raw evidence."""

from collections import Counter
import hashlib
import json
import statistics

from test_midterm_vocab_prefill import BASE, archive_members


def test_gdn_reference_all_observed_partitions():
    root = BASE / "gdn-reference-r2"
    members = archive_members(root)
    raw = archive_members(BASE / "recurrence-r8")
    plan_bytes = members["source/gdn-reference-plan-r2.json"]
    plan = json.loads(plan_bytes)
    summary = json.loads((root / "reference.json").read_text())
    receipt = json.loads(raw["microflow-token-accounting-r8/receipt.json"])
    phase = json.loads(
        (BASE / "vocabulary-prefill-r8/prefill-shapes.json").read_text()
    )["request_phase"]
    events_bytes = raw["microflow-token-accounting-r8/token-accounting-148228.jsonl"]
    assert hashlib.sha256(events_bytes).hexdigest() == plan["source_events_sha256"]
    patterns = Counter()
    for line in events_bytes.splitlines():
        event = json.loads(line)
        if event["event"] != "forward_enter" or event["role"] != "target":
            continue
        if not phase["start_wall"] <= event["wall"] <= receipt["bench_ended_wall"]:
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
            patterns[(group["capacity"], group["requests"], lengths, flags)] += len(
                group["layers"]
            )
    assert len(patterns) == summary["cases_per_rank"] == 136
    assert sum(patterns.values()) == summary["observed_contexts"] * 30 == 4290
    assert {
        (
            c["capacity"],
            c["requests"],
            tuple(c["lengths"]),
            tuple(c["initial_flags"]),
        ): c["observed_contexts"] * 30
        for c in plan["cases"]
    } == patterns
    control = json.loads(members["gdn-reference-control-r2/receipt.json"])
    assert control["probe_exit_code"] == control["cleanup_exit_code"] == 0
    assert control["all_devices_released"] and not control["sampler_errors"]
    for name, digest in control["input_hashes"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    for rank in (0, 1):
        result = json.loads(members[f"gdn-reference-r2/rank-{rank}.json"])
        assert result["status"] == "qualified_component_timings"
        assert result["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
        assert len(result["cases"]) == 136
        for case, expected, derived in zip(
            result["cases"], plan["cases"], summary["ranks"][rank]["cases"]
        ):
            assert case["shape"] == expected
            assert all(
                c["finite"] and c["close"] and c["max_abs"] == 0
                for c in case["checks"].values()
            )
            assert len(case["timings"]) == 3
            values = [t["device_ms"] / t["replays"] for t in case["timings"]]
            assert all(
                t["replays"] == 30 and t["device_ms"] > 0 for t in case["timings"]
            )
            assert derived["replay_ms"] == values
            assert derived["median_ms"] == statistics.median(values)
    assert summary["status"] == "qualified_component_reference"
    assert "attainment" not in summary
