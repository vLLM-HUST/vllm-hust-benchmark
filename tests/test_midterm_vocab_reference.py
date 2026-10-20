"""Independently check measured reference shapes, samples and all repetitions."""

from collections import Counter
import hashlib
import json
import math
import statistics

from test_midterm_vocab_prefill import BASE, archive_members


def test_vocabulary_reference_covers_raw_workload_shapes():
    root = BASE / "vocab-reference-r1"
    members = archive_members(root)
    raw = archive_members(BASE / "recurrence-r8")
    plan_bytes = members["source/vocab-reference-plan-r1.json"]
    plan = json.loads(plan_bytes)
    summary = json.loads((root / "reference.json").read_text())
    phase = json.loads((BASE / "vocabulary-prefill-r8/vocabulary.json").read_text())[
        "request_phase"
    ]
    source_receipt = json.loads(raw["microflow-token-accounting-r8/receipt.json"])
    expected = Counter()
    for line in raw[
        "microflow-token-accounting-r8/token-accounting-148228.jsonl"
    ].splitlines():
        event = json.loads(line)
        if (
            not phase["start_wall"]
            <= event["wall"]
            <= source_receipt["bench_ended_wall"]
        ):
            continue
        if event["event"] == "target_logits_input":
            expected[("target", event["rows"])] += 1
        elif event["event"] == "draft_replay_input":
            expected[("draft", event["first_step_sample_indices_rows"])] += 1
        elif event["event"] == "forward_enter" and event["role"] == "draft":
            expected[("draft", event["draft_steps"][1]["actual_tokens"])] += 1
    assert len(plan["cases"]) == summary["shapes_per_rank"] == 45
    assert set(c["m"] for c in plan["cases"]) == {m for _, m in expected}
    for case in plan["cases"]:
        assert (case["n"], case["k"]) == (124160, 2048)
        for role in ("target", "draft"):
            assert case[role + "_calls"] == expected[(role, case["m"])]
    for name, digest in plan["source_hashes"].items():
        assert (
            hashlib.sha256(raw["microflow-token-accounting-r8/" + name]).hexdigest()
            == digest
        )
    receipt = json.loads(members["vocab-reference-control-r1/receipt.json"])
    assert receipt["probe_exit_code"] == receipt["cleanup_exit_code"] == 0
    assert receipt["all_devices_released"] and not receipt["sampler_errors"]
    for name, digest in receipt["input_hashes"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    for rank in (0, 1):
        result = json.loads(members[f"vocab-reference-r1/rank-{rank}.json"])
        assert result["status"] == "qualified_component_timings"
        assert result["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
        assert len(result["cases"]) == len(summary["ranks"][rank]["cases"]) == 45
        for case, frozen, derived in zip(
            result["cases"], plan["cases"], summary["ranks"][rank]["cases"]
        ):
            assert case["shape"] == frozen == derived["shape"]
            check = case["check"]
            assert check["finite"] and check["close"]
            differences = []
            for actual, oracle in zip(check["actual"], check["cpu_fp32"], strict=True):
                assert len(actual) == len(oracle) == 8
                for x, y in zip(actual, oracle):
                    assert math.isfinite(x) and math.isfinite(y)
                    differences.append(abs(x - y))
                    assert abs(x - y) <= plan["atol"] + plan["rtol"] * abs(y)
            assert max(differences) == check["max_abs"]
            assert len(case["timings"]) == 3
            for index, timing in enumerate(case["timings"]):
                assert timing["replays"] == 10 and timing["repetition"] == index
                assert timing["device_ms"] > 0
            values = [t["device_ms"] / 10 for t in case["timings"]]
            assert derived["replay_ms"] == values
            assert derived["median_ms"] == statistics.median(values)
            assert derived["local_tflops"] == 2 * frozen["m"] * frozen["n"] * frozen[
                "k"
            ] / (derived["median_ms"] * 1e9)
        assert math.isclose(
            summary["ranks"][rank]["call_weighted_reference_seconds"],
            sum(
                c["median_ms"]
                * (c["shape"]["target_calls"] + c["shape"]["draft_calls"])
                / 1000
                for c in summary["ranks"][rank]["cases"]
            ),
            rel_tol=1e-12,
        )
    assert summary["model_compute_capacity"] is None
    assert summary["attainment_percent"] is None
