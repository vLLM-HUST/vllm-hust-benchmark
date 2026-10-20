"""Check deployed norm reference shapes, numeric evidence and resource release."""

from collections import Counter
import hashlib
import json
import math
import statistics

from test_midterm_vocab_prefill import BASE, archive_members


def test_norm_reference_schedule_and_sources():
    members = archive_members(BASE / "norm-reference-r1")
    plan = json.loads(members["source/norm-reference-plan-r1.json"])
    raw = archive_members(BASE / "device-routing-r9")
    phase = plan["request_phase"]
    for rank in plan["ranks"]:
        data = raw["microflow-token-accounting-r9/" + rank["trace"]]
        assert hashlib.sha256(data).hexdigest() == rank["trace_sha256"]
        counts = Counter()
        contexts = Counter()
        for event in map(json.loads, data.splitlines()):
            if event["event"] != "forward_enter":
                continue
            if not phase["start_wall"] <= event["wall"] <= phase["end_wall"]:
                continue
            assert event["graph_mode"] == "FULL" and event["valid_extent"]
            rows = event["padded_tokens"]
            contexts[event["role"]] += 1
            if event["role"] == "target":
                counts["hidden_plain", rows, 2048] += 1
                counts["hidden_residual", rows, 2048] += 80
                counts["gdn_output", rows * 16, 128] += 30
            else:
                assert event["role"] == "draft" and len(event["draft_steps"]) == 2
                counts["hidden_plain", rows, 2048] += 6
                counts["hidden_residual", rows, 2048] += 4
        assert contexts == {"target": 884, "draft": 884}
        assert counts == {
            (c["kind"], c["rows"], c["width"]): c["calls"] for c in rank["shapes"]
        }
    assert plan["ranks"][0]["shapes"] == plan["ranks"][1]["shapes"]
    assert plan["cases"] == [
        dict(case=i, **shape) for i, shape in enumerate(plan["ranks"][0]["shapes"])
    ]
    for name, digest in plan["source_sha256"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest


def test_norm_reference_measurements_and_cleanup():
    root = BASE / "norm-reference-r1"
    members = archive_members(root)
    summary = json.loads((root / "summary.json").read_text())
    plan_data = members["source/norm-reference-plan-r1.json"]
    control_data = members["norm-reference-control-r1/receipt.json"]
    plan, control = json.loads(plan_data), json.loads(control_data)
    assert hashlib.sha256(plan_data).hexdigest() == summary["plan_sha256"]
    assert hashlib.sha256(control_data).hexdigest() == summary["control_sha256"]
    assert control["probe_exit_code"] == control["cleanup_exit_code"] == 0
    assert control["all_devices_released"] and not control["sampler_errors"]
    assert control["reserved_devices"] == [4, 5, 6, 7]
    for name, digest in control["input_hashes"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    for rank in summary["ranks"]:
        data = members[f"norm-reference-r1/rank-{rank['rank']}.json"]
        assert hashlib.sha256(data).hexdigest() == rank["raw_sha256"]
        raw = json.loads(data)
        assert (
            raw["status"] == "qualified_component_timings" and raw["custom_ops_enabled"]
        )
        assert raw["norm_import_path"].endswith("/runtime/vllm_ascend/ops/layernorm.py")
        assert raw["plan_sha256"] == summary["plan_sha256"]
        weighted = 0
        for record, case, summarized in zip(
            raw["cases"], plan["cases"], rank["cases"], strict=True
        ):
            assert record["shape"] == case and record["case"] == case["case"]
            assert len(record["checks"]) == (
                2 if case["kind"] == "hidden_residual" else 1
            )
            for check in record["checks"]:
                assert check["finite"] and check["close"]
                assert math.isfinite(check["max_abs"]) and check["max_abs"] >= 0
                for actual, expected in zip(
                    check["actual_first16"], check["cpu_first16"], strict=True
                ):
                    assert abs(actual - expected) <= check["atol"] + check[
                        "rtol"
                    ] * abs(expected)
            assert len(record["timings"]) == plan["repetitions"] == 3
            values = []
            for i, timing in enumerate(record["timings"]):
                assert timing["repetition"] == i and timing["replays"] == 100
                assert (
                    control["started"]
                    <= timing["start_wall"]
                    < timing["end_wall"]
                    <= control["ended"]
                )
                assert math.isfinite(timing["device_ms"]) and timing["device_ms"] > 0
                values.append(timing["device_ms"] / 100)
            assert summarized["timings_ms"] == values
            assert summarized["median_ms"] == statistics.median(values)
            weighted += statistics.median(values) * case["calls"]
        assert math.isclose(weighted, rank["source_call_weighted_ms"], rel_tol=1e-12)
    assert summary["model_effective_flops"] is summary["attainment_percent"] is None
