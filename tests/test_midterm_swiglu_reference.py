"""Reconstruct SwiGLU shapes from raw routing and check measured evidence."""

from collections import Counter
import hashlib
import json
import math
import statistics

from test_midterm_vocab_prefill import BASE, archive_members


def test_swiglu_raw_schedule():
    members = archive_members(BASE / "swiglu-reference-r1")
    plan = json.loads(members["source/swiglu-reference-plan-r1.json"])
    original = archive_members(BASE / "device-routing-r9")
    for rank in plan["ranks"]:
        data = original["microflow-token-accounting-r9/" + rank["trace"]]
        assert hashlib.sha256(data).hexdigest() == rank["sha256"]
        counts, useful = Counter(), 0
        for line in data.splitlines():
            event = json.loads(line)
            if (
                not plan["request_phase"]["start_wall"]
                <= event.get("wall", -1)
                <= plan["request_phase"]["end_wall"]
            ):
                continue
            if event["event"] == "moe_device_counts":
                for layer in event["layers"]:
                    for groups in layer["group_counts"]:
                        counts["routed", sum(groups)] += 1
            elif event["event"] == "forward_enter":
                assert event["moe_context"]["type"] == "MoECommType.ALLGATHER"
                layers = 40 if event["role"] == "target" else 2
                counts["shared", event["padded_tokens"]] += layers
                useful += (
                    40 * event["actual_tokens"]
                    if event["role"] == "target"
                    else sum(s["actual_tokens"] for s in event["draft_steps"])
                )
        assert useful == rank["useful_token_layers"] == 13949541
        assert counts == {(c["kind"], c["rows"]): c["calls"] for c in rank["cases"]}
    assert len(plan["cases"]) == 28
    assert sum(c["calls"] for c in plan["cases"]) == 74256
    for name, digest in plan["source_sha256"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest


def test_swiglu_measurements():
    root = BASE / "swiglu-reference-r1"
    members = archive_members(root)
    summary = json.loads((root / "summary.json").read_text())
    plan_data = members["source/swiglu-reference-plan-r1.json"]
    plan = json.loads(plan_data)
    receipt = json.loads(members["swiglu-reference-control-r1/receipt.json"])
    assert receipt["probe_exit_code"] == receipt["cleanup_exit_code"] == 0
    assert receipt["all_devices_released"] and receipt["sampler_errors"] == []
    assert receipt["reserved_devices"] == [4, 5, 6, 7]
    for name, digest in receipt["input_hashes"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    assert hashlib.sha256(plan_data).hexdigest() == summary["plan_sha256"]
    for rank in summary["ranks"]:
        data = members[f"swiglu-reference-r1/rank-{rank['rank']}.json"]
        assert hashlib.sha256(data).hexdigest() == rank["raw_sha256"]
        raw = json.loads(data)
        assert raw["status"] == "qualified_component_timings"
        assert raw["activation_import_path"].endswith(
            "/runtime/vllm_ascend/ops/activation.py"
        )
        totals = {"routed": 0.0, "shared": 0.0}
        for case, row in zip(plan["cases"], raw["cases"], strict=True):
            assert row["shape"] == case
            assert row["checks"]["finite"] and row["checks"]["close"]
            assert row["checks"]["max_abs"] == 0
            samples = []
            assert len(row["timings"]) == 3
            for i, sample in enumerate(row["timings"]):
                assert sample["repetition"] == i and sample["replays"] == 100
                assert (
                    receipt["started"]
                    <= sample["start_wall"]
                    < sample["end_wall"]
                    <= receipt["ended"]
                )
                assert math.isfinite(sample["device_ms"]) and sample["device_ms"] > 0
                samples.append(sample["device_ms"] / 100)
            totals[case["kind"]] += statistics.median(samples) * case["calls"]
        for kind, total in totals.items():
            assert math.isclose(
                total, rank["source_call_weighted_ms"][kind], rel_tol=1e-12
            )
    assert summary["admitted_to_whole_model_catalog"] is False
    assert summary["model_effective_flops"] is summary["m01_attainment"] is None
    namespace = {"__name__": "archived_audit", "__file__": "audit_swiglu_reference.py"}
    exec(
        compile(
            members["source/audit_swiglu_reference.py"],
            "audit_swiglu_reference.py",
            "exec",
        ),
        namespace,
    )
    assert namespace["swiglu_counts"](13949541) == summary["candidate_useful_counts"]
