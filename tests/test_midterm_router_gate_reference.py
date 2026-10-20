"""Reconstruct router/gate schedules and verify actual custom-op call evidence."""

from collections import Counter
import hashlib
import json
import math
import statistics

from test_midterm_vocab_prefill import BASE, archive_members


def test_router_gate_source_schedule():
    members = archive_members(BASE / "router-gate-reference-r1")
    plan = json.loads(members["source/router-gate-reference-plan-r1.json"])
    raw = archive_members(BASE / "device-routing-r9")
    for rank in plan["ranks"]:
        data = raw["microflow-token-accounting-r9/" + rank["trace"]]
        assert hashlib.sha256(data).hexdigest() == rank["sha256"]
        counts = Counter()
        for line in data.splitlines():
            e = json.loads(line)
            if e["event"] != "forward_enter":
                continue
            if (
                not plan["request_phase"]["start_wall"]
                <= e["wall"]
                <= plan["request_phase"]["end_wall"]
            ):
                continue
            assert e["valid_extent"] and e["graph_mode"] == "FULL"
            if e["role"] == "target":
                counts["router", e["padded_tokens"]] += 40
                counts["fa_gate", e["padded_tokens"]] += 10
            else:
                assert len(e["draft_steps"]) == 2
                counts["router", e["padded_tokens"]] += 2
                counts["fa_gate", e["padded_tokens"]] += 2
        assert counts == {(c["kind"], c["rows"]): c["calls"] for c in rank["shapes"]}
    assert plan["ranks"][0]["shapes"] == plan["ranks"][1]["shapes"]
    for name, sha in plan["source_sha256"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == sha


def test_router_gate_actual_calls_and_timings():
    root = BASE / "router-gate-reference-r1"
    members = archive_members(root)
    plan_data = members["source/router-gate-reference-plan-r1.json"]
    plan = json.loads(plan_data)
    summary = json.loads((root / "summary.json").read_text())
    control_data = members["router-gate-reference-control-r1/receipt.json"]
    control = json.loads(control_data)
    assert hashlib.sha256(plan_data).hexdigest() == summary["plan_sha256"]
    assert hashlib.sha256(control_data).hexdigest() == summary["control_sha256"]
    assert control["probe_exit_code"] == control["cleanup_exit_code"] == 0
    assert control["all_devices_released"] and control["sampler_errors"] == []
    for name, sha in control["input_hashes"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == sha
    for rank in summary["ranks"]:
        data = members[f"router-gate-reference-r1/rank-{rank['rank']}.json"]
        assert hashlib.sha256(data).hexdigest() == rank["raw_sha256"]
        raw = json.loads(data)
        assert raw["status"] == "qualified_component_timings"
        assert raw["router_type"] == "AscendFusedTopKRouter"
        assert raw["router_import"].endswith(
            "/runtime/vllm_ascend/ops/fused_moe/router/fused_topk_router.py"
        )
        totals = {"router": 0.0, "fa_gate": 0.0}
        for case, row in zip(plan["cases"], raw["cases"], strict=True):
            assert row["shape"] == case
            c = row["checks"]
            assert c["finite"] and c["close"]
            if case["kind"] == "router":
                assert c["ids_exact"] and c["max_weight_sum_error"] <= 1e-5
                assert len(row["router_calls"]) == 4
                for call in row["router_calls"]:
                    assert call == {
                        "shape": [case["rows"], 256],
                        "dtype": "torch.float32",
                        "k": 8,
                        "k_group": 1,
                        "group_count": 1,
                        "group_select_mode": 1,
                        "renorm": 1,
                        "norm_type": 0,
                        "out_flag": False,
                        "routed_scaling_factor": 1.0,
                        "eps": 1e-20,
                        "bias_is_none": True,
                    }
            else:
                assert row["router_calls"] == [] and c["max_abs"] == 0
            values = []
            assert len(row["timings"]) == 3
            for i, t in enumerate(row["timings"]):
                assert t["repetition"] == i and t["replays"] == 100
                assert (
                    control["started"]
                    <= t["start_wall"]
                    < t["end_wall"]
                    <= control["ended"]
                )
                assert math.isfinite(t["device_ms"]) and t["device_ms"] > 0
                values.append(t["device_ms"] / 100)
            totals[case["kind"]] += statistics.median(values) * case["calls"]
        for kind, total in totals.items():
            assert math.isclose(total, rank["weighted_ms_by_kind"][kind], rel_tol=1e-12)
    assert summary["model_effective_flops"] is summary["attainment_percent"] is None


def test_router_gate_disjoint_semantics():
    members = archive_members(BASE / "router-gate-reference-r1")
    old_data = members["source/same-run-partial-accounting-r9-v8.json"]
    old = json.loads(old_data)
    new = json.loads(members["source/same-run-partial-accounting-r9-v9.json"])
    summary = json.loads(members["source/router-gate-reference-summary-r1.json"])
    assert hashlib.sha256(old_data).hexdigest() == new["previous_catalog_sha256"]
    assert old["request_phase"] == new["request_phase"]
    for name, value in old["components"].items():
        assert new["components"][name] == value
    namespace = {"__name__": "archived", "__file__": "audit_router_gate_reference.py"}
    exec(
        compile(
            members["source/audit_router_gate_reference.py"],
            "audit_router_gate_reference.py",
            "exec",
        ),
        namespace,
    )
    counts = namespace["router_counts"](40 * 339760 + 359141)
    assert counts == summary["router_scalar_accounting"]
    assert summary["fa_gate_multiply"] == (10 * 339760 + 359141) * 4096
    assert (
        summary["new_basic_subtotal"]
        == sum(counts["basic"].values()) + summary["fa_gate_multiply"]
    )
    assert (
        new["covered_basic_semantic_subtotal"]
        == old["covered_basic_semantic_subtotal"] + summary["new_basic_subtotal"]
    )
    assert new["covered_basic_semantic_subtotal"] == sum(new["components"].values())
