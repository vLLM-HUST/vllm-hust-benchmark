"""Verify frozen combine evidence, source metadata and disjoint accounting."""

from collections import Counter
import hashlib
import json
import math
import statistics

from test_midterm_vocab_prefill import BASE, archive_members


def test_combine_plan_and_source_metadata():
    members = archive_members(BASE / "moe-combine-reference-r2")
    plan = json.loads(members["source/moe-combine-reference-plan-r2.json"])
    old_plan = members["source/moe-combine-reference-plan-r1.json"]
    assert hashlib.sha256(old_plan).hexdigest() == plan["previous_attempt_plan_sha256"]
    assert json.loads(old_plan)["cases"] == plan["cases"]
    source = json.loads(members["source/moe-scalar-source-audit-r1.json"])
    for name, record in source["sources"].items():
        assert (
            hashlib.sha256(members["source/model-sources/" + name]).hexdigest()
            == record["sha256"]
        )
    assert source["factory_defaults"]["activation"] == "silu"
    assert source["factory_defaults"]["apply_router_weight_on_input"] is False
    for name, digest in plan["source_sha256"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    swiglu = archive_members(BASE / "swiglu-reference-r1")
    raw_plan = swiglu["source/swiglu-reference-plan-r1.json"]
    assert hashlib.sha256(raw_plan).hexdigest() == plan["previous_plan_sha256"]
    shared = [c for c in json.loads(raw_plan)["cases"] if c["kind"] == "shared"]
    assert plan["cases"] == [
        dict(case=i, rows=c["rows"], width=2048, calls=c["calls"])
        for i, c in enumerate(shared)
    ]


def test_combine_measurements_and_failure_preserved():
    root = BASE / "moe-combine-reference-r2"
    members = archive_members(root)
    plan = json.loads(members["source/moe-combine-reference-plan-r2.json"])
    summary = json.loads((root / "summary.json").read_text())
    failed = json.loads(members["moe-combine-reference-control-r1/receipt.json"])
    assert failed["status"] == "failed" and failed["probe_exit_code"] == 1
    assert failed["all_devices_released"]
    assert (
        b"partially initialized module"
        in members["moe-combine-reference-control-r1/probe.log"]
    )
    control = json.loads(members["moe-combine-reference-control-r2/receipt.json"])
    assert control["probe_exit_code"] == control["cleanup_exit_code"] == 0
    assert control["all_devices_released"] and not control["sampler_errors"]
    for attempt in (failed, control):
        for name, digest in attempt["input_hashes"].items():
            assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    for rank in summary["ranks"]:
        data = members[f"moe-combine-reference-r2/rank-{rank['rank']}.json"]
        assert hashlib.sha256(data).hexdigest() == rank["raw_sha256"]
        raw = json.loads(data)
        assert raw["status"] == "qualified_component_timings"
        assert raw["operator_import_path"].endswith(
            "/runtime/vllm_ascend/device/device_op.py"
        )
        total = 0
        for case, row in zip(plan["cases"], raw["cases"], strict=True):
            assert row["shape"] == case
            groups = Counter(
                (token * 17 + route * 31) % 256
                for token in range(case["rows"])
                for route in range(8)
            )
            assert row["group_counts"] == [groups[i] for i in range(256)]
            assert len(row["checks"]) == 2
            assert all(
                c["finite"] and c["close"] and c["max_abs"] == 0 for c in row["checks"]
            )
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
            total += statistics.median(values) * case["calls"]
        assert math.isclose(total, rank["weighted_ms"], rel_tol=1e-12)
    assert summary["model_effective_flops"] is summary["attainment_percent"] is None


def test_combine_disjoint_catalog_extension():
    members = archive_members(BASE / "moe-combine-reference-r2")
    prior_data = members["source/same-run-partial-accounting-r9-v7.json"]
    prior = json.loads(prior_data)
    new = json.loads(members["source/same-run-partial-accounting-r9-v8.json"])
    assert hashlib.sha256(prior_data).hexdigest() == new["previous_catalog_sha256"]
    assert prior["request_phase"] == new["request_phase"]
    for name, value in prior["components"].items():
        assert new["components"][name] == value
    namespace = {
        "__name__": "archived_source_audit",
        "__file__": "audit_moe_scalar_sources.py",
    }
    exec(
        compile(
            members["source/audit_moe_scalar_sources.py"],
            "audit_moe_scalar_sources.py",
            "exec",
        ),
        namespace,
    )
    ops = namespace["scalar_counts"](40 * 339760 + 359141)
    assert sum(ops.values()) == 549946704384
    assert new["components"][
        "moe_activation_gate_and_weighted_combine_basic_ops"
    ] == sum(ops.values())
    assert new["covered_basic_semantic_subtotal"] == prior[
        "covered_basic_semantic_subtotal"
    ] + sum(ops.values())
    assert new["covered_basic_semantic_subtotal"] == sum(new["components"].values())
