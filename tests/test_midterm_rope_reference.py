"""Verify fused RMS/RoPE layouts, independent checks and retained failure."""

from collections import Counter
import hashlib
import json
import math
import statistics
import subprocess
import sys

from test_midterm_vocab_prefill import BASE, archive_members


def test_rotation_scalar_counts(tmp_path):
    members = archive_members(BASE / "rope-reference-r2")
    for name in ("audit_rope_reference.py", "test_rope_arithmetic.py"):
        (tmp_path / name).write_bytes(members["source/" + name])
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_rope_arithmetic.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_rope_schedule_matches_r9_capture_and_offsets():
    members = archive_members(BASE / "rope-reference-r2")
    plan = json.loads(members["source/rope-reference-plan-r2.json"])
    raw = archive_members(BASE / "device-routing-r9")
    phase = plan["request_phase"]
    for rank in plan["ranks"]:
        data = raw["microflow-token-accounting-r9/" + rank["trace"]]
        prefix_data = raw["microflow-token-accounting-r9/" + rank["prefix_trace"]]
        assert hashlib.sha256(data).hexdigest() == rank["trace_sha256"]
        assert hashlib.sha256(prefix_data).hexdigest() == rank["prefix_sha256"]
        events, prefix = (
            list(map(json.loads, data.splitlines())),
            list(map(json.loads, prefix_data.splitlines())),
        )
        captured = {
            (e["layer"], e["tokens"], e["split"])
            for e in prefix
            if e["event"] == "prefix"
            and e["capturing"]
            and e["time"] < phase["start_wall"]
        }
        inventory = next(e for e in events if e["event"] == "loaded_model_inventory")
        fa_layers = [
            m["name"].removesuffix(".self_attn.qkv_proj")
            for m in inventory["target"]["modules"]
            if m["name"].endswith(".self_attn.qkv_proj")
        ]
        assert len(fa_layers) == 10
        counts = Counter()
        for event in events:
            if (
                event["event"] != "forward_enter"
                or not phase["start_wall"] <= event["wall"] <= phase["end_wall"]
            ):
                continue
            assert event["graph_mode"] == "FULL" and event["valid_extent"]
            n = event["padded_tokens"]
            if event["role"] == "target":
                p = {1024: 2, 2048: 2, 4096: 4, 8192: 4}.get(n, 1)
                assert all((layer, n, p > 1) in captured for layer in fa_layers)
                for lo in range(0, n, n // p):
                    counts["partitioned" if p > 1 else "contiguous", n // p, n, lo] += (
                        10
                    )
            else:
                assert event["role"] == "draft" and len(event["draft_steps"]) == 2
                counts["contiguous", n, n, 0] += 2
        assert counts == {
            (s["kind"], s["rows"], s["cos_tokens"], s["offset"]): s["calls"]
            for s in rank["shapes"]
        }
    assert plan["ranks"][0]["shapes"] == plan["ranks"][1]["shapes"]
    assert plan["cases"] == [
        dict(case=i, **s) for i, s in enumerate(plan["ranks"][0]["shapes"])
    ]


def test_rope_reference_evidence_and_disjoint_arithmetic():
    root = BASE / "rope-reference-r2"
    members = archive_members(root)
    summary = json.loads((root / "summary.json").read_text())
    plan_data = members["source/rope-reference-plan-r2.json"]
    plan = json.loads(plan_data)
    assert hashlib.sha256(plan_data).hexdigest() == summary["plan_sha256"]
    control_data = members["rope-reference-control-r2/receipt.json"]
    control = json.loads(control_data)
    assert hashlib.sha256(control_data).hexdigest() == summary["control_sha256"]
    assert control["probe_exit_code"] == control["cleanup_exit_code"] == 0
    assert control["all_devices_released"] and not control["sampler_errors"]
    failed = json.loads(members["rope-reference-control-r1/receipt.json"])
    assert (
        failed["status"] == "failed"
        and failed["probe_exit_code"] == 1
        and failed["all_devices_released"]
    )
    assert (
        b"Device properties not initialized"
        in members["rope-reference-control-r1/probe.log"]
    )
    assert (
        hashlib.sha256(members["source/rope-reference-plan-r1.json"]).hexdigest()
        == plan["previous_plan_sha256"]
    )
    for name, digest in {**control["input_hashes"], **plan["source_sha256"]}.items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    for rank in summary["ranks"]:
        data = members[f"rope-reference-r2/rank-{rank['rank']}.json"]
        assert hashlib.sha256(data).hexdigest() == rank["raw_sha256"]
        raw = json.loads(data)
        assert raw["status"] == "qualified_component_timings"
        weighted = []
        for row, case, summarized in zip(
            raw["cases"], plan["cases"], rank["cases"], strict=True
        ):
            assert row["shape"] == case
            assert row["cos_stride"] == [64 * case["cos_tokens"], 64, 1]
            assert [c["name"] for c in row["checks"]] == ["q", "k", "v", "gate"]
            for check in row["checks"]:
                assert check["finite"] and check["close"]
                assert math.isfinite(check["max_abs"])
                if check["name"] in ("v", "gate"):
                    assert check["max_abs"] == 0
                for a, b in zip(
                    check["actual_first16"], check["cpu_first16"], strict=True
                ):
                    assert abs(a - b) <= check["atol"] + check["rtol"] * abs(b)
            values = []
            assert len(row["timings"]) == 3
            for i, timing in enumerate(row["timings"]):
                assert timing["repetition"] == i and timing["replays"] == 100
                assert (
                    control["started"]
                    <= timing["start_wall"]
                    < timing["end_wall"]
                    <= control["ended"]
                )
                assert math.isfinite(timing["device_ms"]) and timing["device_ms"] > 0
                values.append(timing["device_ms"] / 100)
            assert summarized["timings_ms"] == values and summarized[
                "median_ms"
            ] == statistics.median(values)
            weighted.append(case["calls"] * statistics.median(values))
        assert math.isclose(
            sum(weighted), rank["source_call_weighted_ms"], rel_tol=1e-12
        )
    counts_data = members["source/host-attention-pairs-r9.json"]
    assert hashlib.sha256(counts_data).hexdigest() == summary["counts_sha256"]
    counts = json.loads(counts_data)
    assert counts["request_phase"] == plan["request_phase"]
    roles = counts["ranks"][0]["roles"]
    tokens = 10 * roles["target"]["valid_query_tokens"] + sum(
        roles[f"draft_step{i}"]["valid_query_tokens"] for i in range(2)
    )
    assert summary["useful_rope_arithmetic"] == {
        "multiply": tokens * 18 * 64 * 2,
        "add_or_subtract": tokens * 18 * 64,
    }
    prior_data = members["source/same-run-partial-accounting-r9-v6.json"]
    prior = json.loads(prior_data)
    combined = json.loads((root / "same-run-catalog.json").read_text())
    assert hashlib.sha256(prior_data).hexdigest() == combined["previous_catalog_sha256"]
    assert combined["request_phase"] == plan["request_phase"]
    for key, value in prior["components"].items():
        assert combined["components"][key] == value
    assert combined["components"]["partial_rope_basic_ops"] == tokens * 18 * 64 * 3
    assert (
        sum(combined["components"].values())
        == combined["covered_basic_semantic_subtotal"]
    )
    assert combined["model_effective_flops"] is combined["attainment_percent"] is None
