"""Independently reconstruct FULL graph matrix shapes and all reference sums."""

from collections import Counter, defaultdict
import hashlib
import json
import math
import statistics

import pytest

from test_midterm_vocab_prefill import BASE, archive_members


@pytest.fixture(scope="module")
def evidence():
    root = BASE / "dense-reference-r2"
    members = archive_members(root)
    plan = json.loads(members["source/dense-reference-plan-r2.json"])
    summary = json.loads((root / "reference.json").read_text())
    return members, plan, summary


def test_dense_source_shape_coverage(evidence):
    members, plan, summary = evidence
    raw = archive_members(BASE / "device-routing-r9")
    expected = {(c["m"], c["n"], c["k"]): c["observed_calls"] for c in plan["cases"]}
    assert len(expected) == len(plan["cases"]) == summary["cases_per_rank"] == 121
    assert sum(expected.values()) == summary["source_shaped_calls_per_rank"] == 253046
    for name, digest in plan["source_sha256"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    for name, digest in plan["source_events_sha256"].items():
        assert (
            hashlib.sha256(raw["microflow-token-accounting-r9/" + name]).hexdigest()
            == digest
        )
        if not name.startswith("token-accounting-"):
            continue
        pid = name.removeprefix("token-accounting-").removesuffix(".jsonl")
        events = list(
            map(json.loads, raw["microflow-token-accounting-r9/" + name].splitlines())
        )
        prefix = list(
            map(
                json.loads,
                raw[
                    f"microflow-token-accounting-r9/microflow-{pid}.jsonl"
                ].splitlines(),
            )
        )
        capture = {
            (e["layer"], e["tokens"], e["split"])
            for e in prefix
            if e["event"] == "prefix"
            and e["capturing"]
            and e["time"] < plan["request_phase"]["start_wall"]
        }
        inventory = next(e for e in events if e["event"] == "loaded_model_inventory")
        modules = {
            role: [
                m
                for m in inventory[role]["modules"]
                if m["type"].startswith("vllm_ascend.ops.linear.")
                and len(m["parameters"].get("weight", {}).get("shape", [])) == 2
            ]
            for role in ("target", "draft")
        }
        assert [len(modules[r]) for r in ("target", "draft")] == [270, 7]
        counts = Counter()
        catalog = Counter()
        for event in events:
            if event["event"] != "forward_enter" or not (
                plan["request_phase"]["start_wall"]
                <= event["wall"]
                <= plan["request_phase"]["end_wall"]
            ):
                continue
            assert event["graph_mode"] == "FULL" and event["valid_extent"]
            role, rows = event["role"], event["padded_tokens"]
            steps = len(event["draft_steps"]) if role == "draft" else 1
            assert steps == (2 if role == "draft" else 1)
            for module in modules[role]:
                n, k = module["parameters"]["weight"]["shape"]
                pieces = 1
                if role == "target" and module["name"].endswith(".self_attn.qkv_proj"):
                    pieces = {1024: 2, 2048: 2, 4096: 4, 8192: 4}.get(rows, 1)
                    assert (
                        module["name"].removesuffix(".self_attn.qkv_proj"),
                        rows,
                        pieces > 1,
                    ) in capture
                assert rows % pieces == 0
                counts[rows // pieces, n, k] += pieces * steps
                catalog[role, module["name"], rows // pieces, n, k] += pieces * steps
        assert counts == expected
        assert catalog == {
            (c["role"], c["module"], c["m"], c["n"], c["k"]): c["calls"]
            for c in plan["call_catalog"]
        }
    assert (
        summary["model_effective_flops"] is None
        and summary["attainment_percent"] is None
    )


def test_dense_numerics_timings_and_release(evidence):
    members, plan, summary = evidence
    control = json.loads(members["dense-reference-control-r2/receipt.json"])
    assert control["probe_exit_code"] == control["cleanup_exit_code"] == 0
    assert control["all_devices_released"] and not control["sampler_errors"]
    for name, digest in control["input_hashes"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    for rank in (0, 1):
        raw_bytes = members[f"dense-reference-r2/rank-{rank}.json"]
        raw = json.loads(raw_bytes)
        derived = summary["ranks"][rank]
        assert hashlib.sha256(raw_bytes).hexdigest() == derived["result_sha256"]
        assert raw["status"] == "qualified_component_timings"
        assert (
            raw["plan_sha256"]
            == hashlib.sha256(
                members["source/dense-reference-plan-r2.json"]
            ).hexdigest()
        )
        assert len(raw["cases"]) == len(plan["cases"])
        groups = defaultdict(lambda: [0.0, [0.0] * 5, 0])
        maximum = 0.0
        for result, case in zip(raw["cases"], plan["cases"]):
            assert result["shape"] == case
            assert (
                result["weight_stride"] == [case["k"], 1]
                and result["weight_format"] == 2
            )
            check = result["check"]
            assert check["finite"] and check["close"]
            assert all(0 <= i < case["m"] for i in check["rows"])
            assert all(0 <= i < case["n"] for i in check["columns"])
            differences = []
            for arow, erow in zip(check["actual"], check["cpu_fp32"]):
                for a, e in zip(arow, erow):
                    assert math.isfinite(a) and math.isfinite(e)
                    assert abs(a - e) <= plan["atol"] + plan["rtol"] * abs(e)
                    differences.append(abs(a - e))
            assert max(differences) == pytest.approx(check["max_abs"], abs=1e-7)
            maximum = max(maximum, max(differences))
            assert len(result["timings"]) == plan["repetitions"] == 5
            values = []
            group = groups[case["n"], case["k"]]
            for i, timing in enumerate(result["timings"]):
                assert (
                    timing["repetition"] == i
                    and timing["replays"] == plan["timed_replays"] == 100
                )
                assert timing["device_ms"] > 0 and timing["host_seconds"] > 0
                assert (
                    control["started"]
                    <= timing["start_wall"]
                    < timing["end_wall"]
                    <= control["ended"]
                )
                values.append(timing["device_ms"] / 100)
                group[1][i] += values[-1] * case["observed_calls"]
            group[0] += statistics.median(values) * case["observed_calls"]
            group[2] += 2 * case["m"] * case["n"] * case["k"] * case["observed_calls"]
        assert derived["max_sample_abs_error"] == maximum
        assert len(groups) == len(derived["weights"]) == 9
        for weight in derived["weights"]:
            actual = groups[weight["n"], weight["k"]]
            assert weight["weighted_ms"] == actual[0]
            assert weight["repetitions_weighted_ms"] == actual[1]
            assert weight["device_matrix_flops_including_padding"] == actual[2]
            assert weight["component_tflops"] == actual[2] / (actual[0] * 1e9)
        assert derived["sum_case_median_weighted_ms"] == sum(
            g[0] for g in groups.values()
        )


def test_original_dispersion_is_preserved(evidence):
    members, plan, _ = evidence
    original_bytes = members["source/dense-reference-plan-r1.json"]
    original = json.loads(original_bytes)
    assert plan["previous_plan_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert plan["cases"] == original["cases"]
    assert original["repetitions"] == 3 and original["timed_replays"] == 10
    assert plan["graph_warmups_per_repetition"] == 20
    control = json.loads(members["dense-reference-control-r1/receipt.json"])
    assert control["probe_exit_code"] == control["cleanup_exit_code"] == 0
    assert control["all_devices_released"]
    for name, sha in control["input_hashes"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == sha
    ratios = []
    for rank in (0, 1):
        raw = json.loads(members[f"dense-reference-r1/rank-{rank}.json"])
        assert raw["plan_sha256"] == hashlib.sha256(original_bytes).hexdigest()
        assert len(raw["cases"]) == 121
        for case in raw["cases"]:
            assert len(case["timings"]) == 3
            values = [t["device_ms"] for t in case["timings"]]
            ratios.append(max(values) / min(values))
    assert max(ratios) > 100
