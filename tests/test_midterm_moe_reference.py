"""Recompute exact group-list coverage and per-case medians from raw evidence."""

from collections import Counter, defaultdict
import hashlib
import json
import math
import statistics

import pytest

from test_midterm_vocab_prefill import BASE, archive_members


@pytest.fixture(scope="module")
def evidence():
    root = BASE / "moe-reference-r2"
    members = archive_members(root)
    plan = json.loads(members["source/moe-reference-plan-r2.json"])
    summary = json.loads((root / "reference.json").read_text())
    return members, plan, summary


def test_all_device_distributions_in_plan(evidence):
    members, plan, summary = evidence
    source = archive_members(BASE / "device-routing-r9")
    expected = {tuple(c["group_counts"]): c["observed_calls"] for c in plan["cases"]}
    assert (
        len(expected) == len(plan["cases"]) == summary["unique_distributions"] == 37128
    )
    assert {c["case"] for c in plan["cases"]} == set(range(37128))
    for case in plan["cases"]:
        assert case["rows"] == sum(case["group_counts"])
        assert len(case["group_counts"]) == 256
        assert all(type(n) is int and n >= 0 for n in case["group_counts"])
    for name, digest in plan["source_events_sha256"].items():
        raw = source["microflow-token-accounting-r9/" + name]
        assert hashlib.sha256(raw).hexdigest() == digest
        counts = Counter()
        for line in raw.splitlines():
            event = json.loads(line)
            if event["event"] == "moe_device_counts" and (
                plan["request_phase"]["start_wall"]
                <= event["wall"]
                <= plan["request_phase"]["end_wall"]
            ):
                counts.update(
                    tuple(row)
                    for layer in event["layers"]
                    for row in layer["group_counts"]
                )
        assert counts == expected
    assert summary["model_effective_flops"] is None
    assert summary["attainment_percent"] is None


def test_cleanup_dynamic_group_checks_and_sample_errors(evidence):
    members, plan, summary = evidence
    for version in (1, 2):
        control_bytes = members[f"moe-reference-control-r{version}/receipt.json"]
        control = json.loads(control_bytes)
        assert control["probe_exit_code"] == control["cleanup_exit_code"] == 0
        assert control["all_devices_released"] and not control["sampler_errors"]
        for name, sha in control["input_hashes"].items():
            assert hashlib.sha256(members["source/" + name]).hexdigest() == sha
    buckets = defaultdict(list)
    for case in plan["cases"]:
        buckets[case["rows"]].append(case["case"])
    assert len(buckets) == summary["row_buckets"] == 14
    for rank in (0, 1):
        raw_bytes = members[f"moe-reference-r2/rank-{rank}.json"]
        raw = json.loads(raw_bytes)
        assert (
            hashlib.sha256(raw_bytes).hexdigest()
            == summary["ranks"][rank]["result_sha256"]
        )
        assert (
            raw["plan_sha256"]
            == hashlib.sha256(members["source/moe-reference-plan-r2.json"]).hexdigest()
        )
        assert raw["status"] == "qualified_component_timings"
        assert len(raw["buckets"]) == 28
        identities = set()
        maximum = 0.0
        for bucket in raw["buckets"]:
            op = bucket["operator"]
            assert op in plan["operators"]
            identity = op["name"], bucket["rows"]
            assert identity not in identities
            identities.add(identity)
            ids = buckets[bucket["rows"]]
            assert bucket["timed_cases"] == 3 * len(ids)
            assert bucket["weight_format"] == 2
            assert bucket["weight_stride"] == [op["k"] * op["n"], op["n"], 1]
            assert [c["case"] for c in bucket["checks"]] == sorted({ids[0], ids[-1]})
            for check in bucket["checks"]:
                assert check["samples"]
                for sample in check["samples"]:
                    assert sample["finite"] and sample["close"]
                    differences = []
                    for actual_row, expected_row in zip(
                        sample["actual"], sample["cpu_fp32"]
                    ):
                        for actual, expected in zip(actual_row, expected_row):
                            assert math.isfinite(actual) and math.isfinite(expected)
                            difference = abs(actual - expected)
                            assert difference <= plan["atol"] + plan["rtol"] * abs(
                                expected
                            )
                            differences.append(difference)
                    assert max(differences) == pytest.approx(
                        sample["max_abs"], abs=1e-7
                    )
                    maximum = max(maximum, max(differences))
        assert summary["ranks"][rank]["max_sample_abs_error"] == maximum


def test_all_three_repetitions_recompute_reference(evidence):
    members, plan, summary = evidence
    assert plan["repetitions"] == 3 and plan["timed_replays"] == 1
    for rank in (0, 1):
        raw = members[f"moe-reference-r2/timings-{rank}.jsonl"]
        assert (
            hashlib.sha256(raw).hexdigest() == summary["ranks"][rank]["timings_sha256"]
        )
        times = defaultdict(dict)
        for line in raw.splitlines():
            row = json.loads(line)
            identity = row["operator"], row["case"]
            assert row["repetition"] not in times[identity]
            assert row["replays"] == 1 and row["device_ms"] > 0
            assert row["host_seconds"] > 0 and row["end_wall"] > row["start_wall"]
            times[identity][row["repetition"]] = row["device_ms"]
        assert len(times) == 37128 * 2
        for op, result in zip(plan["operators"], summary["ranks"][rank]["operators"]):
            weighted = [0.0, 0.0, 0.0]
            medians = 0.0
            flops = 0
            for case in plan["cases"]:
                values = times[op["name"], case["case"]]
                assert set(values) == {0, 1, 2}
                for repetition in range(3):
                    weighted[repetition] += values[repetition] * case["observed_calls"]
                medians += statistics.median(values.values()) * case["observed_calls"]
                flops += 2 * case["rows"] * op["k"] * op["n"] * case["observed_calls"]
            assert result["repetition_weighted_device_ms"] == weighted
            assert result["sum_case_median_weighted_ms"] == medians
            assert result["device_matrix_flops_including_padding"] == flops
            assert result["component_tflops_from_case_medians"] == flops / (
                medians * 1e9
            )
