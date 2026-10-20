"""Validate preprocessing reference and reconstruct useful versus padded rows."""

from collections import Counter
import hashlib
import json
import math
import statistics
import subprocess
import sys

import pytest

from test_midterm_vocab_prefill import BASE, archive_members


@pytest.fixture(scope="module")
def evidence():
    root = BASE / "preprocess-reference-r1"
    members = archive_members(root)
    return (
        members,
        json.loads(members["source/preprocess-reference-plan-r1.json"]),
        json.loads((root / "reference.json").read_text()),
    )


def test_preprocessing_schedule_and_effective_rows(evidence):
    members, plan, summary = evidence
    raw = archive_members(BASE / "device-routing-r9")
    for name, digest in plan["source_sha256"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    for rank in plan["ranks"]:
        content = raw["microflow-token-accounting-r9/" + rank["trace"]]
        assert hashlib.sha256(content).hexdigest() == rank["trace_sha256"]
        shapes, counts = Counter(), Counter()
        for event in map(json.loads, content.splitlines()):
            if event["event"] != "forward_enter" or event["role"] != "target":
                continue
            if (
                not plan["request_phase"]["start_wall"]
                <= event["wall"]
                <= plan["request_phase"]["end_wall"]
            ):
                continue
            counts["target_contexts"] += 1
            for group in event["owned_gdn"]:
                factor = len(group["layers"])
                counts["valid_token_layers"] += group["cu"][-1] * factor
                cap = group["capacity"]
                if group["decode_only"]:
                    shapes[cap, cap, False, False] += factor
                    counts["source_token_layers"] += cap * factor
                else:
                    v = group["requests"] * group["width"]
                    shapes[cap, cap, True, True] += factor
                    shapes[cap, v, True, False] += factor
                    counts["source_token_layers"] += (cap + v) * factor
        assert counts == rank["counts"]
        assert shapes == {
            (s["source_rows"], s["output_rows"], s["mapped"], s["q_head_major"]): s[
                "calls"
            ]
            for s in plan["cases"]
        }
    assert summary["cases_per_rank"] == len(plan["cases"]) == 15
    assert summary["qk_useful_basic_arithmetic"] == sum(
        r["counts"]["valid_token_layers"] * 6144 for r in plan["ranks"]
    )
    assert summary["model_effective_flops"] is summary["attainment_percent"] is None


def test_preprocessing_cpu_checks_timings_and_release(evidence):
    members, plan, summary = evidence
    receipt_data = members["preprocess-reference-control-r1/receipt.json"]
    receipt = json.loads(receipt_data)
    assert hashlib.sha256(receipt_data).hexdigest() == summary["control_sha256"]
    assert receipt["probe_exit_code"] == receipt["cleanup_exit_code"] == 0
    assert receipt["all_devices_released"] and not receipt["sampler_errors"]
    for name, digest in receipt["input_hashes"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    for rank in summary["ranks"]:
        data = members[f"preprocess-reference-r1/rank-{rank['rank']}.json"]
        assert hashlib.sha256(data).hexdigest() == rank["raw_sha256"]
        raw = json.loads(data)
        assert raw["status"] == "qualified_component_timings"
        assert raw["plan_sha256"] == summary["plan_sha256"]
        weighted = 0
        for case, observed in zip(plan["cases"], raw["cases"], strict=True):
            assert case == observed["shape"]
            assert {c["name"] for c in observed["checks"]} == {
                "q",
                "k",
                "v",
                "g",
                "beta",
            }
            for check in observed["checks"]:
                assert check["finite"] and check["close"]
                assert math.isfinite(check["max_abs"]) and check["max_abs"] >= 0
                if check["name"] == "v":
                    assert check["max_abs"] == 0
                for a, b in zip(
                    check["actual_first16"], check["cpu_first16"], strict=True
                ):
                    assert abs(a - b) <= check["atol"] + check["rtol"] * abs(b)
            timings = observed["timings"]
            assert [t["repetition"] for t in timings] == [0, 1, 2]
            assert all(
                t["replays"] == 100
                and math.isfinite(t["device_ms"])
                and t["device_ms"] > 0
                for t in timings
            )
            weighted += (
                statistics.median(t["device_ms"] / 100 for t in timings) * case["calls"]
            )
        assert weighted == pytest.approx(rank["source_call_weighted_ms"])
        assert sum(c["sample_gate_gt20"] for c in raw["cases"]) > 0
        assert sum(c["sample_gate_le20"] for c in raw["cases"]) > 0


def test_preprocessing_archived_formula_boundaries(evidence, tmp_path):
    members, _, _ = evidence
    for name in ("audit_preprocess_reference.py", "test_preprocess_accounting.py"):
        (tmp_path / name).write_bytes(members["source/" + name])
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_preprocess_accounting.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
