"""Verify archived component samples without an NPU or mutable local evidence."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/r16-component-references"
)
SOURCE = "root/gennova-evidence/20261019-midterm-aligned"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("r16-components")
    index = json.loads((BASE / "archive.json").read_text())
    archive_path = root / "evidence.tar.gz"
    digest = hashlib.sha256()
    with archive_path.open("wb") as stream:
        for part in index["parts"]:
            data = (BASE / part["file"]).read_bytes()
            assert len(data) == part["bytes"]
            assert hashlib.sha256(data).hexdigest() == part["sha256"]
            digest.update(data)
            stream.write(data)
    assert digest.hexdigest() == index["archive_sha256"]
    assert archive_path.stat().st_size == index["archive_bytes"]
    expected = {m["name"]: m for m in index["members"]}
    seen = set()
    with tarfile.open(archive_path) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in seen
            seen.add(member.name)
            data = archive.extractfile(member).read()
            assert len(data) == expected[member.name]["bytes"]
            assert hashlib.sha256(data).hexdigest() == expected[member.name]["sha256"]
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    assert seen == set(expected)
    return root


def test_source_pins_and_failures(evidence):
    for path in BASE.glob("*.json"):
        if path.name == "archive.json":
            continue
        raw = evidence / SOURCE / path.name
        assert path.read_bytes() == raw.read_bytes()
        summary = json.loads(raw.read_text())
        for source, expected in summary["source_sha256"].items():
            assert (
                hashlib.sha256((evidence / source.lstrip("/")).read_bytes()).hexdigest()
                == expected
            )
    for run, expected in [
        ("fia-reference-control-r1", 1),
        ("fia-reference-control-r2", 0),
        ("fia-timing-control-r1", 0),
    ]:
        receipt = json.loads((evidence / SOURCE / run / "receipt.json").read_text())
        assert receipt["probe_exit_code"] == expected
        assert receipt["all_devices_released"] and not receipt["sampler_errors"]


def test_all_sampled_outputs_and_timings(evidence):
    root = evidence / SOURCE
    plan = json.loads((root / "fia-timing-plan-r1.json").read_text())
    summary = json.loads((root / "fia-timing-summary-r1.json").read_text())
    assert plan["atol"] == 0.005 and plan["rtol"] == 0.02
    for expected in summary["ranks"]:
        rank = expected["rank"]
        counts, weighted = Counter(), Counter()
        max_error = max_ratio = 0.0
        elements = 0
        with (root / f"fia-timing-r1/rank-{rank}.jsonl").open() as stream:
            for slot, line in enumerate(stream):
                row = json.loads(line)
                assert row["slot"] == slot
                check = row["check"]
                assert len(check["actual"]) == len(check["reference"]) == 8
                for actual, reference in zip(
                    check["actual"], check["reference"], strict=True
                ):
                    assert len(actual) == len(reference) == 256
                    for a, b in zip(actual, reference, strict=True):
                        assert math.isfinite(a) and math.isfinite(b)
                        error = abs(a - b)
                        assert error <= plan["atol"] + plan["rtol"] * abs(b)
                        max_error = max(max_error, error)
                        elements += 1
                assert len(row["timings"]) == plan["repetitions"] == 3
                times = []
                for rep, timing in enumerate(row["timings"]):
                    assert timing["repetition"] == rep and timing["replays"] == 10
                    assert (
                        math.isfinite(timing["device_ms"]) and timing["device_ms"] > 0
                    )
                    assert timing["end_wall"] > timing["start_wall"]
                    times.append(timing["device_ms"] / timing["replays"])
                max_ratio = max(max_ratio, max(times) / min(times))
                role = (
                    "target"
                    if row["role"] == "target"
                    else f"draft_step{row['draft_step']}"
                )
                counts[role] += 1
                counts[row["backend"]] += 1
                weighted[role] += statistics.median(times) * (
                    10 if role == "target" else 1
                )
        assert counts == expected["counts"]
        assert counts["target"] == counts["draft_step0"] == counts["draft_step1"] == 880
        assert elements == expected["sampled_output_elements"] == 5406720
        assert max_error == expected["max_abs_error"]
        assert max_ratio == expected["max_repetition_ratio"]
        assert weighted == expected["layer_weighted_component_ms"]
    assert summary["model_effective_flops"] is summary["m01_attainment"] is None


def test_window_cohorts_are_consistent(evidence):
    root = evidence / SOURCE
    attention = json.loads((root / "r16-request-attention.json").read_text())
    matrix = json.loads((root / "r16-matrix-accounting.json").read_text())
    gdn = json.loads((root / "r16-gdn-window.json").read_text())
    aligned = json.loads((root / "r16-attention-reference-aligned.json").read_text())
    assert (
        attention["request_phase"]
        == matrix["request_phase"]
        == gdn["request_phase"]
        == aligned["request_phase"]
    )
    for name, cohort in attention["cohorts"].items():
        assert {role: row["publications"] for role, row in cohort.items()} == aligned[
            "cohorts"
        ][name]["counts"]
        assert (
            sum(row["semantic_qk_av_arithmetic"] for row in cohort.values())
            == matrix["cohorts"][name]["causal_qk_av_semantic_subtotal"]
        )
        assert cohort["target"]["valid_query_tokens"] * 30 == (
            gdn["cohorts"][name]["prefill_token_layers"]
            + gdn["cohorts"][name]["verify_token_layers"]
        )
