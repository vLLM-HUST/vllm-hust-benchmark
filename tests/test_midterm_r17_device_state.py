"""Recompute device eligibility and hardware coverage from archived r17 bytes."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/r17-device-state"
)
SOURCE = "root/gennova-evidence/20261019-midterm-aligned"
RUN = "workspace/gennova-measurements/microflow-token-accounting-r17"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("r17")
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


def test_source_identity_and_lifecycle(evidence):
    for path in BASE.glob("*.json"):
        if path.name == "archive.json":
            continue
        assert path.read_bytes() == (evidence / SOURCE / path.name).read_bytes()
        for source, digest in json.loads(path.read_text())["source_sha256"].items():
            assert (
                hashlib.sha256((evidence / source.lstrip("/")).read_bytes()).hexdigest()
                == digest
            )
    receipt = json.loads((evidence / RUN / "receipt.json").read_text())
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    assert receipt["bench_exit_code"] == 0
    for name, digest in receipt["input_hashes"].items():
        path = (
            evidence / name.lstrip("/")
            if name.startswith("/")
            else evidence / SOURCE / name
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_device_state_controls_ledger(evidence, monkeypatch):
    monkeypatch.syspath_prepend(str(evidence / SOURCE))
    from audit_r17_gdn_publications import join, count
    from derive_r17_device_gdn import operations
    from derive_r17_gdn_prefill import chunks, ho, wy, inverse

    summary = json.loads((BASE / "r17-gdn-publication-summary.json").read_text())
    ledger = json.loads((BASE / "r17-device-gdn-ledger.json").read_text())
    prefill = json.loads((BASE / "r17-device-gdn-prefill.json").read_text())
    for expected, expected_ledger in zip(
        summary["ranks"], ledger["ranks"], strict=True
    ):
        pid = expected["pid"]
        capture = json.loads((evidence / RUN / f"moe-deferred-{pid}.json").read_text())
        publication = json.loads(
            (evidence / RUN / f"gdn-publication-{pid}.json").read_text()
        )
        counts, arithmetic = Counter(), Counter()
        histogram = Counter()
        skipped = 0
        for frame, pairs in join(capture["frames"], publication["records"]):
            for group, record in pairs:
                counts.update(count(group, record))
                arithmetic.update(operations(group, record))
                histogram.update(chunks(group, record))
                fields = record["device_fields"]
                for j, i in enumerate(record["metadata"]["verify_ids"]):
                    if fields["verify_slots"][j][0] == -1:
                        skipped += fields["cu"][i + 1][0] - fields["cu"][i][0]
        assert skipped == counts["device_skipped_verify_tokens"] == 360
        assert counts == expected["cohorts"]["inside_request_under_clock_tolerance"]
        arithmetic["convolution_basic_arithmetic"] = (
            arithmetic["convolution_multiply"] + arithmetic["convolution_add"]
        )
        assert (
            arithmetic
            == expected_ledger["cohorts"]["inside_request_under_clock_tolerance"]
        )
        expected_prefill = next(r for r in prefill["ranks"] if r["pid"] == pid)[
            "cohorts"
        ]["inside_request_under_clock_tolerance"]
        assert [
            {"length": n, "has_state": state, "chunk_layers": number}
            for (n, state), number in sorted(histogram.items())
        ] == expected_prefill["chunk_catalog"]
        totals = Counter()
        for (n, state), number in histogram.items():
            for prefix, values in [
                ("ho", ho(n, state)),
                ("wy", wy(n)),
                ("inverse", inverse(n)),
            ]:
                totals.update(
                    {
                        f"{prefix}_{key}": value * number * 16
                        for key, value in values.items()
                    }
                )
        assert totals == expected_prefill["operations_per_rank"]
    assert ledger["model_effective_flops"] is ledger["m01_attainment"] is None


def test_request_coverage_and_uncovered_export_tail(evidence, monkeypatch):
    monkeypatch.syspath_prepend(str(evidence / SOURCE))
    from integrate_hbm_samples import has_complete_hbm_coverage, integrate
    from integrate_r12_hardware import link_bounds

    summary = json.loads((BASE / "r17-hardware-coverage.json").read_text())
    rows = [
        json.loads(line)
        for line in (evidence / SOURCE / "r17-system-control/hccs64.jsonl")
        .read_text()
        .splitlines()
    ]
    for index, window in enumerate(summary["windows"]):
        start, end = window["start_wall"] * 1e9, window["end_wall"] * 1e9
        assert link_bounds(rows, start, end) == window["hccs"]
        coverage = []
        for device in window["hbm_devices"]:
            path = evidence / device["db"].lstrip("/")
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                samples = connection.execute(
                    "SELECT device_id,timestamp,counts,hbmid,event_type FROM HBMOriginalData WHERE replayid=0"
                ).fetchall()
            actual = integrate(samples, start, end)
            coverage.append(has_complete_hbm_coverage(actual))
            for key, value in actual.items():
                if key == "prorated_bytes":
                    assert abs(value - device[key]) <= 8 * max(
                        math.ulp(value), math.ulp(device[key])
                    )
                else:
                    assert value == device[key]
        assert all(coverage) == window["hbm_coverage_complete"] == (index < 2)
        assert (window["hbm_bytes"] is None) == (index == 2)
