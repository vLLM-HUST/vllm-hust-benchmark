"""Recompute r16 coverage from immutable raw evidence on CPU."""

import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/owned-fia-r15-r16"
)
SOURCE = "root/gennova-evidence/20261019-midterm-aligned"
RUN = "workspace/gennova-measurements/microflow-token-accounting-r16"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    index = json.loads((BASE / "archive.json").read_text())
    parts = []
    for part in index["parts"]:
        data = (BASE / part["file"]).read_bytes()
        assert len(data) == part["bytes"]
        assert hashlib.sha256(data).hexdigest() == part["sha256"]
        parts.append(data)
    data = b"".join(parts)
    assert hashlib.sha256(data).hexdigest() == index["archive_sha256"]
    root = tmp_path_factory.mktemp("r16")
    expected = {m["name"]: m for m in index["members"]}
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in seen
            seen.add(member.name)
            content = archive.extractfile(member).read()
            assert len(content) == expected[member.name]["bytes"]
            assert (
                hashlib.sha256(content).hexdigest() == expected[member.name]["sha256"]
            )
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    assert seen == set(expected)
    return root


@pytest.fixture
def sources(evidence, monkeypatch):
    monkeypatch.syspath_prepend(str(evidence / SOURCE))
    return evidence / SOURCE


def test_pairing_branches_and_page_ranges(evidence, sources):
    from await_owned_fia_plans import validate_plans
    from audit_r15_blocks import used_pages

    receipt = json.loads((evidence / RUN / "receipt.json").read_text())
    for name, digest in receipt["input_hashes"].items():
        path = evidence / name.lstrip("/") if name.startswith("/") else sources / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    summary = json.loads((BASE / "r16-capture-summary.json").read_text())
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    for export in receipt["owned_fia_exports"]:
        assert validate_plans(evidence / RUN, export) == export["plan_capture"]
        pid = export["pid"]
        rank = next(r for r in summary["ranks"] if r["pid"] == pid)
        lengths = json.loads((evidence / RUN / f"owned-fia-{pid}.json").read_text())
        blocks = json.loads(
            (evidence / RUN / f"owned-fia-blocks-{pid}.json").read_text()
        )
        totals = {}
        for length, block in zip(lengths["records"], blocks["records"], strict=True):
            meta = length["metadata"]
            backend = "context_parallel" if meta["context_parallel"] else "owned"
            capacities = {
                g["tensors"]["key"]["shape"][0]
                for g in rank["geometry"]
                if g["tokens"] == meta["tokens"] and g["backend"] == backend
            }
            assert len(capacities) == 1
            role = (
                "target"
                if meta["role"] == "target"
                else f"draft_step{meta['draft_step']}"
            )
            counts = {"publications": 1, **used_pages(block, length, capacities.pop())}
            for key, value in counts.items():
                totals.setdefault(role, {}).setdefault(key, 0)
                totals[role][key] += value
        assert totals == rank["roles"]
    assert summary["model_effective_flops"] is summary["m01_attainment"] is None


def test_same_interval_counters(evidence, sources):
    from integrate_r16_hardware import boundaries
    from integrate_r12_hardware import link_bounds
    from integrate_hbm_samples import check_export, has_complete_hbm_coverage, integrate

    summary = json.loads((BASE / "r16-hardware-intervals.json").read_text())
    ranks, windows, _ = boundaries(evidence / RUN, summary["request_phase"])
    assert ranks == summary["ranks"]
    rows = [
        json.loads(line)
        for line in (sources / "r16-system-control/hccs64.jsonl")
        .read_text()
        .splitlines()
    ]
    for actual_window, window in zip(windows, summary["windows"], strict=True):
        assert all(window[k] == v for k, v in actual_window.items())
        start, end = window["start_wall"] * 1e9, window["end_wall"] * 1e9
        assert link_bounds(rows, start, end) == window["hccs"]
        for device in window["hbm_devices"]:
            db = evidence / device["db"].lstrip("/")
            assert hashlib.sha256(db.read_bytes()).hexdigest() == device["sha256"]
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                samples = conn.execute(
                    "SELECT device_id,timestamp,counts,hbmid,event_type FROM HBMOriginalData WHERE replayid=0"
                ).fetchall()
                exported = conn.execute("SELECT * FROM HBMbwData").fetchall()
            assert check_export(samples, exported) == device["decoder_crosscheck"]
            actual = integrate(samples, start, end)
            assert has_complete_hbm_coverage(actual)
            for key, value in actual.items():
                if key == "prorated_bytes":
                    assert abs(value - device[key]) <= 8 * max(
                        math.ulp(value), math.ulp(device[key])
                    )
                else:
                    assert value == device[key]


def test_source_pins_and_preserved_failure(evidence):
    for filename in (
        "r15-capture-gap.json",
        "r16-capture-summary.json",
        "r16-hardware-intervals.json",
    ):
        summary = json.loads((BASE / filename).read_text())
        for path, digest in summary["source_sha256"].items():
            assert (
                hashlib.sha256((evidence / path.lstrip("/")).read_bytes()).hexdigest()
                == digest
            )
    old = json.loads(
        (
            evidence
            / "workspace/gennova-measurements/microflow-token-accounting-r15/receipt.json"
        ).read_text()
    )
    assert old["status"] == "failed" and old["all_devices_released"]
