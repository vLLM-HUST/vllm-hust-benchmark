"""Validate r19 raw archive, logits boundaries and same-window hardware traffic."""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019"
    / "m01-recovery/r19-direct-logits"
)
SOURCE = "root/gennova-evidence/20261019-midterm-aligned"
RUN = "workspace/gennova-measurements/microflow-token-accounting-r19"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("r19")
    index = json.loads((BASE / "archive.json").read_text())
    packed = root / "evidence.tar.gz"
    digest = hashlib.sha256()
    with packed.open("wb") as stream:
        for part in index["parts"]:
            data = (BASE / part["file"]).read_bytes()
            assert len(data) == part["bytes"]
            assert hashlib.sha256(data).hexdigest() == part["sha256"]
            digest.update(data)
            stream.write(data)
    assert packed.stat().st_size == index["archive_bytes"]
    assert digest.hexdigest() == index["archive_sha256"]
    expected = {item["name"]: item for item in index["members"]}
    seen = set()
    with tarfile.open(packed) as archive:
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
        item = json.loads(path.read_text())
        for source, digest in item["source_sha256"].items():
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
    result = json.loads((evidence / RUN / "mmlu-pro.json").read_text())
    assert result["completed"] == 200 and result["failed"] == 0
    assert result["total_output_tokens"] == 51200


def test_direct_events_and_tail_recomputed(evidence, monkeypatch):
    monkeypatch.syspath_prepend(str(evidence / SOURCE))
    from audit_r19_logits import reconcile
    from integrate_r18_hardware import classify

    summary = json.loads((BASE / "vocab-projection-summary-r19.json").read_text())
    direct = json.loads((BASE / "r19-direct-logits.json").read_text())
    window = json.loads((BASE / "r19-vocab-window.json").read_text())
    for rank in direct["ranks"]:
        raw = json.loads(
            (evidence / RUN / f"moe-deferred-{rank['pid']}.json").read_text()
        )
        events = [
            json.loads(line)
            for line in (evidence / RUN / f"token-accounting-{rank['pid']}.jsonl")
            .read_text()
            .splitlines()
        ]
        phase = summary["request_phase"]
        host = [
            item
            for item in events
            if item["event"] == "target_logits_input"
            and phase["start_wall"] <= item["wall"] <= summary["host_selection_end"]
        ]
        records = raw["target_logits"]["records"]
        reconcile(records, host)
        assert len(records) == rank["calls"] == 876
        a, b = raw["anchor_host_bracket"]
        counts = {}
        for item in records:
            start = [
                a + item["start_ms"] / 1000 - 0.002,
                b + item["start_ms"] / 1000 + 0.002,
            ]
            end = [a + item["end_ms"] / 1000 - 0.002, b + item["end_ms"] / 1000 + 0.002]
            key = classify(start, end, phase)
            counts[key] = counts.get(key, 0) + item["rows"]
        assert (
            counts
            == rank["target_rows_by_device_window"]
            == {"inside": 57433, "after": 3}
        )
        draft_counts = {}
        for frame in raw["frames"][1::2]:
            assert frame["role"] == "draft"
            start = [
                a + frame["start_ms"] / 1000 - 0.002,
                b + frame["start_ms"] / 1000 + 0.002,
            ]
            end = [
                a + frame["end_ms"] / 1000 - 0.002,
                b + frame["end_ms"] / 1000 + 0.002,
            ]
            key = classify(start, end, phase)
            draft_counts[key] = (
                draft_counts.get(key, 0)
                + frame["fields"]["draft_steps"][1]["actual_tokens"]
            )
        rows = next(row for row in window["ranks"] if row["rank"] == rank["rank"])[
            "rows_by_device_window"
        ]
        assert rows["target"] == counts
        assert (
            rows["draft_step0"]
            == rows["draft_step1"]
            == draft_counts
            == {"inside": 19291, "after": 1}
        )
    total = 0
    for role, categories in window["semantic_projection_flops_by_window"].items():
        head = "target" if role == "target" else "draft"
        factor = sum(
            2 * row["loaded_head_shapes"][head][0] * row["loaded_head_shapes"][head][1]
            for row in summary["ranks"]
        )
        for key, value in categories.items():
            assert (
                value == factor * window["ranks"][0]["rows_by_device_window"][role][key]
            )
            if key == "inside":
                total += value
    assert total == window["inside_projection_subtotal"] == 97658653900800
    assert total < summary["semantic_projection_subtotal"]
    assert window["model_effective_flops"] is window["m01_attainment"] is None


def test_hardware_intervals_from_raw_samples(evidence, monkeypatch):
    monkeypatch.syspath_prepend(str(evidence / SOURCE))
    from integrate_hbm_samples import has_complete_hbm_coverage, integrate
    from integrate_r12_hardware import link_bounds

    summary = json.loads((BASE / "r19-hardware-intervals.json").read_text())
    rows = [
        json.loads(line)
        for line in (evidence / SOURCE / "r19-system-control/hccs64.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(summary["windows"]) == 3
    for window in summary["windows"]:
        start, end = window["start_wall"] * 1e9, window["end_wall"] * 1e9
        assert link_bounds(rows, start, end) == window["hccs"]
        for device in window["hbm_devices"]:
            path = evidence / device["db"].lstrip("/")
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                samples = connection.execute(
                    "SELECT device_id,timestamp,counts,hbmid,event_type FROM HBMOriginalData WHERE replayid=0"
                ).fetchall()
            actual = integrate(samples, start, end)
            assert has_complete_hbm_coverage(actual)
            for key, value in actual.items():
                if key == "prorated_bytes":
                    assert abs(value - device[key]) <= 8 * max(
                        math.ulp(value), math.ulp(device[key])
                    )
                else:
                    assert value == device[key]
        for key, value in window["hbm_bytes"].items():
            assert math.isclose(
                value, sum(d[key] for d in window["hbm_devices"]), rel_tol=1e-14
            )
