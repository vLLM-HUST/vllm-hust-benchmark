"""Verify r20 portable raw evidence, observed layouts and interval arithmetic."""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile
import xml.etree.ElementTree as ET

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/r20-layout-accounting"
)
SOURCE = "root/gennova-evidence/20261019-midterm-aligned"
RUN = "workspace/gennova-measurements/microflow-token-accounting-r20"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("r20")
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
    expected = {row["name"]: row for row in index["members"]}
    assert len(expected) == len(index["members"])
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
    for path in BASE.glob("r20-*.json"):
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
    assert (result["completed"], result["failed"], result["total_output_tokens"]) == (
        200,
        0,
        51200,
    )
    suite = ET.parse(evidence / SOURCE / "r20-recovery-tests.xml")
    assert len(suite.findall(".//testcase")) == 17
    assert not suite.findall(".//failure") and not suite.findall(".//error")


def test_layouts_recovered_from_worker_records(evidence):
    summary = json.loads((BASE / "r20-loaded-gmm-layout.json").read_text())
    observed = {}
    for trace in (evidence / RUN).glob("token-accounting-*.jsonl"):
        selected = {}
        with trace.open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["event"] in (
                    "observer_installed",
                    "loaded_model_inventory",
                    "loaded_layout_policy",
                ):
                    assert event["event"] not in selected
                    selected[event["event"]] = event
        rank = selected["observer_installed"]["rank"]
        observed[rank] = selected
    assert set(observed) == {0, 1}
    total = 0
    for rank in summary["ranks"]:
        raw = observed[rank["rank"]]
        assert rank["policy"] == raw["loaded_layout_policy"]
        for role, count in (("target", 40), ("draft", 1)):
            inventory = raw["loaded_model_inventory"][role]
            assert inventory["schema_version"] == 3
            modules = {
                m["name"]: m
                for m in inventory["modules"]
                if m["type"].endswith(".AscendRoutedExperts")
            }
            assert len(modules) == count
            assert len(rank["weights"][role]) == count * 2
            for row in rank["weights"][role]:
                tensor = modules[row["module"]]["parameters"][row["weight"]]
                assert all(row[key] == value for key, value in tensor.items())
                _, k, n = tensor["shape"]
                assert tensor["stride"] == [k * n, n, 1]
                assert tensor["npu_format"] == 2 and tensor["storage_offset"] == 0
                assert tensor["is_contiguous"] and row["matches_numeric_probe_layout"]
                total += 1
    assert total == 164 and summary["all_match_numeric_probe_layout"]
    families = json.loads((BASE / "r20-accounted-families.json").read_text())
    assert families["model_effective_flops"] is families["m01_attainment"] is None
    assert len({r["family"] for r in families["families"]}) == 10
    assert (
        sum(r["global_basic_operations"] for r in families["families"])
        == families["accounted_basic_operation_subtotal"]
    )


def test_hardware_intervals_from_raw_samples(evidence, monkeypatch):
    monkeypatch.syspath_prepend(str(evidence / SOURCE))
    from integrate_hbm_samples import has_complete_hbm_coverage, integrate
    from integrate_r12_hardware import link_bounds

    summary = json.loads((BASE / "r20-hardware-intervals.json").read_text())
    rows = [
        json.loads(line)
        for line in (evidence / SOURCE / "r20-system-control/hccs64.jsonl")
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
