"""Recompute communication shape coverage and qualified calibration summaries."""

import bisect
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import statistics
import tarfile

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/collective-grid-v1"
)
PREFIX = "root/gennova-evidence/20261019-midterm-aligned/"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    index = json.loads((BASE / "archive.json").read_text())
    chunks = []
    for part in index["parts"]:
        data = (BASE / part["name"]).read_bytes()
        assert (
            len(data) == part["bytes"]
            and hashlib.sha256(data).hexdigest() == part["sha256"]
        )
        chunks.append(data)
    packed = b"".join(chunks)
    assert (
        len(packed) == index["archive_bytes"]
        and hashlib.sha256(packed).hexdigest() == index["archive_sha256"]
    )
    root = tmp_path_factory.mktemp("collective-grid")
    expected = {item["name"]: item for item in index["members"]}
    assert len(expected) == len(index["members"])
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in seen
            seen.add(member.name)
            data = archive.extractfile(member).read()
            assert (
                len(data) == expected[member.name]["bytes"]
                and hashlib.sha256(data).hexdigest() == expected[member.name]["sha256"]
            )
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            assert not target.exists()
            target.write_bytes(data)
    assert seen == set(expected)
    return root / PREFIX, root, index


def test_sources_and_original_evidence_retained(evidence):
    local, root, index = evidence
    for path in local.rglob("*.json"):
        record = json.loads(path.read_text())
        for source, digest in record.get("source_sha256", {}).items():
            candidate = root / source.lstrip("/")
            if candidate.exists():
                assert hashlib.sha256(candidate.read_bytes()).hexdigest() == digest
            else:
                assert index["external_full_profiler_database_hashes"][source] == digest
    receipt = json.loads(
        (local / "collective-grid-control-r1/receipt.json").read_text()
    )
    assert receipt["probe_exit_code"] == receipt["cleanup_exit_code"] == 0
    assert receipt["all_devices_released"] and not receipt["sampler_errors"]
    for name, digest in receipt["input_hashes"].items():
        assert hashlib.sha256((local / name).read_bytes()).hexdigest() == digest
    assert (local / "collective-grid-control-r1/summary.json").is_file()


def test_histograms_recomputed_from_exact_source_tables(evidence):
    local, _, _ = evidence
    grid = json.loads(
        (local / "diagnostic-collective-payload-grid-v1.json").read_text()
    )
    provenance = json.loads(
        (local / "collective-source-tables-v1/provenance.json").read_text()
    )
    for rank, record in enumerate(provenance["records"]):
        path = local / "collective-source-tables-v1" / record["extracted"]
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == record["extracted_sha256"]
        )
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as database:
            rows = database.execute(
                "SELECT s.value,d.name,o.count,o.rankSize,COUNT(*) FROM COMMUNICATION_OP o JOIN STRING_IDS s ON s.id=o.opType JOIN ENUM_HCCL_DATA_TYPE d ON d.id=o.dataType GROUP BY s.value,d.name,o.count,o.rankSize ORDER BY s.value,d.name,o.count,o.rankSize"
            ).fetchall()
            assert (
                database.execute("SELECT COUNT(*) FROM COMMUNICATION_OP").fetchone()[0]
                == 119098
            )
        recovered = []
        for name, dtype, elements, ranks, calls in rows:
            width = {"BFP16": 2, "INT64": 8}[dtype]
            recovered.append(
                dict(
                    kind={
                        "hcom_allGather_": "all_gather",
                        "hcom_allReduce_": "all_reduce",
                    }[name],
                    dtype=dtype,
                    elements=elements,
                    rank_size=ranks,
                    calls=calls,
                    input_bytes_per_call=elements * width,
                    logical_input_bytes=elements * width * calls,
                )
            )
        assert recovered == grid["ranks"][rank]["histogram"]
        assert len(recovered) == 56
        assert sum(row["logical_input_bytes"] for row in recovered) == 161361797408


def test_all_repeats_and_medians_not_peak_selection(evidence):
    local, _, _ = evidence
    output = json.loads(
        (local / "collective-grid-control-r1/summary-v2.json").read_text()
    )
    ranks = [
        json.loads((local / f"collective-grid-r1/rank-{rank}.json").read_text())[
            "phases"
        ]
        for rank in (0, 1)
    ]
    assert len(output["phases"]) == len(ranks[0]) == len(ranks[1]) == 168
    assert len(output["shapes"]) == 56
    assert {(p["shape_index"], p["repetition"]) for p in output["phases"]} == {
        (shape, repeat) for shape in range(56) for repeat in range(3)
    }
    assert output["m01_attainment"] is None
    assert output["boundary_rule"]["start_offset_seconds"] == -0.25
    for phase, a, b in zip(output["phases"], *ranks, strict=True):
        assert (
            a["calls"] == b["calls"] == 512
            and a["numerics_passed"]
            and b["numerics_passed"]
        )
        duration = max(
            a["end_monotonic"] - a["start_monotonic"],
            b["end_monotonic"] - b["start_monotonic"],
        )
        assert duration == phase["duration_seconds"]
        assert phase["counter_qualified"] and phase["peer_exact"]
        assert (
            phase["aggregate_tx_flits_per_second"]
            == sum(sum(d["tx_flits"]) for d in phase["devices"]) / duration
        )
    for index, shape in enumerate(output["shapes"]):
        phases = [p for p in output["phases"] if p["shape_index"] == index]
        assert len(phases) == shape["counter_qualified_repetitions"] == 3
        signatures = {
            tuple(tuple(d[k]) for d in p["devices"] for k in ("tx_flits", "rx_flits"))
            for p in phases
        }
        assert len(signatures) == 1
        rates = [p["aggregate_tx_flits_per_second"] for p in phases]
        assert shape["median_aggregate_tx_flits_per_second"] == statistics.median(rates)
        assert shape["min_aggregate_tx_flits_per_second"] == min(rates)
        assert shape["max_aggregate_tx_flits_per_second"] == max(rates)


def test_raw_counter_anchors_recomputed(evidence):
    local, _, _ = evidence
    control = local / "collective-grid-control-r1"
    output = json.loads((control / "summary-v2.json").read_text())
    raw = [
        json.loads(line)
        for line in (control / "counters.jsonl").read_text().splitlines()
    ]
    indexed = {}
    for card in (4, 5):
        rows = [r for r in raw if r["card"] == card]
        times = [(r["before"]["wall_ns"] + r["after"]["wall_ns"]) // 2 for r in rows]
        assert times == sorted(times)
        assert all(
            y >= x
            for before, after in zip(rows, rows[1:])
            for key in ("tx_counts", "rx_counts")
            for x, y in zip(before[key], after[key], strict=True)
        )
        indexed[card] = rows, times
    ranks = [
        json.loads((local / f"collective-grid-r1/rank-{rank}.json").read_text())[
            "phases"
        ]
        for rank in (0, 1)
    ]
    for phase, a, b in zip(output["phases"], *ranks, strict=True):
        start, end = (
            min(a["start_wall"], b["start_wall"]),
            max(a["end_wall"], b["end_wall"]),
        )
        for device in phase["devices"]:
            rows, times = indexed[device["card"]]
            endpoints = []
            for offset, target in enumerate((start - 0.25, end + 0.75)):
                ns = round(target * 1e9)
                location = bisect.bisect_left(times, ns)
                index = min((location - 1, location), key=lambda i: abs(times[i] - ns))
                assert 0 <= index < len(times) and abs(times[index] - ns) <= 100000000
                left = bisect.bisect_left(times, times[index] - 100000000)
                right = bisect.bisect_right(times, times[index] + 100000000)
                assert right - left >= 3 and times[right - 1] - times[left] >= 100000000
                assert all(
                    row[key] == rows[index][key]
                    for row in rows[left:right]
                    for key in ("tx_counts", "rx_counts")
                )
                if offset == 0:
                    assert times[index] / 1e9 + 0.1 < start
                    assert (
                        times[index] / 1e9 - 0.1
                        > max(a["start_wall"], b["start_wall"]) - 1.0
                    )
                else:
                    assert times[index] / 1e9 - 0.1 > end
                endpoints.append(rows[index])
            for key in ("tx", "rx"):
                assert [
                    y - x
                    for x, y in zip(
                        endpoints[0][key + "_counts"],
                        endpoints[1][key + "_counts"],
                        strict=True,
                    )
                ] == device[key + "_flits"]
