"""Verify portable GMM evidence and recompute graph reference rates."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import tarfile

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/gmm-reference"
)
RUN = "workspace/gennova-measurements/gmm-capacity-r5-graph"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("gmm-reference")
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
    assert digest.hexdigest() == index["archive_sha256"]
    assert packed.stat().st_size == index["archive_bytes"]
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


def test_lifecycle_sources_and_changed_input(evidence):
    run = evidence / RUN
    receipt = json.loads((run / "receipt.json").read_text())
    assert receipt["exit_code"] == receipt["cleanup_exit_code"] == 0
    assert receipt["all_four_idle_after"]
    assert not receipt["protocol"]["prior_cache_imported"]
    for name, digest in receipt["source_sha256"].items():
        assert (
            hashlib.sha256((evidence / name.lstrip("/")).read_bytes()).hexdigest()
            == digest
        )
    cpu = [
        json.loads(line)
        for line in (run / "device-samples.jsonl").read_text().splitlines()
    ]
    cpu = [row["cpu"] for row in cpu if "cpu" in row]
    assert cpu and max(row["foreign_cpu_cores"] for row in cpu) <= 1.0
    for rank in range(2):
        rows = [
            json.loads(line)
            for line in (run / f"graph-numerics-{rank}.jsonl").read_text().splitlines()
        ]
        assert [row["case_index"] for row in rows] == list(range(146))
        assert all(
            row["changed_input_scale"] == -1 and row["checked_outputs"] == 8
            for row in rows
        )


def test_reference_rates_not_full_model_attainment(evidence):
    run = evidence / RUN
    result = json.loads((run / "summary.json").read_text())
    assert result["model_effective_flops"] is result["m01_attainment"] is None
    ranks = [
        [
            json.loads(line)
            for line in (run / f"phases-{rank}.jsonl").read_text().splitlines()
        ]
        for rank in range(2)
    ]
    for rows in ranks:
        assert [(r["case_index"], r["repetition"]) for r in rows] == [
            (c, r) for c in range(146) for r in range(3)
        ]
    assert len(result["cases"]) == 146
    for case in result["cases"]:
        config = case["case"]
        rates = []
        for rep in range(3):
            a, b = [rank[config["index"] * 3 + rep] for rank in ranks]
            assert a["calls"] == b["calls"] == 128
            operations = 2 * sum(config["counts"]) * config["K"] * config["N"] * 128
            assert a["operations_per_rank"] == b["operations_per_rank"] == operations
            duration = max(a["end_monotonic"], b["end_monotonic"]) - min(
                a["start_monotonic"], b["start_monotonic"]
            )
            rates.append(2 * operations / duration)
        assert case["median_operations_per_second"] == statistics.median(rates)
        assert case["sample_stdev_operations_per_second"] == statistics.stdev(rates)


def test_rejected_runs_not_admitted(evidence):
    for name in ("gmm-capacity-r3-graph", "gmm-capacity-r4-graph"):
        receipt = json.loads(
            (
                evidence / "workspace/gennova-measurements" / name / "receipt.json"
            ).read_text()
        )
        assert receipt["status"] == "failed"
        assert receipt["all_four_idle_after"]
        assert not receipt["performance_publication"]
