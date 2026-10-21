"""Portable verification of recovered FIA evidence, not M01 admission."""

import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import tarfile

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/runtime-recovery"
)


@pytest.fixture(scope="module")
def evidence():
    index = json.loads((BASE / "archive.json").read_text())
    chunks = []
    for part in index["parts"]:
        data = (BASE / part["file"]).read_bytes()
        assert len(data) == part["bytes"]
        assert hashlib.sha256(data).hexdigest() == part["sha256"]
        chunks.append(data)
    packed = b"".join(chunks)
    assert len(packed) == index["archive_bytes"]
    assert hashlib.sha256(packed).hexdigest() == index["archive_sha256"]
    expected = {r["name"]: r for r in index["members"]}
    assert len(expected) == len(index["members"])
    records = {}
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in records
            data = archive.extractfile(member).read()
            assert len(data) == expected[member.name]["size"]
            assert hashlib.sha256(data).hexdigest() == expected[member.name]["sha256"]
            records[member.name] = data
    assert records.keys() == expected.keys()
    return records


def flatten(value):
    if isinstance(value, list):
        for child in value:
            yield from flatten(child)
    else:
        yield value


@pytest.mark.parametrize("graph", [False, True])
def test_saved_numeric_arrays(evidence, graph):
    prefix = "fia-recovery-graph" if graph else "fia-recovery"
    plan = json.loads(evidence[f"{prefix}-plan-r1.json"])
    receipt = json.loads(evidence[f"{prefix}-control-r1/receipt.json"])
    assert receipt["probe_exit_code"] == receipt["cleanup_exit_code"] == 0
    assert receipt["all_devices_released"] and not receipt["sampler_errors"]
    for rank in (0, 1):
        raw = json.loads(evidence[f"{prefix}-r1/rank-{rank}.json"])
        assert len(raw["cases"]) == 18 and raw["rank"] == rank
        total = 0
        for case in raw["cases"]:
            variants = case["variants"] if graph else [case]
            if graph:
                assert [v["query_scale"] for v in variants] == [1, -1, 2]
                original = [c["actual"] for c in variants[0]["checks"]]
                for variant in variants[1:]:
                    assert [c["actual"] for c in variant["checks"]] != original
            for variant in variants:
                for check in variant["checks"]:
                    actual, reference = (
                        list(flatten(check["actual"])),
                        list(flatten(check["reference"])),
                    )
                    assert len(actual) == len(reference) > 0
                    for a, b in zip(actual, reference, strict=True):
                        assert math.isfinite(a) and math.isfinite(b)
                        assert abs(a - b) <= plan["atol"] + plan["rtol"] * abs(b)
                    total += len(actual)
        assert total == (331776 if graph else 110592)


def test_no_missing_run_or_model_admission(evidence):
    recovery = json.loads(evidence["r23-recovery-audit.json"])
    assert (
        recovery["status"]
        == "original_paths_unavailable_after_container_change_not_admitted"
    )
    assert not any(recovery["required_path_exists"].values())
    for name in ("fia-recovery-numeric-audit.json", "fia-recovery-graph-audit.json"):
        audit = json.loads(evidence[name])
        assert (
            audit["model_effective_flops"] is None and audit["m01_attainment"] is None
        )
    build = json.loads(evidence["recovery-fia-build-r1/recovery-receipt.json"])
    assert build["matches_original_artifact"] is False
    assert build["artifact_sha256"] != build["original_artifact_sha256"]
