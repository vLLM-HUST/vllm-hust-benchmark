"""Portable qualification coverage and provenance, not new device execution."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/preprocess-publication-20261021"
)


@pytest.fixture(scope="module")
def evidence():
    index = json.loads((BASE / "archive.json").read_text())
    data = (BASE / index["archive"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == index["archive_sha256"]
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for item in archive:
            path = PurePosixPath(item.name)
            assert item.isfile() and not path.is_absolute() and ".." not in path.parts
            assert item.name not in files
            value = archive.extractfile(item).read()
            assert (
                hashlib.sha256(value).hexdigest()
                == index["members"][item.name]["sha256"]
            )
            assert len(value) == index["members"][item.name]["bytes"]
            files[item.name] = value
    assert files.keys() == index["members"].keys()
    return index, files


def test_device_graph_coverage(evidence):
    _, files = evidence
    prefix = "m01-preprocess-publication-recovery-r3/"
    result = json.loads(files[prefix + "result.json"])
    receipt = json.loads(files[prefix + "receipt.json"])
    audit = json.loads(files[prefix + "independent-audit.json"])
    assert receipt["exit_code"] == 0 and receipt["all_four_idle_after"]
    assert result["status"] == "passed" and audit["status"] == "cpu_reaudit_passed"
    assert (
        audit["result_sha256"]
        == hashlib.sha256(files[prefix + "result.json"]).hexdigest()
    )
    assert audit["cases"] == len(result["cases"]) == 64
    assert audit["numerical_tensor_checks"] == 560
    assert {
        (r["device"], r["capacity"], r["bank"], r["replay"]) for r in result["cases"]
    } == {
        (device, capacity, bank, replay)
        for device in (0, 1)
        for capacity in (80, 1024, 2048, 4096)
        for bank in (0, 1)
        for replay in range(4)
    }
    assert all(r["variant"] == r["replay"] % 2 for r in result["cases"])
    assert all(
        len(r["checks"]) == (5 if r["capacity"] == 80 else 10) for r in result["cases"]
    )
    assert all(
        check["max_abs"] == 0
        for row in result["cases"]
        for check in row["checks"]
        if check["output"] == "v"
    )


def test_failure_retention_and_no_false_admission(evidence):
    index, files = evidence
    for attempt in ("r1", "r2"):
        receipt = json.loads(
            files[f"m01-preprocess-publication-recovery-{attempt}/receipt.json"]
        )
        assert receipt["status"] == "failed" and receipt["exit_code"] != 0
        assert receipt["all_four_idle_after"]
    assert not index["full_model"] and not index["historical_device_maps_recovered"]
    assert index["model_effective_flops"] is index["m01_attainment"] is None
    manifest = json.loads(files["raw-archive-manifest.json"])
    assert index["raw_archive"]["sha256"] == manifest["sha256"]
    assert sum(name.endswith(".pt") for name in manifest["members"]) == 64
    result = json.loads(files["m01-preprocess-publication-recovery-r3/result.json"])
    for case in result["cases"]:
        raw = manifest["members"][
            f"m01-preprocess-publication-recovery-r3/{case['id']}.pt"
        ]
        assert raw["sha256"] == case["tensor_sha256"]


def test_request_identity_does_not_become_quality(evidence):
    _, files = evidence
    result = json.loads(files["source/recovery-request-identity-r1.json"])
    assert not result["quality_admission"]
    assert result["first_divergent_generated_token"] is None
    assert len({r["case_id"] for r in result["records"]}) == 200
    for run in result["runs"]:
        assert run["outputs_at_configured_cap"] == 200
        assert run["task_accuracy"] is None
        assert not run["finish_reason_retained"] and not run["token_ids_retained"]
