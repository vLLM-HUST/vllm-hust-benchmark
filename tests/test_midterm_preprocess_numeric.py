"""Validate published numerical evidence without importing the NPU runtime."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import xml.etree.ElementTree as ET

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/preprocess-numeric-v2"
)
ROOT = "root/gennova-evidence/20261019-midterm-aligned/"
RUN = "workspace/gennova-measurements/preprocess-numeric-v{}-qualification/"


@pytest.fixture(scope="module")
def files():
    index = json.loads((BASE / "archive.json").read_text())
    chunks = []
    for part in index["parts"]:
        value = (BASE / part["name"]).read_bytes()
        assert len(value) == part["bytes"]
        assert hashlib.sha256(value).hexdigest() == part["sha256"]
        chunks.append(value)
    packed = b"".join(chunks)
    assert len(packed) == index["archive_bytes"]
    assert hashlib.sha256(packed).hexdigest() == index["archive_sha256"]
    expected = {item["name"]: item for item in index["members"]}
    assert len(expected) == len(index["members"])
    result = {}
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        for item in archive:
            path = PurePosixPath(item.name)
            assert item.isfile() and not path.is_absolute() and ".." not in path.parts
            assert item.name not in result
            value = archive.extractfile(item).read()
            assert len(value) == expected[item.name]["bytes"]
            assert hashlib.sha256(value).hexdigest() == expected[item.name]["sha256"]
            result[item.name] = value
    assert set(result) == set(expected)
    return result


def test_provenance_failures_and_release(files):
    for version, code, status in ((1, 1, "failed"), (2, 0, "passed")):
        receipt = json.loads(files[RUN.format(version) + "receipt.json"])
        assert receipt["exit_code"] == code and receipt["status"] == status
        assert receipt["all_four_idle_after"] and not receipt["performance_publication"]
        for source, digest in receipt.get("source_sha256", {}).items():
            assert hashlib.sha256(files[source.lstrip("/")]).hexdigest() == digest
    assert b"ModuleNotFoundError" in files[RUN.format(1) + "child.log"]
    result = json.loads(files[RUN.format(2) + "device-result.json"])
    for source, digest in result["source_sha256"].items():
        assert hashlib.sha256(files[source.lstrip("/")]).hexdigest() == digest
    assert result["model_run"] is result["performance_publication"] is False
    raw = result["tensor_archive"]
    data = files[RUN.format(2) + raw["name"]]
    assert (
        len(data) == raw["bytes"] and hashlib.sha256(data).hexdigest() == raw["sha256"]
    )


def test_cases_tolerances_and_local_recomputation(files):
    result = json.loads(files[RUN.format(2) + "device-result.json"])
    cases = result["cases"]
    assert len(cases) == len({item["id"] for item in cases}) == 10
    assert {(item["output_rows"], item["head_major"]) for item in cases} == {
        (rows, major) for rows in (1, 16, 17, 72, 257) for major in (False, True)
    }
    for item in cases:
        assert {check["output"] for check in item["checks"]} == {
            "q",
            "k",
            "v",
            "g",
            "beta",
        }
        for check in item["checks"]:
            expected = (
                (0, 0)
                if check["output"] == "v"
                else (2e-5, 2e-5)
                if check["output"] == "g"
                else (0.016, 0.001)
            )
            assert (check["rtol"], check["atol"]) == expected
            assert check["max_absolute_error"] >= 0
    suite = ET.fromstring(files[ROOT + "preprocess-numeric-v2-tests.xml"])
    tests = suite.findall(".//testcase")
    assert len(tests) == 3
    assert (
        not suite.findall(".//failure")
        and not suite.findall(".//error")
        and not suite.findall(".//skipped")
    )
