"""Check portable GMM evidence without loading NPU libraries."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import xml.etree.ElementTree as ET


def test_gmm_archive_and_claims():
    base = (
        Path(__file__).resolve().parents[1]
        / "reports/vllm-hust-midterm-20261019/m01-recovery/gmm-numeric-v1"
    )
    index = json.loads((base / "archive.json").read_text())
    chunks = []
    for part in index["parts"]:
        value = (base / part["name"]).read_bytes()
        assert len(value) == part["bytes"]
        assert hashlib.sha256(value).hexdigest() == part["sha256"]
        chunks.append(value)
    packed = b"".join(chunks)
    assert len(packed) == index["archive_bytes"]
    assert hashlib.sha256(packed).hexdigest() == index["archive_sha256"]
    expected = {row["name"]: row for row in index["members"]}
    assert len(expected) == len(index["members"])
    files = {}
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        for item in archive:
            path = PurePosixPath(item.name)
            assert item.isfile() and not path.is_absolute() and ".." not in path.parts
            assert item.name not in files
            value = archive.extractfile(item).read()
            assert len(value) == expected[item.name]["bytes"]
            assert hashlib.sha256(value).hexdigest() == expected[item.name]["sha256"]
            files[item.name] = value
    assert files.keys() == expected.keys()
    run = "workspace/gennova-measurements/gmm-numeric-v1-qualification/"
    root = "root/gennova-evidence/20261019-midterm-aligned/"
    receipt = json.loads(files[run + "receipt.json"])
    assert receipt["status"] == "passed" and receipt["exit_code"] == 0
    assert receipt["all_four_idle_after"] and not receipt["performance_publication"]
    for name, digest in receipt["source_sha256"].items():
        assert hashlib.sha256(files[name.lstrip("/")]).hexdigest() == digest
    result = json.loads(files[run + "device-result.json"])
    assert (
        hashlib.sha256(files[run + "sampled-numerics.pt"]).hexdigest()
        == result["tensor_sha256"]
    )
    assert result["model_effective_flops"] is result["m01_attainment"] is None
    assert {case["id"] for case in result["cases"]} == {
        f"device{device}-m{m}-{op}"
        for device in (0, 1)
        for m in (3, 4096)
        for op in ("gmm1", "gmm2")
    }
    for case in result["cases"]:
        assert case["total_rows"] == sum(case["group_counts"])
        assert case["sampled_rows"] < case["total_rows"]
        assert case["rtol"] == case["atol"] == 0.02
        assert case["weight_npu_format"] == 2
    layout = json.loads(files[root + "r19-gmm-layout-source-audit-v1.json"])
    assert (
        layout["actual_r19_device_format"] is layout["actual_r19_device_stride"] is None
    )
    suite = ET.fromstring(files[root + "gmm-numeric-v1-tests.xml"])
    assert len(suite.findall(".//testcase")) == 10
    assert not suite.findall(".//failure") and not suite.findall(".//error")
