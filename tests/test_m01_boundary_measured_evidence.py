import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "reports/vllm-hust-midterm-20261019/m01-recovery/boundary-measured-20261023"
)


def test_measured_boundary_identity_and_no_premature_attainment():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    archive = EVIDENCE / manifest["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"]
    files = {}
    with tarfile.open(archive) as bundle:
        for member in bundle:
            assert (
                not Path(member.name).is_absolute()
                and ".." not in Path(member.name).parts
            )
            assert member.isdir() or member.isfile()
            if member.isfile():
                assert member.name not in files
                files[member.name] = bundle.extractfile(member).read()
    assert {
        name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
    } == manifest["members"]
    prefix = "m01-boundary-b1-r1/"
    for line in files[prefix + "SHA256SUMS"].decode().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(files[prefix + name]).hexdigest() == digest
    receipt = json.loads(files[prefix + "receipt.json"])
    result = json.loads(files[prefix + "mmlu-pro.json"])
    boundary = json.loads(files[prefix + "boundary.jsonl"])
    assert receipt["client_exit_code"] == 0
    assert receipt["sessions_drained"] and receipt["all_devices_released"]
    assert result["completed"] == 200 and result["failed"] == 0
    assert sum(result["input_lens"]) == 282265
    assert result["output_lens"] == [256] * 200
    assert not any(result["errors"])
    assert boundary["client_duration_seconds"] == result["duration"]
    assert boundary["synchronized_duration_seconds"] >= result["duration"]
    assert (
        boundary["synchronized_duration_seconds"]
        == manifest["measurement"]["synchronized_seconds"]
    )
    assert [c["path"] for c in boundary["calls"]] == [
        "/pause?mode=wait&clear_cache=false",
        "/collective_rpc",
        "/resume",
        "/pause?mode=wait&clear_cache=false",
        "/collective_rpc",
    ]
    for call in (boundary["calls"][1], boundary["calls"][4]):
        assert {r["rank"] for r in call["response"]["results"]} == {0, 1}
        assert all(r["device_sync_completed"] for r in call["response"]["results"])
    assert manifest["mod_control"] == receipt["mod_control"]
    assert manifest["mod_control"]["kvmat"]["reused_tokens"] == 0
    assert manifest["m01_attainment"] is None
    assert manifest["effective_flops"] is None
    assert manifest["efficiency_percent"] is None
