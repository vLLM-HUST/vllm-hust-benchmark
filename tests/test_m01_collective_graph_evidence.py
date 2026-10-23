import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "reports/vllm-hust-midterm-20261019/m01-recovery/collective-graph-20261023"
)


def test_sealed_collective_population_and_lifecycle():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    archive = EVIDENCE / manifest["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"]
    assert archive.stat().st_size == manifest["bytes"]
    files = {}
    with tarfile.open(archive) as bundle:
        for member in bundle:
            assert not Path(member.name).is_absolute()
            assert ".." not in Path(member.name).parts
            assert member.isdir() or member.isfile()
            if member.isfile():
                assert member.name not in files
                files[member.name] = bundle.extractfile(member).read()
    assert {
        name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
    } == manifest["members"]
    prefix = "m01-collective-graph-r1/"
    for line in files[prefix + "SHA256SUMS"].decode().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(files[prefix + name]).hexdigest() == digest
    receipt = json.loads(files[prefix + "receipt.json"])
    assert receipt["worker_exit_code"] == 0
    assert receipt["sessions_drained"] and receipt["all_devices_released"]
    assert receipt["fallback"] is False
    plan = json.loads(files[prefix + "execution-plan.json"])
    expected = {
        (shape["kind"], shape["dtype"], shape["elements"], batch, rep, sample)
        for shape in plan["shapes"]
        for batch in [1, 8, 32]
        for rep in range(3)
        for sample in range(16)
    }
    assert len(plan["shapes"]) == 75
    assert len(expected) == 10800
    for rank in (0, 1):
        rows = [
            json.loads(line)
            for line in files[prefix + f"samples-{rank}.jsonl"].splitlines()
        ]
        assert len(rows) == 10800
        assert all(row["numerics_passed"] and row["rank"] == rank for row in rows)
        assert {
            (
                r["shape"]["kind"],
                r["shape"]["dtype"],
                r["shape"]["elements"],
                r["batch"],
                r["repetition"],
                r["sample"],
            )
            for r in rows
        } == expected
    audit = json.loads(
        files["m01-critical-path-review-20261023/collective-graph-audit-r1.json"]
    )
    assert audit["samples_per_rank"] == 10800
    assert audit["reference_selection"] is None
    assert audit["m01_efficiency_percent"] is None
    assert manifest["m01_attainment"] is None
