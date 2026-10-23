import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "reports/vllm-hust-midterm-20261019/m01-recovery/reference-boundary-20261023"
)


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def test_preserved_bytes_and_source_bindings():
    manifest = read("archive-manifest.json")
    assert manifest["m01_attainment"] is None
    for name, record in manifest["files"].items():
        data = (EVIDENCE / name).read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
    archive = EVIDENCE / manifest["archive"]["name"]
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        == manifest["archive"]["sha256"]
    )
    hashes = {}
    with tarfile.open(archive) as bundle:
        for member in bundle:
            assert member.isfile() and Path(member.name).name == member.name
            hashes[member.name] = hashlib.sha256(
                bundle.extractfile(member).read()
            ).hexdigest()
    assert hashes == manifest["source_files"]
    checked = set()
    for name in manifest["files"]:
        for path, digest in read(name).get("source_sha256", {}).items():
            if Path(path).name in hashes:
                assert digest == hashes[Path(path).name]
                checked.add(Path(path).name)
    assert {
        "recover_retained_draft_steps.py",
        "bound_draft_arithmetic.py",
        "join_prefix_reference.py",
        "audit_moe_schedule.py",
        "run_boundary_b1.py",
    } <= checked


def test_partial_derivations_never_become_attainment():
    for name in (
        "retained-draft-step-bounds-r1.json",
        "draft-arithmetic-sensitivity-r1.json",
        "prefix-reference-join-r1.json",
        "moe-schedule-constraints-r1.json",
    ):
        assert read(name)["efficiency_percent"] is None
    bounds = read("retained-draft-step-bounds-r1.json")
    assert (bounds["joint_total_lower"], bounds["joint_total_upper"]) == (32021, 32056)
    assert (
        bounds["first_retained_lower"] + bounds["second_retained_lower"]
        != bounds["joint_total_lower"]
    )
    assert read("prefix-reference-join-r1.json")["model_reference_seconds"] is None
    assert read("moe-schedule-constraints-r1.json")["reference_seconds"] is None
    plan = read("boundary-execution-plan-r6.json")
    assert plan["participating"] == [0, 1]
    assert plan["efficiency_percent"] is None
    assert read("boundary-b1-import-preflight-r3.json")["service_started"] is False
