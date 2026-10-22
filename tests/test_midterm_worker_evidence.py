import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "reports/vllm-hust-midterm-20261019"
EVIDENCE = ROOT / "m01-recovery/worker-model-20261022"


def test_worker_receipts_are_pinned_and_not_formal_b1():
    manifest = json.loads((EVIDENCE / "archive.json").read_text())
    assert len(manifest["archives"]) == 11
    assert len({row["name"] for row in manifest["archives"]}) == 11
    for row in manifest["selected_members"]:
        assert (
            hashlib.sha256((EVIDENCE / row["name"]).read_bytes()).hexdigest()
            == row["sha256"]
        )
        assert row["archive"] in {a["name"] for a in manifest["archives"]}
    assert manifest["m01_attainment"] is None
    assert manifest["model_effective_flops"] is None
    assert not manifest["formal_b1_admission"]
    audit = json.loads((EVIDENCE / "m01-worker-r1-audit.json").read_text())
    assert (audit["completed"], audit["failed"], audit["output_tokens"]) == (
        200,
        0,
        51200,
    )
    assert {rank["rank"] for rank in audit["ranks"]} == {0, 1}
    assert all(
        rank["rows"]["TASK"]["fully_inside_envelope"] > 0 for rank in audit["ranks"]
    )
    assert audit["all_devices_released"]
    ledger = json.loads(
        (EVIDENCE / "recovery_model_worker_r1-accounted-families.json").read_text()
    )
    assert len(ledger["families"]) == 10
    assert (
        sum(f["global_basic_operations"] for f in ledger["families"])
        == ledger["accounted_basic_operation_subtotal"]
    )
    assert ledger["model_effective_flops"] is ledger["m01_attainment"] is None
    assert (
        f"{ledger['accounted_basic_operation_subtotal']:,}"
        in (ROOT / "中期测试报告.md").read_text()
    )


def test_worker_packager_rejects_corruption_before_writing(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "worker_package", ROOT / "package_worker_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ARCHIVES", {"broken": "0" * 64})
    (tmp_path / "broken-20261022.tar.gz").write_bytes(b"corrupt")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="digest mismatch"):
        module.package(tmp_path, output)
    assert not output.exists()
