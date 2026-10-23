import hashlib
import json
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = (
    ROOT
    / "reports/vllm-hust-midterm-20261019/m01-recovery/logits-collective-reference-20261021"
)


def load():
    index = json.loads((DIRECTORY / "archive.json").read_text())
    packed = DIRECTORY / index["archive"]
    assert hashlib.sha256(packed.read_bytes()).hexdigest() == index["archive_sha256"]
    with tarfile.open(packed) as archive:
        data = {m.name: archive.extractfile(m).read() for m in archive}
    assert data.keys() == index["members"].keys()
    for name, raw in data.items():
        assert hashlib.sha256(raw).hexdigest() == index["members"][name]["sha256"]
        assert len(raw) == index["members"][name]["bytes"]
    return index, data


def test_all_repetitions_and_medians_preserved():
    _, data = load()
    summary = json.loads(data["recovery-logits-collective-control-r1/summary.json"])
    assert len(summary["shapes"]) == 76
    assert len(summary["phases"]) == 228
    for index, shape in enumerate(summary["shapes"]):
        phases = [p for p in summary["phases"] if p["shape_index"] == index]
        assert {p["repetition"] for p in phases} == {0, 1, 2}
        assert shape["counter_qualified_repetitions"] == 3
        assert all(p["counter_qualified"] and p["peer_exact"] for p in phases)
        durations = [p["duration_seconds"] for p in phases]
        assert shape["seconds_per_512_calls"] == durations
        assert shape["median_seconds_per_call"] == statistics.median(durations) / 512
        assert shape["duration_sample_stdev_seconds"] == statistics.stdev(durations)


def test_real_exit_release_and_no_full_model_claim():
    index, data = load()
    receipt = json.loads(data["recovery-logits-collective-control-r1/receipt.json"])
    summary = json.loads(data["recovery-logits-collective-control-r1/summary.json"])
    assert receipt["probe_exit_code"] == receipt["cleanup_exit_code"] == 0
    assert receipt["all_devices_released"] and receipt["sampler_errors"] == []
    assert summary["counter_resets"] == {"4": 0, "5": 0}
    assert summary["model_effective_flops"] is summary["m01_attainment"] is None
    assert index["model_effective_flops"] is index["m01_attainment"] is None
    assert (
        "recovery-logits-collective-control-r1/counters.jsonl"
        in index["raw_archive"]["members"]
    )


def test_rank_numerics_plan_and_source_identity():
    _, data = load()
    receipt = json.loads(data["recovery-logits-collective-control-r1/receipt.json"])
    for name, digest in receipt["input_hashes"].items():
        assert hashlib.sha256(data[name]).hexdigest() == digest
    plan = data["recovery-logits-collective-plan-r1.json"]
    for rank in (0, 1):
        record = json.loads(data[f"recovery-logits-collective-r1/rank-{rank}.json"])
        assert record["grid_sha256"] == hashlib.sha256(plan).hexdigest()
        assert len(record["phases"]) == 228
        assert all(p["calls"] == 512 and p["numerics_passed"] for p in record["phases"])


def test_m01_limited_verdict_matches_retained_model_evidence():
    report_dir = ROOT / "reports/vllm-hust-midterm-20261019"
    folder = report_dir / "m01-recovery/model-runtime-20261021"
    index = json.loads((folder / "archive.json").read_text())
    packed = folder / index["archive"]
    assert hashlib.sha256(packed.read_bytes()).hexdigest() == index["archive_sha256"]
    with tarfile.open(packed) as archive:
        ledger = json.load(
            archive.extractfile("recovery_model_r7-accounted-families.json")
        )
        hardware = json.load(archive.extractfile("recovery_model_r7-hardware.json"))
    assert ledger["model_effective_flops"] is ledger["m01_attainment"] is None
    assert len(ledger["families"]) == 10
    assert (
        sum(f["global_basic_operations"] for f in ledger["families"])
        == ledger["accounted_basic_operation_subtotal"]
    )
    assert all(
        not r["capture_audit"]["all_production_streams_proven"]
        for r in hardware["ranks"]
    )
    report = (report_dir / "中期测试报告.md").read_text()
    historical = json.loads(
        (ROOT / "website-publication/data/midterm-v062.json").read_text()
    )
    verdict = next(s for s in historical["sections"] if s["id"] == "m01-verdict")
    original = json.dumps(verdict, ensure_ascii=False)
    assert f"{ledger['accounted_basic_operation_subtotal']:,}" in original
    assert f"{hardware['request_phase']['duration_seconds']:.3f}" in original
    assert "证据受限" in original
    assert "未通过" in original
    assert "30.2443%–30.2505%" in report
    assert "不是V1.5当前C、η或达标判定" in report
    assert "复算中，当前C、η与达标结论均未判定" in report
