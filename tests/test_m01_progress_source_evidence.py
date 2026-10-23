import hashlib
import json
from pathlib import Path
import tarfile


BASE = Path(__file__).resolve().parents[1] / (
    "reports/vllm-hust-midterm-20261019/m01-recovery/progress-source-20261023"
)


def test_device_originals_and_guard_result():
    manifest = json.loads((BASE / "manifest.json").read_text())
    for name, expected in manifest["archives"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == expected
    with tarfile.open(
        BASE / "m01-progress-source-device-20261023-r1.tar.gz"
    ) as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    prefix = "m01-progress-source-device-r1/"
    for line in files[prefix + "SHA256SUMS"].decode().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256(files[prefix + name]).hexdigest() == expected
    receipt = json.loads(files[prefix + "receipt.json"])
    assert receipt == dict(
        child_exit=0, all_devices_released=True, model_loaded=False, status="passed"
    )
    result = json.loads(files[prefix + "device-probe.json"])
    assert result["actual_model_dma_timing_proven"] is False
    assert result["mutation_observed"] is True
    assert result["guard_buffer_count"] == 2
    assert len(result["cases"]) == 6
    for case in result["cases"]:
        assert case["original_value_count"] == (32 if case["protected"] else 0)
        assert case["mutated_value_count"] == (0 if case["protected"] else 32)
    assert [row["expected"] for row in result["guard_cases"]] == [
        100,
        99,
        98,
        97,
        96,
        95,
    ]
    for row in result["guard_cases"]:
        assert row["values"] == [row["expected"]] * 32
    assert (
        hashlib.sha256(
            files["m01-critical-path-review-20261023/m01_progress_source_snapshot.py"]
        ).hexdigest()
        == result["guard_source_sha256"]
    )
    for key in (
        "effective_flops",
        "reference_seconds",
        "efficiency_percent",
        "model_acceptance",
    ):
        assert manifest[key] is None


def test_prepared_plan_is_not_a_measurement():
    manifest = json.loads((BASE / "manifest.json").read_text())
    with tarfile.open(
        BASE / "m01-progress-protection-integration-20261023-r1.tar.gz"
    ) as archive:
        raw = archive.extractfile("capture-execution-plan-r11.json").read()
    assert hashlib.sha256(raw).hexdigest() == manifest["prepared_model_plan"]["sha256"]
    plan = json.loads(raw)
    assert plan["run_id"] == manifest["prepared_model_plan"]["run_id"]
    assert manifest["prepared_model_plan"]["executed"] is False
    assert plan["protect_progress_source"] is True
    assert plan["implementation_variant"] == "diagnostic-progress-source-protected/v1"
    assert plan["performance_acceptance"] is False
