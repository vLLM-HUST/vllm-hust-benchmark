"""Verify archived dataset measurements independently of accelerator libraries."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import mdformat
import pytest

REPORT = Path(__file__).resolve().parents[1] / "reports/vllm-hust-midterm-20261019"
spec = importlib.util.spec_from_file_location(
    "aligned_verify", REPORT / "verify_aligned_pair.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def evidence(tmp_path):
    for arm in ("common", "b0", "b1"):
        with tarfile.open(REPORT / f"aligned-mmlu-{arm}.tar.gz") as archive:
            archive.extractall(tmp_path, filter="data")
    return tmp_path / "20261019-midterm-aligned"


def test_pair_recomputes(evidence):
    actual = verifier.compare(evidence, "mmlu-b0-r2", "mmlu-b1-r2")
    expected = json.loads((REPORT / "aligned-mmlu-results.json").read_text())
    assert actual == expected
    for arm in ("b0", "b1"):
        assert actual[arm]["requests"] == 200
        assert actual[arm]["failures"] == 0
        assert actual[arm]["output_tokens"] == 51200
        assert actual[arm]["input_tokens"] == 282265
    assert actual["output_throughput_change_percent"] == pytest.approx(
        100 * (actual["b1"]["output_tps"] / actual["b0"]["output_tps"] - 1)
    )
    assert actual["p99_tpot_change_percent"] > 0
    assert actual["lifecycle_resource_cost_change_percent"] > 0
    assert not actual["accuracy_evaluated"]


def test_source_mutation_rejected(evidence):
    path = evidence / "measure_pair.py"
    path.write_text(path.read_text() + "\n# altered\n")
    with pytest.raises(AssertionError, match="measure_pair.py"):
        verifier.verify(evidence, "mmlu-b0-r2")


@pytest.mark.parametrize(
    "field,value",
    [
        ("bench_exit_code", 1),
        ("all_devices_released", False),
        ("child_exit_codes", [0, 1]),
    ],
)
def test_invalid_lifecycle_rejected(evidence, field, value):
    path = evidence / "mmlu-b1-r2/receipt.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        verifier.verify(evidence, "mmlu-b1-r2")


def test_inconsistent_tokens_rejected(evidence):
    path = evidence / "mmlu-b1-r2/mmlu-pro.json"
    data = json.loads(path.read_text())
    data["input_lens"][0] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        verifier.compare(evidence, "mmlu-b0-r2", "mmlu-b1-r2")


def test_report_generated_from_results():
    spec = importlib.util.spec_from_file_location(
        "aligned_render", REPORT / "render_aligned_section.py"
    )
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    result = json.loads((REPORT / "aligned-mmlu-results.json").read_text())
    text = renderer.render(
        result, json.loads((REPORT / "aligned-runtime.json").read_text())
    )
    assert f"{result['output_throughput_change_percent']:.2f}%" in text
    assert f"{result['p99_tpot_change_percent']:.2f}%" in text
    assert f"{result['lifecycle_resource_cost_change_percent']:.2f}%" in text
    report = (REPORT / "中期测试报告.md").read_text()
    section = report.split("<!-- BEGIN ALIGNED_MMLU -->")[1].split(
        "<!-- END ALIGNED_MMLU -->"
    )[0]
    # The archived renderer remains byte-identical; migrate only its display label.
    expected = mdformat.text(
        text.replace("B0", "开发栈对照"), options={"wrap": 100}, extensions={"gfm"}
    )
    assert section.strip() == expected.strip()


def test_manifest(evidence):
    manifest = json.loads((evidence / "pair-archive-manifest.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == expected, (
            name
        )


def test_profiler_coverage_is_not_attainment(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "profile_audit", REPORT / "audit_aligned_profile.py"
    )
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    path = tmp_path / "kernel_details.csv"
    path.write_text('Type,Input Shapes\nMatMulV2,"64;256"\nCustom,N/A\n')
    result = auditor.audit(tmp_path)
    assert result["files"][0]["kernel_rows"] == 2
    assert result["files"][0]["rows_with_input_shapes"] == 1
    assert result["files"][0]["rows_with_memory_counters"] == 0
    assert result["whole_model_effective_flops"] is None
    assert result["attainment_percent"] is None


@pytest.mark.parametrize("group", ["control", "probes"])
def test_profile_archive_integrity(tmp_path, group):
    name = f"aligned-profile-{group}"
    with tarfile.open(REPORT / f"{name}.tar.gz") as archive:
        archive.extractall(tmp_path, filter="data")
    root = tmp_path / "20261019-midterm-aligned"
    manifest = json.loads((root / f"{name}-manifest.json").read_text())
    for path, expected in manifest.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
    if group == "control":
        index = json.loads((root / "aligned-profile-trace-index.json").read_text())
        coverage = json.loads((REPORT / "aligned-profile-coverage.json").read_text())
        for item in coverage["files"]:
            assert item["sha256"] == index[item["path"]]["sha256"]
        for run in ("mmlu-profile-r1", "mmlu-profile-r2"):
            receipt = json.loads((root / run / "receipt.json").read_text())
            assert receipt["all_devices_released"]
            assert receipt["status"] == "failed"
            assert not receipt["performance_measurement"]
        original = root / "mmlu-profile-r1/launch_profile.py"
        receipt = json.loads((root / "mmlu-profile-r1/receipt.json").read_text())
        assert (
            hashlib.sha256(original.read_bytes()).hexdigest()
            == receipt["input_hashes"]["launch_profile.py"]
        )


def test_full_profile_coverage_keeps_unknown_metrics():
    result = json.loads((REPORT / "aligned-profile-coverage.json").read_text())
    assert len(result["files"]) == 2
    for item in result["files"]:
        assert item["kernel_rows"] > 0
        assert (
            item["rows_with_input_shapes"]
            + sum(item["missing_shapes_by_type"].values())
            == item["kernel_rows"]
        )
        assert (
            item["rows_with_memory_counters"]
            + sum(item["missing_memory_counters_by_type"].values())
            == item["kernel_rows"]
        )
    assert result["whole_model_effective_flops"] is None
    assert result["physical_hbm_bytes"] is None
    assert result["attainment_percent"] is None
