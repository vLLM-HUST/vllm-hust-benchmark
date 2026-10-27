import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vllm-hust-midterm-20261019"
EVIDENCE = REPORT / "m01-recovery/accounting-assessment-20261022"


def module(name):
    spec = importlib.util.spec_from_file_location(name, REPORT / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def assessment():
    return json.loads((EVIDENCE / "assessment.json").read_text())


def test_imported_audits_preserve_original_bytes_and_scope():
    manifest = json.loads((EVIDENCE / "archive.json").read_text())
    assert (
        manifest["archive"]["sha256"] == module("package_accounting_assessment").SHA256
    )
    assert len(manifest["selected_members"]) == 3
    for entry in manifest["selected_members"]:
        raw = (EVIDENCE / entry["name"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    result = assessment()
    assert not result["formal_b1_admission"]
    assert result["uninstrumented_request_only_attainment"] is None
    assert not result["prior_raw_evidence_modified"]
    assert sum(result["families"].values()) == result["basic_operations"]
    assert len(result["families"]) == 10


def test_resource_formula_uses_max_and_the_matching_envelope():
    result = assessment()
    resources = result["resource_terms"]
    tref = max(resources["resource_mean_seconds"].values())
    assert tref == result["empirical_reference_seconds"]
    assert tref == resources["resource_mean_seconds"]["communication"]
    window = json.loads((EVIDENCE / "window.json").read_text())
    assert result["measured_window_seconds"] == window["device_envelope_seconds"]
    expected = 100 * tref / window["device_envelope_seconds"]
    assert math.isclose(result["empirical_attainment_percent"], expected)
    assert result["empirical_attainment_percent"] < result["internal_threshold_percent"]
    assert not result["threshold_met"]
    assert result["verdict"] == "internal_threshold_not_met"
    assert resources["sensitivity_is_confidence_interval"] is False


def test_report_table_is_rendered_from_the_assessment():
    builder_spec = importlib.util.spec_from_file_location(
        "builder", ROOT / "scripts/build_midterm_publication.py"
    )
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    report = (REPORT / "中期测试报告.md").read_text()
    actual = builder.tables(builder.section(report, "#### 历史设备窗口经验参考核算"))
    expected = builder.tables(
        module("render_accounting_assessment").render(assessment()) + "\n"
    )
    assert actual == expected
    assert "未插桩请求阶段达成率仍不赋值" in report


def test_corrupt_archive_fails_before_output_creation(tmp_path):
    source = tmp_path / "wrong.tar.gz"
    source.write_bytes(b"invalid archive")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="digest mismatch"):
        module("package_accounting_assessment").package(source, output)
    assert not output.exists()
