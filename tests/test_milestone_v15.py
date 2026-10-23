import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vllm-hust-midterm-20261019"


def test_v15_exact_extract_and_offline_reuse():
    spec = importlib.util.spec_from_file_location(
        "v15", ROOT / "scripts/build_milestone_v15_status.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = json.loads((REPORT / "milestone-v15/reuse-audit.json").read_text())
    assert actual == module.build()
    assert actual["method"] == json.loads(
        (REPORT / "milestone-v15/method-status.json").read_text()
    )
    assert actual["method"]["method_version"] == "V1.5"
    assert all(v is None for v in actual["method"]["current"].values())
    assert actual["method"]["fair_comparison"]["fixed_b0"] == {
        "vllm": "0.18",
        "vllm_ascend": "0.18",
    }
    assert actual["method"]["fair_comparison"]["b1_base_version_constraint"] is None
    assert actual["cost_method"]["lifecycle_is_current_gate"] is False
    assert actual["stability_method"]["six_probe_pass_is_task_stability_pass"] is False
    assert actual["new_measurements_started"] is False
    assert 258 < actual["recomputed"]["c16_fixed_b0"]["descriptive_gain_percent"] < 259
    assert (
        22.60
        < actual["recomputed"]["historical_development_pair_resource_cost"][
            "reduction_percent"
        ]
        < 22.61
    )
    # Independent handoff hash: formatting must not normalize original CRLF.
    assert (
        hashlib.sha256(
            (REPORT / "milestone-v15/project4-v1.5.txt").read_bytes()
        ).hexdigest()
        == (
            "7f9352a85ec8025731b37b57362d80bfc48f8204ae9d1877eba521377267fa24"  # pragma: allowlist secret
        )
    )
    report = (REPORT / "中期测试报告.md").read_text()
    assert "报告版本：V0.74" in report
    assert "V1.5第3.4.4节按测量区间分摊费用" in report
    assert "M06全流程验收未通过" not in report
    assert "已完成75种通信图参考" in report
