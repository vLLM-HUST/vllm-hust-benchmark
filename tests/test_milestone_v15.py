import hashlib
import importlib.util
import json
import subprocess
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
    current = actual["method"]["current"]
    assert current["effective_flops"] == 1812905151088034
    assert current["participating_chips"] == 2
    assert current["matched_uninstrumented_seconds"] is None
    assert current["matched_synchronized_seconds"] == 127.59352582134306
    assert current["effective_tflops_per_chip"] == 7.1042207644080255
    assert current["reference_seconds"] is None
    assert current["reference_effective_tflops_per_chip"] is None
    assert current["efficiency_percent"] is None
    assert current["passed"] is None
    prior = actual["method"]["reference_closure"]
    assert prior["efficiency_percent"] == 37.2302188966015
    assert prior["attainment"] is False
    pair = actual["method"]["paired_own_compute"]
    assert pair["rows"]["B0"]["F"] == 1829161955258989
    assert 84.97 < pair["C_gain_percent"] < 84.98
    assert pair["eta_percent"] is None
    assert actual["method"]["current_measurement"]["strict_uninstrumented"] is False
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
    assert "报告版本：V0.88" in report
    assert "V1.5第3.4.4节按测量区间分摊费用" in report
    assert "M06全流程验收未通过" not in report
    assert "官方0.18/0.18 B0与vLLM-HUST B1的同轮F、T、C已闭合" in report


def test_v074_publication_identity_and_bilingual_current_scope(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v074 as builder

    published = json.loads(builder.OUTPUT.read_text())
    assert published["version"] == "V0.74"
    assert published["source_commit"] == builder.REVISION
    exists = (
        subprocess.run(
            ["git", "cat-file", "-e", f"{builder.REVISION}:{builder.REPORT}"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    if exists:
        assert published == builder.build()
        assert (
            published["report_sha256"]
            == hashlib.sha256(builder.pinned(builder.REPORT)).hexdigest()
        )
    assert published["m01_method"]["method_version"] == "V1.5"
    sections = {s["id"]: s for s in published["sections"]}
    assert sections["milestone-v15"]["tables_en"][0]["rows"]
    assert "258.30%" in json.dumps(sections["milestone-v15"], ensure_ascii=False)
    for ident in (
        "milestone-v15",
        "m01-verdict",
        "resources",
        "paired-lifecycle",
        "long-context-parser",
    ):
        assert sections[ident]["scope_en"]
    for item in published["sections"]:
        for evidence in item["evidence"]:
            assert (ROOT / evidence["path"]).exists()
    historical = json.loads(
        (ROOT / "website-publication/data/midterm-v072.json").read_text()
    )
    old_sections = {s["id"]: s for s in historical["sections"]}
    for ident in old_sections:
        if ident != "m01-verdict":
            assert sections[ident]["tables"] == old_sections[ident]["tables"]
    assert published["kv_audit"]["runs"] == historical["kv_audit"]["runs"]
