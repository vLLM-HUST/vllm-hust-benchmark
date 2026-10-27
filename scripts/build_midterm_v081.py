"""Publish independently verified whole-chain reference and paired attainment."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = "reports/vllm-hust-midterm-20261019"
REPORT = REPORT_DIR + "/中期测试报告.md"
RESULT = REPORT_DIR + "/m01-recovery/wholechain-reference-20261025/summary.json"
OUTPUT = ROOT / "website-publication/data/midterm-v081.json"


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    raw = read(REPORT)
    assert "报告版本：V0.81" in raw.decode()
    data = json.loads(read("website-publication/data/midterm-v080.json"))
    result = json.loads(read(RESULT))
    assert (
        result["reference_qualified"] and result["all_frozen_applicability_checks_pass"]
    )
    cref = result["reference_C_TFLOPS_per_chip"]
    b0, b1 = result["B0"], result["B1"]
    for arm in (b0, b1):
        assert arm["eta_percent"] == 100 * arm["own_C"] / cref
        assert arm["T_reference_seconds"] == arm["ownF"] / (2 * cref * 1e12)
        assert arm["eta_target_met"] == (arm["eta_percent"] >= 90)
    data.update(
        version="V0.81",
        source_commit=revision,
        report_sha256=hashlib.sha256(raw).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        wholechain_reference=result,
    )
    zh = (
        f"独立整链参考与配对验收完成。C参={cref:.4f} TFLOPS/芯片；"
        f"官方0.18/0.18 B0：C={b0['own_C']:.4f}、T参={b0['T_reference_seconds']:.4f}秒、η={b0['eta_percent']:.4f}%，未达90%；"
        f"当前vLLM-HUST B1：C={b1['own_C']:.4f}、T参={b1['T_reference_seconds']:.4f}秒、η={b1['eta_percent']:.4f}%，达到90%。"
        "参考与B1共享代码，采用事前选定实现的三次独立C200实测能力中位数及不重叠H200验证，不是理论峰值或全局最优上界；未扣除计量开销，未以已有B1耗时设分母。"
        "质量复用B0 151/200、B1 156/200，固定子集点估计不代表全集或统计非劣。纠正KV参考候选143/200失败及原控制器失败回执保留。"
    )
    en = (
        f"Independent whole-chain reference and paired assessment are complete. Reference compute={cref:.4f} TFLOPS/chip. "
        f"Official 0.18/0.18 B0: C={b0['own_C']:.4f}, reference time={b0['T_reference_seconds']:.4f}s, eta={b0['eta_percent']:.4f}%, below 90%. "
        f"Current vLLM-HUST B1: C={b1['own_C']:.4f}, reference time={b1['T_reference_seconds']:.4f}s, eta={b1['eta_percent']:.4f}%, meets 90%. "
        "Reference shares code with B1; capacity is the median of three independently executed C200 windows, validated on nonoverlapping H200. It is not theoretical peak or a global optimum. "
        "Accounting overhead remains; existing B1 timing did not set the denominator. Reused quality: B0 151/200 and B1 156/200, fixed-subset point estimates only. Corrected KV 143/200 failure and original failed controller receipts are preserved."
    )
    evidence = dict(
        path=RESULT,
        kind="blob",
        label="整链参考与配对验收 / Whole-chain reference and paired assessment",
    )
    sections = {s["id"]: s for s in data["sections"]}
    for name in ("corrected-quality", "m01-verdict", "v15-remaining"):
        for key, text, marker in (
            ("scope", zh, " 成本采用"),
            ("scope_en", en, " Cost uses"),
        ):
            tail = sections[name][key].partition(marker)
            sections[name][key] = text + (
                tail[1] + tail[2] if name == "v15-remaining" and tail[1] else ""
            )
        sections[name]["evidence"].append(evidence)
    headers = [
        "方案",
        "C（TFLOPS/芯片）",
        "C参（TFLOPS/芯片）",
        "T参（秒）",
        "η",
        "90%判定",
    ]
    headers_en = [
        "Arm",
        "C (TFLOPS/chip)",
        "Reference C",
        "Reference T (s)",
        "Eta",
        "90% assessment",
    ]
    rows = [
        [
            name,
            f"{r['own_C']:.4f}",
            f"{cref:.4f}",
            f"{r['T_reference_seconds']:.4f}",
            f"{r['eta_percent']:.4f}%",
            "通过" if r["eta_target_met"] else "未通过",
        ]
        for name, r in (("B0", b0), ("B1", b1))
    ]
    rows_en = [
        row[:-1] + ["Passed" if arm["eta_target_met"] else "Not met"]
        for row, arm in zip(rows, (b0, b1))
    ]
    sections["m01-verdict"]["tables"] = [dict(headers=headers, rows=rows)]
    sections["m01-verdict"]["tables_en"] = [dict(headers=headers_en, rows=rows_en)]
    sections["milestone-v15"]["tables"][0]["rows"][0][2] = zh
    sections["milestone-v15"]["tables_en"][0]["rows"][0][2] = en
    sections["m01-verdict"]["source_heading"] = "6.1 独立整链参考与配对验收"
    data["current_summary"]["reference"] = result
    data["paired_own_compute"].update(
        wholechain_reference=result,
        status="own_F_T_C_and_independent_reference_closed",
        reference_method_status="independent_wholechain_reference_qualified",
        reference_seconds=b1["T_reference_seconds"],
        reference_tflops_per_chip=cref,
        eta_percent=b1["eta_percent"],
        attainment=b1["eta_target_met"],
        scalar_reference_fields_arm="B1",
        scope="Own-run compute and independent empirical whole-chain reference attainment; shared code disclosed. Not nominal chip peak or a task-quality score.",
    )
    for arm in ("B0", "B1"):
        data["paired_own_compute"]["rows"][arm].update(
            reference_seconds=result[arm]["T_reference_seconds"],
            reference_tflops_per_chip=cref,
            eta_percent=result[arm]["eta_percent"],
            attainment=result[arm]["eta_target_met"],
        )
    quality_reference = data["corrected_quality"]["reference"]
    data["corrected_quality"]["reference_at_quality_audit"] = quality_reference.copy()
    quality_reference.update(
        status="independent_wholechain_reference_qualified",
        eta_percent=b1["eta_percent"],
        C_ref_TFLOPS_per_chip=cref,
        T_ref_seconds=b1["T_reference_seconds"],
        attainment=b1["eta_target_met"],
        current_result_path=RESULT,
        scalar_reference_fields_arm="B1",
    )
    method = data["m01_method"]
    data["m01_method_at_v080"] = json.loads(json.dumps(method))
    method.update(
        status="independent_wholechain_reference_qualified",
        threshold_percent=90,
        equivalent_threshold="C >= 0.9 * C_reference",
        pending_evidence=[],
        paired_own_compute=data["paired_own_compute"],
    )
    method["current"].update(
        reference_seconds=b1["T_reference_seconds"],
        reference_effective_tflops_per_chip=cref,
        efficiency_percent=b1["eta_percent"],
        passed=b1["eta_target_met"],
    )
    method["requirements"]["timing"] = (
        "Same-run effective F and synchronized T retain all accounting costs; no estimated overhead subtraction."
    )
    method["requirements"]["timing_refinement"] = (
        "Official B0/B1 own F/T/C and independently calibrated whole-chain reference are closed. Not strictly uninstrumented."
    )
    method["requirements"]["reference"] = result["method"]
    method["whole_chain_method_revision"].update(
        status="execution_and_independent_verification_complete",
        freeze_sha256=result["method"]["freeze_sha256"],
    )
    method["wholechain_reference"] = result
    for key, text in (("tables", zh), ("tables_en", en)):
        for row in sections["v15-remaining"][key][0]["rows"]:
            if "M01" in str(row[0]) or "有效算力" in str(row[0]):
                row[-1] = text
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    OUTPUT.write_text(
        json.dumps(build(args.revision), ensure_ascii=False, indent=2) + "\n"
    )
