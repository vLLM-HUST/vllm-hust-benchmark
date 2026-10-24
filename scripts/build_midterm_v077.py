"""Project the merged same-run compute evidence without changing B1 selection."""

import hashlib
import json
import subprocess
from pathlib import Path

from build_midterm_publication import section, tables

ROOT = Path(__file__).resolve().parents[1]
REVISION = "54361519b325d4dee9814aa3f90ee21e6f12ee17"  # pragma: allowlist secret
REPORT_DIR = "reports/vllm-hust-midterm-20261019"
REPORT = REPORT_DIR + "/中期测试报告.md"
OUTPUT = ROOT / "website-publication/data/midterm-v077.json"


def pinned(path):
    return subprocess.check_output(["git", "show", f"{REVISION}:{path}"], cwd=ROOT)


def build(read=pinned):
    assert len(REVISION) == 40
    raw = read(REPORT)
    report = raw.decode()
    assert "报告版本：V0.77" in report
    data = json.loads(read("website-publication/data/midterm-v076.json"))
    audit = json.loads(read(REPORT_DIR + "/milestone-v15/reuse-audit.json"))
    method = audit["method"]
    current = method["current"]
    measured = json.loads(
        read(REPORT_DIR + "/m01-recovery/storage-timing-20261024/arithmetic.json")
    )
    assert current["effective_flops"] == measured["basic_operation_flops"]
    assert (
        current["effective_tflops_per_chip"]
        == measured["basic_operation_tflops_per_chip"]
    )
    assert current["efficiency_percent"] is current["passed"] is None
    data.update(
        version="V0.77",
        source_commit=REVISION,
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{REVISION}/{REPORT}",
        report_sha256=hashlib.sha256(raw).hexdigest(),
        m01_method=method,
        milestone_contract=audit,
    )
    sections = {s["id"]: s for s in data["sections"]}
    value = f"{current['effective_tflops_per_chip']:.4f}"
    seconds = current["matched_synchronized_seconds"]
    flops = current["effective_flops"]
    summary_zh = (
        f"本轮同工作量基础运算F={flops} FLOPs，双卡同步T={seconds}秒，"
        f"C={value} TFLOPS/卡。原生详细统计与CPU prefill计数开启，"
        "全部成本保留，不是严格无插桩计时；新服务、新缓存、200请求零失败。"
        "独立T参尚未完成，C参、η与达标结论未判定。"
    )
    summary_en = (
        f"Same-workload basic-operation F={flops} FLOPs; synchronized T={seconds} s "
        f"on two chips; C={value} TFLOPS/chip. Native detailed statistics and CPU "
        "prefill counters are enabled with all costs retained; not strictly "
        "uninstrumented. Fresh service/cache, 200 requests, zero failures. "
        "Independent reference time, reference compute, eta and attainment remain undetermined."
    )
    verdict = sections["m01-verdict"]
    verdict["scope"] = summary_zh + (
        " 依据V1.5，C=F/(k×T×10¹²)，C参=F/(k×T参×10¹²)，η=C/C参×100%=T参/T×100%；"
        "同一F和k，门槛50%。F采用加减乘及矩阵乘加约定，不给特殊函数任意FLOP权重。"
        "T参由独立标定和阶段依赖推导，不以资源时间全局最大值、实测耗时反向拟合或较快样本择优替代；"
        "参考不是芯片标称峰值。既有最佳B1吞吐、质量及0.18/0.18 B0身份不变。"
    )
    verdict["scope_en"] = summary_en + (
        " V1.5: C=F/(k*T*10^12); C_reference=F/(k*T_reference*10^12); "
        "eta=C/C_reference*100%=T_reference/T*100%, same F and k, threshold 50%. "
        "Basic arithmetic excludes arbitrary SFU equivalents. Reference timing uses "
        "independent calibration and stage dependencies, not a global resource-time maximum, "
        "reverse-fitting measured time or selecting faster samples; it is not vendor peak. "
        "Prior best B1 performance/quality and fixed B0 0.18/0.18 are unchanged."
    )
    verdict["tables"] = tables(section(report, "#### M01 本轮验收判定"))
    verdict["tables_en"] = [
        {
            "headers": ["Current check", "Measured evidence", "Status / scope"],
            "rows": [
                [
                    "Basic-operation F",
                    str(flops) + " FLOPs",
                    "Own ordered acceptance and prefill counts; no numerator from another run",
                ],
                [
                    "Synchronized T",
                    str(seconds) + " s; 200/0 requests",
                    "Host/native counters enabled; all costs retained",
                ],
                [
                    "Independent reference",
                    "Existing calibration retained; same-workload dependency review ongoing",
                    "No reference time or efficiency claim",
                ],
                [
                    "C and attainment",
                    value + " TFLOPS/chip",
                    "Compute measured; eta and 50% gate undetermined",
                ],
            ],
        }
    ]
    source = {
        "path": REPORT_DIR + "/m01-recovery/storage-timing-20261024/arithmetic.json",
        "kind": "blob",
        "label": "同轮算量与计时 / Same-run work and timing",
    }
    verdict["evidence"].insert(0, source)
    sections["milestone-v15"]["tables"][0]["rows"][0][2] = summary_zh
    sections["milestone-v15"]["tables_en"][0]["rows"][0][2] = summary_en
    sections["milestone-v15"]["scope"] = sections["milestone-v15"]["scope"].replace(
        "优先复用复算，不新增补测或完整生命周期成本要求。",
        "已有证据复用；本轮M01计数方法及成本单独披露，不新增完整生命周期成本要求。",
    )
    sections["milestone-v15"]["scope_en"] = sections["milestone-v15"][
        "scope_en"
    ].replace(
        "Reuse/recalculate first; no new measurement or full-lifecycle cost requirement.",
        "Existing evidence is reused; this M01 run discloses counting and costs. No additional full-lifecycle cost requirement.",
    )
    remaining = sections["v15-remaining"]
    _, marker, tail = remaining["scope"].partition(" 成本采用")
    assert marker
    remaining["scope"] = summary_zh + "其余既有记录复算如下。" + marker + tail
    _, marker, tail = remaining["scope_en"].partition(" Cost uses")
    assert marker
    remaining["scope_en"] = (
        summary_en + " Other retained reanalyses follow." + marker + tail
    )
    remaining["evidence"].append(source)
    data["current_summary"]["effective_compute"] = {
        **current,
        "run_id": measured["run_id"],
        "scope": method["current_measurement"],
    }
    return data


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n")
