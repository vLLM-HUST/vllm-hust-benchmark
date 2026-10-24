"""Project the bounded reference supplement; never promote partial efficiency."""

import hashlib
import json
import subprocess
from pathlib import Path

from build_midterm_publication import section, tables

ROOT = Path(__file__).resolve().parents[1]
REVISION = "dfd2311fd75b1ee055e225acae125d7ba9211698"  # pragma: allowlist secret
REPORT_DIR = "reports/vllm-hust-midterm-20261019"
REPORT = REPORT_DIR + "/中期测试报告.md"
SUPPLEMENT = REPORT_DIR + "/m01-recovery/storage-reference-supplement-20261024"
OUTPUT = ROOT / "website-publication/data/midterm-v078.json"


def pinned(path):
    return subprocess.check_output(["git", "show", f"{REVISION}:{path}"], cwd=ROOT)


def build(read=pinned):
    raw = read(REPORT)
    report = raw.decode()
    assert "报告版本：V0.78" in report
    data = json.loads(read("website-publication/data/midterm-v077.json"))
    audit = json.loads(read(REPORT_DIR + "/milestone-v15/reuse-audit.json"))
    supplement = json.loads(read(SUPPLEMENT + "/summary.json"))
    method = audit["method"]
    assert (
        supplement["reference_seconds"]
        is supplement["efficiency_percent"]
        is supplement["attainment"]
        is None
    )
    assert method["current"] == data["m01_method"]["current"]
    data.update(
        version="V0.78",
        source_commit=REVISION,
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{REVISION}/{REPORT}",
        report_sha256=hashlib.sha256(raw).hexdigest(),
        m01_method=method,
        milestone_contract=audit,
        reference_supplement=supplement,
    )
    zh = (
        "同工作量F=1812905151088034 FLOPs、双卡同步T=127.59352582134306秒，C=7.1042 TFLOPS/卡不变；开启原生详细统计及CPU计数，"
        "不是严格无插桩计时，全部成本保留。独立参考A/B/C/D四批完成281个原计划案例及18个必要字段修订，"
        "1402项数值检查通过；计时稳定性478组通过、60组未通过。唯一一次模型更新的32组留出验证中6组通过、26组未通过。"
        "三场景依赖图复算完成，部分必要耗时及几何适用性仍未闭合；T参、C参、η与50%判定保持空值，既有B1性能收益不变。"
    )
    en = (
        "Same-workload F=1812905151088034 FLOPs and synchronized T=127.59352582134306 seconds on two chips remain unchanged; C=7.1042 TFLOPS/chip. "
        "Native detailed statistics and CPU counters retain all costs; not strictly uninstrumented. "
        "Independent batches A/B/C/D completed 281 original cases and 18 necessary-field revisions, with 1402 numerical checks passing. "
        "Timing stability: 478 groups pass, 60 fail. The single model update passes 6 of 32 heldout groups; 26 fail. "
        "Three dependency scenarios were recomputed, but necessary timing and geometry applicability remain unresolved. "
        "Reference time, reference compute, eta and the 50% verdict remain null; prior B1 performance gains are unchanged."
    )
    sections = {s["id"]: s for s in data["sections"]}
    verdict = sections["m01-verdict"]
    _, marker, tail = verdict["scope"].partition(" 依据V1.5，")
    assert marker
    verdict["scope"] = zh + marker + tail
    _, marker, tail = verdict["scope_en"].partition(" V1.5:")
    assert marker
    verdict["scope_en"] = en + marker + tail
    verdict["tables"] = tables(section(report, "#### M01 本轮验收判定")) + tables(
        section(report, "#### 独立参考最小补测与适用性")
    )
    verdict["tables_en"][0]["rows"][2] = [
        "Independent reference",
        "Four batches completed; three acyclic dependency scenarios recomputed",
        "Timing and geometry qualification remain incomplete; no zero substitution",
    ]
    verdict["tables_en"].append(
        {
            "headers": [
                "Batch",
                "Completed cases",
                "Numerical checks",
                "Timing / scope",
            ],
            "rows": [
                [
                    "A: necessary host/device programs",
                    "60 + 18 revisions",
                    "300/300 pass",
                    "Active-field metadata measured; 42 local-offset groups fail stability",
                ],
                [
                    "B: small geometries and composite chains",
                    "22",
                    "88/88 pass",
                    "26 stable groups; 18 unstable; initialization failures retained",
                ],
                [
                    "C: preprocessing, projection, RoPE",
                    "90",
                    "360/360 pass",
                    "180 stable groups; 8 conditional cases covered by validated reuse",
                ],
                [
                    "D: convolution, attention, GDN",
                    "109",
                    "654/654 pass",
                    "218 stable groups; oracle-layout repair continued only unfinished cases",
                ],
            ],
        }
    )
    evidence = {
        "path": SUPPLEMENT + "/summary.json",
        "kind": "blob",
        "label": "四批补测与适用性 / Bounded supplement and applicability",
    }
    verdict["evidence"].insert(0, evidence)
    verdict["evidence"].insert(
        1,
        {
            "path": SUPPLEMENT + "/archive.json",
            "kind": "blob",
            "label": "原始归档与校验 / Raw archive and checksums",
        },
    )
    sections["milestone-v15"]["tables"][0]["rows"][0][2] = zh
    sections["milestone-v15"]["tables_en"][0]["rows"][0][2] = en
    remaining = sections["v15-remaining"]
    _, marker, tail = remaining["scope"].partition(" 成本采用")
    assert marker
    remaining["scope"] = zh + marker + tail
    _, marker, tail = remaining["scope_en"].partition(" Cost uses")
    assert marker
    remaining["scope_en"] = en + marker + tail
    remaining["evidence"].append(evidence)
    return data


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n")
