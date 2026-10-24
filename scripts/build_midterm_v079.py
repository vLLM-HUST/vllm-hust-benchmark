"""Publish own-compute pairing separately from unresolved whole-chain eta."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from build_midterm_publication import section, tables

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = "reports/vllm-hust-midterm-20261019"
REPORT = REPORT_DIR + "/中期测试报告.md"
PAIR = REPORT_DIR + "/m01-recovery/official018-paired-compute-20261024"
OUTPUT = ROOT / "website-publication/data/midterm-v079.json"


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    raw = read(REPORT)
    report = raw.decode()
    assert "报告版本：V0.79" in report
    data = json.loads(read("website-publication/data/midterm-v078.json"))
    audit = json.loads(read(REPORT_DIR + "/milestone-v15/reuse-audit.json"))
    pair = json.loads(read(PAIR + "/summary.json"))
    prior = json.loads(
        read(
            REPORT_DIR + "/m01-recovery/storage-reference-closed-20261024/summary.json"
        )
    )
    assert pair["eta_percent"] is pair["attainment"] is None
    b0, b1 = pair["rows"]["B0"], pair["rows"]["B1"]
    gain = (b1["C_TFLOPS_per_chip"] / b0["C_TFLOPS_per_chip"] - 1) * 100
    assert abs(gain - pair["C_gain_percent"]) < 1e-10
    zh = (
        f"官方vLLM 0.18/Ascend 0.18 B0与vLLM-HUST B1同轮F/T/C配对已闭合；"
        f"C分别为{b0['C_TFLOPS_per_chip']:.4f}和{b1['C_TFLOPS_per_chip']:.4f} TFLOPS/芯片，提升{gain:.4f}%。"
        "两臂均为双芯片、200请求、282265输入及51200输出token；全部计量成本保留，不是严格无插桩计时。"
        "新整链参考实现、校准与验证执行方法尚未冻结，T参/C参/η及90%判定不提供；C提升不是参考达成率。"
        "旧算子依赖图η=37.2302%–37.2724%按原方法保留，不替代新整链验收。"
    )
    en = (
        f"Official vLLM 0.18/Ascend 0.18 B0 and vLLM-HUST B1 own-run F/T/C are closed: "
        f"{b0['C_TFLOPS_per_chip']:.4f} and {b1['C_TFLOPS_per_chip']:.4f} TFLOPS/chip, a {gain:.4f}% compute gain. "
        "Both use two chips, 200 requests, 282265 input and 51200 output tokens. All accounting costs remain; not strictly uninstrumented. "
        "The new whole-chain reference implementation and calibration/validation protocol are not frozen. Reference time, compute, eta and 90% acceptance remain unavailable. "
        "Compute gain is not reference attainment. Prior operator-DAG eta of 37.2302%–37.2724% retains its original method, not new whole-chain acceptance."
    )
    data.update(
        version="V0.79",
        source_commit=revision,
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        report_sha256=hashlib.sha256(raw).hexdigest(),
        m01_method=audit["method"],
        milestone_contract=audit,
        paired_own_compute=pair,
        prior_operator_reference=prior,
    )
    sections = {s["id"]: s for s in data["sections"]}
    verdict = sections["m01-verdict"]
    verdict["scope"], verdict["scope_en"] = zh, en
    verdict["tables"] = tables(section(report, "#### M01 本轮验收判定")) + tables(
        section(report, "#### 独立参考最小补测与适用性")
    )
    verdict["tables_en"] = [
        {
            "headers": ["Metric", "Official B0", "vLLM-HUST B1"],
            "rows": [
                ["Effective FLOPs", str(b0["F"]), str(b1["F"])],
                ["Synchronized seconds", str(b0["T_seconds"]), str(b1["T_seconds"])],
                ["Participating chips", "2", "2"],
                [
                    "TFLOPS/chip",
                    str(b0["C_TFLOPS_per_chip"]),
                    str(b1["C_TFLOPS_per_chip"]),
                ],
                ["Completed / failed", "200 / 0", "200 / 0"],
                ["New reference time / compute / eta", "Not closed", "Not closed"],
            ],
        },
        {
            "headers": [
                "Prior operator-DAG scenario",
                "Reference seconds",
                "TFLOPS/chip",
                "Eta (%)",
                "Original 50% verdict",
            ],
            "rows": [
                [
                    str(s["seed"]),
                    str(s["reference_seconds"]),
                    str(s["reference_tflops_per_chip"]),
                    str(s["efficiency_percent"]),
                    "Not met",
                ]
                for s in prior["scenarios"]
            ],
        },
    ]
    evidence = {
        "path": PAIR + "/summary.json",
        "kind": "blob",
        "label": "官方基线与B1有效算力配对 / Official B0 and B1 own compute",
    }
    verdict["evidence"].insert(0, evidence)
    verdict["evidence"].append(
        {
            "path": REPORT_DIR
            + "/m01-recovery/storage-reference-closed-20261024/summary.json",
            "kind": "blob",
            "label": "旧算子依赖图参考 / Prior operator-DAG reference",
        }
    )
    sections["milestone-v15"]["tables"][0]["rows"][0][2] = zh
    sections["milestone-v15"]["tables_en"][0]["rows"][0][2] = en
    remaining = sections["v15-remaining"]
    for key, prefix, marker in (
        ("scope", zh, " 成本采用"),
        ("scope_en", en, " Cost uses"),
    ):
        _, found, tail = remaining[key].partition(marker)
        remaining[key] = prefix + (found + tail if found else "")
    remaining["evidence"].append(evidence)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    OUTPUT.write_text(
        json.dumps(build(args.revision), ensure_ascii=False, indent=2) + "\n"
    )
