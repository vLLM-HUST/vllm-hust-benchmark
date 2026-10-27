"""Project corrected quality independently of unresolved reference attainment."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = "reports/vllm-hust-midterm-20261019"
REPORT = REPORT_DIR + "/中期测试报告.md"
QUALITY = REPORT_DIR + "/m01-recovery/corrected-quality-20261025/summary.json"
OUTPUT = ROOT / "website-publication/data/midterm-v080.json"


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    raw = read(REPORT)
    assert "报告版本：V0.80" in raw.decode()
    data = json.loads(read("website-publication/data/midterm-v079.json"))
    quality = json.loads(read(QUALITY))
    assert quality["reference"]["eta_percent"] is None
    data.update(
        version="V0.80",
        source_commit=revision,
        report_sha256=hashlib.sha256(raw).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        corrected_quality=quality,
    )
    current = data["current_summary"]["quality"]
    current["b1"] = {
        "candidate_run": "m01-current-b1-standalone-quality-r1",
        "candidate_correct": 156,
        "candidate_failed": 0,
        "candidate_truncated": 20,
        "candidate_unparsed": 0,
        "accuracy_delta_percentage_points": (156 - 151) / 200 * 100,
        "candidate_summary_sha256": quality["evidence"]["B1_summary"],
        "original_controller_status": "failed",
        "derived_quality_and_controls": "passed",
    }
    current["evidence_path"] = QUALITY
    current["candidate_pool"].extend(
        [
            {
                "label": "Corrected KV reference",
                "correct": 143,
                "total": 200,
                "quality_passed": False,
            },
            {
                "label": "Current corrected vLLM-HUST combination",
                "correct": 156,
                "total": 200,
                "quality_passed": True,
            },
        ]
    )
    rows = data["paired_own_compute"]["rows"]
    c0, c1 = (rows[arm]["C_TFLOPS_per_chip"] for arm in ("B0", "B1"))
    gain = (c1 / c0 - 1) * 100
    zh = (
        "官方B0与B1同轮F/T/C配对已闭合。"
        "全部计量成本保留，不是严格无插桩计时。"
        "当前纠正B1质量156/200、失败0，原控制器failed保留；FULL图重放判据独立派生更正后质量及控制通过。"
        "纠正KV参考候选143/200未通过151/200门槛。官方B0质量151/200复用。"
        f"B0/B1的C分别为{c0:.4f}/{c1:.4f} TFLOPS/芯片，提升{gain:.4f}%。C提升不是参考达成率；独立整链校准和适用性验证尚未完成，T参/C参/η及90%判定不提供。"
        "参考可与候选共享代码，不能以已有B1耗时确定分母。"
    )
    en = (
        "Official B0 and B1 own-run F/T/C are closed. "
        "All accounting costs remain; not strictly uninstrumented. "
        "Current corrected B1 scored 156/200 with zero failures. The failed original controller receipt is preserved; "
        "quality and controls passed an independent derived correction of the FULL-graph replay predicate. "
        "Corrected KV scored 143/200 and failed the 151/200 gate; official B0 quality 151/200 is reused. "
        f"B0/B1 C is {c0:.4f}/{c1:.4f} TFLOPS/chip, a {gain:.4f}% gain. Compute gain is not reference attainment. Independent whole-chain calibration and applicability validation remain incomplete: "
        "reference time/compute, eta and 90% attainment are unavailable. Shared implementation is allowed, but existing B1 timing cannot set the denominator."
    )
    evidence = {
        "path": QUALITY,
        "kind": "blob",
        "label": "纠正实现质量 / Corrected implementation quality",
    }
    sections = {item["id"]: item for item in data["sections"]}
    for name in ("m01-verdict", "v15-remaining"):
        for key, prefix, marker in (
            ("scope", zh, " 成本采用"),
            ("scope_en", en, " Cost uses"),
        ):
            tail = sections[name][key].partition(marker)
            sections[name][key] = prefix + (
                tail[1] + tail[2] if name == "v15-remaining" and tail[1] else ""
            )
        sections[name]["evidence"].append(evidence)
    sections["milestone-v15"]["tables"][0]["rows"][0][2] = zh
    sections["milestone-v15"]["tables_en"][0]["rows"][0][2] = en
    data["sections"].insert(
        0,
        {
            "id": "corrected-quality",
            "title": "当前纠正实现质量 / Corrected implementation quality",
            "source_heading": "5.12 当前纠正实现质量",
            "title_en": "Corrected implementation quality",
            "scope": zh,
            "scope_en": en,
            "tables": [
                {
                    "headers": ["实现", "正确/总数", "失败", "截断", "质量判定"],
                    "rows": [
                        ["官方B0（复用）", "151/200", "0", "18", "对照"],
                        ["纠正KV参考候选", "143/200", "0", "22", "未通过"],
                        [
                            "当前纠正vLLM-HUST B1",
                            "156/200",
                            "0",
                            "20",
                            "独立派生审计通过",
                        ],
                    ],
                }
            ],
            "tables_en": [
                {
                    "headers": [
                        "Implementation",
                        "Correct/total",
                        "Failed",
                        "Truncated",
                        "Quality",
                    ],
                    "rows": [
                        ["Official B0 (reused)", "151/200", "0", "18", "Control"],
                        ["Corrected KV reference", "143/200", "0", "22", "Failed"],
                        [
                            "Current corrected vLLM-HUST B1",
                            "156/200",
                            "0",
                            "20",
                            "Independent derived audit passed",
                        ],
                    ],
                }
            ],
            "evidence": [evidence],
        },
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    OUTPUT.write_text(
        json.dumps(build(args.revision), ensure_ascii=False, indent=2) + "\n"
    )
