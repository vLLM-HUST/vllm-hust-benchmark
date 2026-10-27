"""Publish the frozen FrontierScience quality pair without changing M01."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告.md"
EVIDENCE = "reports/frontierscience-native018-paired-20261025"
OUTPUT = ROOT / "website-publication/data/midterm-v082.json"


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    raw = read(REPORT)
    assert "报告版本：V0.82" in raw.decode()
    data = json.loads(read("website-publication/data/midterm-v081.json"))
    pair = {
        arm: json.loads(read(f"{EVIDENCE}/{prefix}-summary.json"))["tracks"]
        for arm, prefix in (("B0", "b0"), ("B1", "paired-b1"))
    }
    rows = []
    for track in ("olympiad", "research"):
        b0, b1 = (pair[arm][track] for arm in ("B0", "B1"))
        assert b0["total"] == b1["total"]
        delta = 100 * (b1["successes"] - b0["successes"]) / b0["total"]
        rows.append(
            [
                track.title(),
                f"{b0['successes']} / {b0['total']}",
                f"{b1['successes']} / {b1['total']}",
                f"{b0['judge_unresolved']} / {b1['judge_unresolved']}",
                f"{delta:+.4f}",
            ]
        )
    pair["scope"] = "Frozen paired candidate; not a cross-configuration best-of result"
    pair["quality_threshold"] = None
    pair["quality_passed"] = None
    pair["source"] = EVIDENCE
    pair["package_sha256"] = (
        "3b27ce3d0f87cff3908b0b87d1e908eb1e6305af4120ed83114fa2ace9a3f2db"  # pragma: allowlist secret
    )
    data.update(
        version="V0.82",
        source_commit=revision,
        report_sha256=hashlib.sha256(raw).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        frontierscience_designated_pair=pair,
    )
    zh = "FrontierScience指定0.18/0.18 B0与原冻结组合完成160题质量配对。B0为66/100、3/60；B1为62/100、2/60，未决分别4/1与5/3。两个赛道均未观察到该组合质量提升，不宣称质量达标。原128份评分原样复用，仅补32份；替代裁判不是B0求解栈，未决仍留完整分母。"
    en = "The designated 0.18/0.18 B0 and frozen optimized candidate have completed 160-case FrontierScience quality evaluation. B0: 66/100 and 3/60; B1: 62/100 and 2/60. Unresolved grades: 4/1 and 5/3. No quality improvement was observed for this pair; no quality-pass claim is made. All 128 prior grades were retained, with only 32 missing grades added. The substitute judge is distinct from the B0 solver runtime; unresolved cases remain in full denominators."
    sections = {s["id"]: s for s in data["sections"]}
    inventory = sections["dataset-quality-inventory"]
    inventory["scope"] = (
        zh
        + " HLE已有完整优化方案可复用；Gold最好173/668与Revision最好320/1143来自不同部署，指定0.18图文基线仍在补测。"
    )
    inventory["scope_en"] = (
        en
        + " Existing HLE optimized results are reusable: best observed Gold 173/668 and Revision 320/1143 belong to different deployments. The designated 0.18 multimodal baseline remains in progress."
    )
    for row in inventory["tables"][0]["rows"]:
        row[0] += "（既有开发栈）"
        if "FrontierScience" in row[0]:
            row[-1] = "指定0.18配对见独立表"
    data["sections"].insert(
        data["sections"].index(inventory) + 1,
        {
            "id": "frontierscience-designated-pair",
            "title": "FrontierScience指定基线配对 / Designated baseline pair",
            "scope": zh,
            "scope_en": en,
            "tables": [
                {
                    "headers": [
                        "赛道",
                        "B0判对数",
                        "配对B1判对数",
                        "B0/B1未决",
                        "判对率差（百分点）",
                    ],
                    "rows": rows,
                }
            ],
            "tables_en": [
                {
                    "headers": [
                        "Track",
                        "B0 correct",
                        "Paired B1 correct",
                        "B0/B1 unresolved",
                        "Delta (percentage points)",
                    ],
                    "rows": rows,
                }
            ],
            "source_heading": "FrontierScience指定0.18与原冻结组合配对",
            "evidence": [
                {"path": f"{EVIDENCE}/{name}", "kind": "blob", "label": name}
                for name in (
                    "b0-summary.json",
                    "paired-b1-summary.json",
                    "b0-completed-audit.json",
                    "provenance.json",
                )
            ],
        },
    )
    for table in sections["v15-remaining"].get("tables", []):
        for row in table["rows"]:
            if "数据集覆盖" in str(row[0]):
                row[-1] = (
                    "FrontierScience指定0.18质量配对已完成且未观察到该组合提升；HLE指定0.18完整图文基线、SWE-bench-Pro及TerminalBench2.1完整任务通过率仍待补齐。"
                )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    OUTPUT.write_text(
        json.dumps(build(args.revision), ensure_ascii=False, indent=2) + "\n"
    )
