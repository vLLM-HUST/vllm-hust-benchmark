"""Project completed HLE baseline and the single SWE Agent observation."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告.md"
HLE = "reports/hle-native018-completed-20261026"
SWE = "reports/swe-native018-single-agent-20261026"
OUTPUT = ROOT / "website-publication/data/midterm-v084.json"


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    report = read(REPORT)
    assert "报告版本：V0.84" in report.decode()
    data = json.loads(read("website-publication/data/midterm-v083.json"))
    hle = {
        track: json.loads(read(f"{HLE}/b0-{track}-summary.json"))["tracks"][track]
        for track in ("gold", "revision")
    }
    swe = json.loads(read(f"{SWE}/summary.json"))
    data.update(
        version="V0.84",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        hle_designated_baseline=hle,
        swe_single_agent=swe,
    )
    zh = (
        "HLE指定vLLM 0.18/Ascend 0.18完整图文基线1811题已执行全部求解与评分请求；"
        "Gold 157/668、Revision 316/1143，未决23/31，完整分母保留。"
        "SWE-bench-Pro五项资格通过，唯一Agent触及100步上限，独立官方评分0/1；"
        "单任务观测不是731题全量验收。"
    )
    en = (
        "The designated vLLM 0.18/Ascend 0.18 multimodal HLE baseline completed all solver and judge requests: "
        "Gold 157/668 and Revision 316/1143, with 23/31 unresolved grades retained in full denominators. "
        "SWE-bench-Pro passed five qualification calls; the sole Agent reached 100 steps and scored 0/1 "
        "under independent official grading. This is not full 731-task acceptance."
    )
    sections = {s["id"]: s for s in data["sections"]}
    inventory = sections["dataset-quality-inventory"]
    inventory["scope"] = (
        zh + " FrontierScience指定基线配对与M01结果保持不变。HLE各优化部署独立列示。"
    )
    inventory["scope_en"] = (
        en
        + " FrontierScience and M01 results remain unchanged. HLE optimized deployments remain separate."
    )

    def section(identifier, title, scope, scope_en, headers, headers_en, rows, source):
        return dict(
            id=identifier,
            title=title,
            scope=scope,
            scope_en=scope_en,
            source_heading=title,
            tables=[dict(headers=headers, rows=rows)],
            tables_en=[dict(headers=headers_en, rows=rows)],
            evidence=[dict(path=source, kind="blob", label="Evidence / 证据")],
        )

    hrows = [
        [
            track.title(),
            f"{v['successes']} / {v['total']}",
            str(v["unresolved"]),
            str(v["solver_truncated"]),
            str(v["judge_truncated"]),
        ]
        for track, v in hle.items()
    ]
    data["sections"].insert(
        data["sections"].index(inventory) + 1,
        section(
            "hle-designated-baseline",
            "HLE指定0.18图文基线 / Designated HLE baseline",
            "BF16、TP2、EP开启、上下文65536、APC关闭、FULL_DECODE_ONLY。原1811份回答与668份Gold评分复用；只续Revision评分。替代裁判、单次观测，未决不算成功，也不等同确认错误。原无法恢复的进程退出码保持null；接续流程与资源释放已核验。",
            "BF16, TP2, EP on, 65536 context, APC off, FULL_DECODE_ONLY. All 1811 answers and 668 Gold grades were reused; only Revision grading resumed. Substitute judge and single observation; unresolved grades are nonsuccesses, not confirmed wrong answers. Unrecoverable original exit codes remain null; continuation and resource release were verified.",
            ["赛道", "判对数 / 完整分母", "未决", "求解截断", "裁判截断"],
            [
                "Track",
                "Correct / full denominator",
                "Unresolved",
                "Solver truncated",
                "Judge truncated",
            ],
            hrows,
            f"{HLE}/b0-completed-audit.json",
        ),
    )
    data["sections"].insert(
        data["sections"].index(inventory) + 2,
        section(
            "swe-single-agent-observation",
            "SWE-bench-Pro单Agent实测 / Single-Agent observation",
            "指定0.18/0.18；独立服务、新任务容器和独立评分容器。100步/1200秒/max_tokens4096/零自动重试；步数耗尽后保留5782字节补丁。官方评分未解决，未决0，评分不调用模型。不得外推完整数据集通过率。",
            "Designated 0.18/0.18 with a separate service, fresh task container and independent grading container. Budget: 100 steps, 1200 seconds, max_tokens 4096, no automatic retries. The 5782-byte partial patch was retained after the step limit. Official grading: unresolved task, zero indeterminate grades, zero model calls during grading. No full-dataset extrapolation.",
            ["已执行 / 全集", "解决数", "模型交互", "Agent耗时（秒）", "退出原因"],
            [
                "Executed / dataset",
                "Resolved",
                "Model calls",
                "Agent seconds",
                "Exit reason",
            ],
            [
                [
                    f"{swe['executed_tasks']} / {swe['total_dataset_tasks']}",
                    str(swe["resolved"]),
                    str(swe["agent"]["model_calls"]),
                    f"{swe['agent']['elapsed_seconds']:.6f}",
                    swe["agent"]["status"],
                ]
            ],
            f"{SWE}/summary.json",
        ),
    )
    remaining = sections["v15-remaining"]
    additions = {
        "tables": {
            "HLE-Verified": "；新增指定0.18全图文基线：157/668、316/1143，未决23/31；全部求解/评分请求执行并释放服务",
            "SWE-bench-Pro": "；新增指定0.18单Agent：100步耗尽，官方评分0/1，未决0，非731题全量覆盖",
        },
        "tables_en": {
            "HLE-Verified": "; new designated 0.18 multimodal baseline: 157/668 and 316/1143, unresolved 23/31; all solver/judge requests executed and service released",
            "SWE-bench-Pro": "; new designated 0.18 single Agent: 100-step limit, official score 0/1, zero indeterminate grades; not full 731-task coverage",
        },
    }
    for key, by_dataset in additions.items():
        matched = set()
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] in by_dataset:
                    row[1] += by_dataset[row[0]]
                    matched.add(row[0])
        assert matched == set(by_dataset)
    remaining["evidence"] += [
        dict(
            path=f"{HLE}/b0-completed-audit.json",
            kind="blob",
            label="HLE指定基线 / Designated HLE baseline",
        ),
        dict(
            path=f"{SWE}/summary.json",
            kind="blob",
            label="SWE单Agent / Single SWE Agent",
        ),
    ]
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    OUTPUT.write_text(
        json.dumps(build(args.revision), ensure_ascii=False, indent=2) + "\n"
    )
