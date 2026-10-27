"""Render the bounded V1.5 reanalysis in Chinese and English."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "reports/vllm-hust-midterm-20261019"
AUDIT = BASE + "/milestone-v15/remaining-audit-r1.json"


def section(audit):
    cost = audit["cost"]["arms"]
    windows = [
        r
        for r in audit["swe_continuous_records"]["windows"]
        if r["window_seconds"] == [0, 3600]
    ]
    rows = []
    for arm in ("b0", "b1"):
        r = cost[arm]
        u = r["utilization"]["per_card_percent"]
        rows.append(
            [
                "开发栈对照 / Development control"
                if arm == "b0"
                else "B1 E24/R36 + KV + Microflow",
                f"{r['interval_seconds']:.6f}",
                str(r["output_tokens"]),
                f"{r['card_seconds_per_output_token']:.9f}",
                f"{u['4']:.2f}% / {u['5']:.2f}% / {u['6']:.2f}% / {u['7']:.2f}%",
                f"{r['yuan_per_million_coefficient_times_P_over_D']:.9f} × P/D",
            ]
        )
    coverage_zh = [
        [
            "MMLU-Pro",
            "固定 B0 200/200完成，18长度终止；三个组合各200题、22/23/25长度终止；全量12032题及其他方案另见规范矩阵",
            "长度预算终止不等于静默截断；连续区间及服务异常仍须关联",
        ],
        [
            "HLE-Verified",
            "既有Gold 668、Revision 1143题求解无传输失败；裁判32/40项未决；所有已测方案保留在索引",
            "文本/图像、求解/裁判分开；不能从评分汇总推定服务异常为零",
        ],
        [
            "SWE-bench-Pro",
            "三组各3600秒真实Agent记录，实际≥16K请求已复算，见下表",
            "0.23系列对照不是固定B0；窗口末128/128/122未返回、TA排空失败保留；缺匹配4K及逐请求TTFT/TPOT",
        ],
        [
            "FrontierScience",
            "组合完成160题求解，41长度终止、8未决评分；其他已测方案见规范矩阵",
            "受控中断与裁判恢复记录保留；质量完成不等于连续稳定性通过",
        ],
        [
            "TerminalBench 2.1",
            "89任务定义已有；当前组合的隔离执行与任务记录依赖#403",
            "任务定义不是执行结果；不以SWE或旧Terminal结果替代",
        ],
    ]
    coverage_en = [
        [
            "MMLU-Pro",
            "Fixed B0: 200/200 completed, 18 length stops; three candidates: 200 each, 22/23/25 length stops; full 12,032-case and other solutions retained in matrix",
            "Budget termination is not silent truncation; continuous intervals and service anomalies still need attribution",
        ],
        [
            "HLE-Verified",
            "Existing Gold 668 / Revision 1,143 solver cases: no transport failures; 32/40 unresolved judgments; all measured solutions indexed",
            "Separate text/image and solver/judge; score summaries do not establish absence of service anomalies",
        ],
        [
            "SWE-bench-Pro",
            "Three existing 3,600-second real-agent runs; actual >=16K coverage recalculated below",
            "0.23-series controls are not fixed B0; 128/128/122 outstanding requests and TA drain failure retained; matched 4K and request TTFT/TPOT remain open",
        ],
        [
            "FrontierScience",
            "Combination completed 160 solver cases; 41 length stops and 8 unresolved judgments; other measured solutions retained",
            "Controlled interruption and judge recovery retained; quality completion is not continuous-stability acceptance",
        ],
        [
            "TerminalBench 2.1",
            "89 task definitions available; current combination execution depends on isolated backend and records in #403",
            "Task definitions are not executions; no substitution with SWE or older Terminal results",
        ],
    ]
    swe_rows = [
        [
            r["arm"],
            str(r["normal_returns"]),
            str(r["actual_context_ge16k_returns"]),
            f"{r['delivered_output_tokens_per_second']:.3f}",
            f"{r['e2e_p95_seconds']:.3f} / {r['e2e_p99_seconds']:.3f}",
        ]
        for r in windows
    ]
    zh = "复用既有原件，不启动补测。M01已核验333265个目标语义位置、1723317104312320 FLOPs目标矩阵及1730876356085005 FLOPs已列目标分项；GDN prefill条件范围为29101726539840–38619850757760 FLOPs，不含完整草稿有效算量，不是完整F或效率。实际分块、有效草稿上下文及模型阶段依赖参考仍待闭合。"
    en = "Existing evidence only; no new measurements. M01: 333,265 target semantic positions, 1,723,317,104,312,320 target matrix FLOPs and 1,730,876,356,085,005 listed target FLOPs verified. Conditional GDN prefill range: 29,101,726,539,840–38,619,850,757,760 FLOPs. These are not complete effective F or efficiency; actual partitions, useful draft contexts and model dependency reference remain open."
    cost_zh = "成本采用回溯的首请求发起至末成功响应包络，与原benchmark duration相差毫秒，原数值保留。每臂200请求均在区间内、282265输入/51200输出，全部4张保留卡计费。等费用基础的资源系数下降22.601951%，卡4/5利用率却未提高，卡6/7为零。利用率按采样命令结束时间左保持、裁剪至同一区间；最大采样间隔对照1.068秒/B1 1.250秒，属回溯估计。P/D未绑定费用凭据，新输入/缓存命中输入拆分未闭合，且对照为0.28.1/0.25.1rc2开发栈，不作固定B0货币成本达标判定。图表的4元/卡时显示假设不是本场景综合月费依据。"
    cost_en = "Cost uses a retrospective first-start/last-success envelope, milliseconds different from the retained original benchmark duration. Each arm has 200 enclosed requests, 282,265 input and 51,200 output tokens; all four reserved cards count. Equal-fee resource coefficient falls 22.601951%, but card 4/5 utilization does not improve; cards 6/7 are zero. Utilization is left-held at sample-command end and clipped to the same interval; maximum gaps are 1.068/1.250 seconds (control/B1). This is a retrospective estimate. Fee P/month-days D and new/cache-hit input split remain unbound. The control is 0.28.1/0.25.1rc2, not fixed B0: no monetary acceptance. The chart's 4 CNY/card-hour display assumption is not a documented all-in fee for this run."
    stability_zh = "SWE按完成时间归窗，实际上下文为输入+输出token，分位数用最近秩，与原插值分位数分别保留。下表为全小时，逐600/1200/1800秒窗口在复算JSON内；混合上下文不是受控4K/16K配对，E2E不等于TTFT/TPOT。不能由正常返回摘录证明全部任务无OOM、死锁、非预期退出或静默截断。"
    stability_en = "SWE windows use completion time, actual input+output tokens and nearest-rank quantiles; original interpolated quantiles remain separate. Full-hour rows are below, with phase windows in the JSON. Mixed contexts are not a controlled 4K/16K pair, and E2E is not TTFT/TPOT. Normal-return extracts do not establish absence of OOM, deadlock, unexpected exits or silent truncation across all tasks."
    return {
        "id": "v15-remaining",
        "title": "V1.5 证据复算与剩余边界 / V1.5 evidence reanalysis and remaining limits",
        "scope": zh + " " + cost_zh + " " + stability_zh,
        "scope_en": en + " " + cost_en + " " + stability_en,
        "tables": [
            {
                "headers": ["任务", "已有完成与异常记录", "影响当前结论的边界"],
                "rows": coverage_zh,
            },
            {
                "headers": [
                    "SWE方案（非固定B0）",
                    "全小时正常返回",
                    "实际≥16K返回",
                    "交付输出token/s",
                    "E2E P95/P99秒（最近秩）",
                ],
                "rows": swe_rows,
            },
            {
                "headers": [
                    "历史成本方案",
                    "请求包络秒",
                    "成功输出token",
                    "卡秒/token",
                    "卡4/5/6/7 AI Core",
                    "元/百万输出token（符号式）",
                ],
                "rows": rows,
            },
        ],
        "tables_en": [
            {
                "headers": [
                    "Task",
                    "Existing completion/anomaly evidence",
                    "Decision-relevant limit",
                ],
                "rows": coverage_en,
            },
            {
                "headers": [
                    "SWE arm (not fixed B0)",
                    "Normal returns/hour",
                    "Actual >=16K returns",
                    "Delivered output tok/s",
                    "E2E P95/P99 s (nearest rank)",
                ],
                "rows": swe_rows,
            },
            {
                "headers": [
                    "Historical cost arm",
                    "Request envelope s",
                    "Successful output tokens",
                    "Card-s/token",
                    "Cards 4/5/6/7 AI Core",
                    "CNY/million output tokens (symbolic)",
                ],
                "rows": rows,
            },
        ],
        "source_heading": "V1.5 既有证据复算 / Existing-evidence reanalysis",
        "evidence": [
            {"path": p, "kind": "blob", "label": label}
            for p, label in [
                (AUDIT, "复算与原件路径 / Reanalysis and original paths"),
                (
                    BASE + "/m01-recovery/boundary-accounting-20261023/manifest.json",
                    "M01 原件清单 / M01 originals",
                ),
                (
                    BASE + "/m06-paired-20261021/archive.json",
                    "成本原件清单 / Cost originals",
                ),
            ]
        ],
    }


def markdown(s):
    lines = ["### V1.5 既有证据复算", "", s["scope"], ""]
    for t in s["tables"]:
        lines += [
            "| " + " | ".join(t["headers"]) + " |",
            "| " + " | ".join(["---"] * len(t["headers"])) + " |",
        ]
        lines += ["| " + " | ".join(row) + " |" for row in t["rows"]]
        lines += [""]
    lines += [
        "证据：[复算、原件路径及完整方案索引](milestone-v15/remaining-audit-r1.json)、[M01分项归档](m01-recovery/boundary-accounting-20261023/manifest.json)、[成本原件](m06-paired-20261021/archive.json)。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    audit = json.loads((ROOT / AUDIT).read_text())
    path = ROOT / BASE / "中期测试报告.md"
    text = path.read_text().replace("报告版本：V0.74", "报告版本：V0.75")
    heading = "### V1.5 既有证据复算"
    if heading not in text:
        pos = text.index("\n## ", text.index("### V1.5 当前方法与复用范围"))
        text = text[:pos] + "\n" + markdown(section(audit)) + text[pos:]
    path.write_text(text)
