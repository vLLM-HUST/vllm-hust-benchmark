"""Project a completed, same-response full-answer pair without replacing history."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from recompute_full_answer_joint import paired_metrics

ROOT = Path(__file__).resolve().parents[1]
BASE = "reports/vllm-hust-midterm-20261019"
NEW = "reports/full-answer-joint-20261030"
REPORT = BASE + "/中期测试报告.md"
OUTPUT = "website-publication/data/midterm-v103.json"


def compact(arm):
    return {key: value for key, value in arm.items() if key != "rows"}


def measurements(read):
    pair = json.loads(read(NEW + "/joint-summary.json"))
    recomputed = paired_metrics(pair)
    for key in (
        "paired_quality",
        "output_throughput_change_percent",
        "rental_cost_reduction_percent",
    ):
        if pair[key] != recomputed[key]:
            raise ValueError("Public pair does not match its scalar records: " + key)
    a, b = pair["arms"]
    pair["latency_reduction_percent"] = {
        key: (1 - b["timing"][key]["mean"] / a["timing"][key]["mean"]) * 100
        for key in ("ttft_ms", "tpot_ms")
    }
    pair["numeric_thresholds"] = {
        "throughput_at_least_10_percent": pair["output_throughput_change_percent"]
        >= 10,
        "mean_ttft_at_least_5_percent": pair["latency_reduction_percent"]["ttft_ms"]
        >= 5,
        "mean_tpot_at_least_5_percent": pair["latency_reduction_percent"]["tpot_ms"]
        >= 5,
        "rental_cost_at_least_10_percent": pair["rental_cost_reduction_percent"] >= 10,
    }
    pair["quality_observation"] = {
        "fixed_subset_correct_count_not_lower": b["correct"] >= a["correct"],
        "statistical_noninferiority_established": False,
        "scope": "Original scorer and all200 tasks; length stops and paired regressions retained. No post-hoc quality tolerance.",
    }
    pair["arms"] = [compact(arm) for arm in pair["arms"]]
    continuous = json.loads(read(NEW + "/continuous-summary.json"))
    for key in ("b0", "b1"):
        continuous[key] = compact(continuous[key])
        phases = continuous[key]["phases"]
        if {p["name"] for p in phases} != {
            "4096-warmup",
            "4096-formal",
            "16384-warmup",
            "16384-formal",
        } or any(p["failures"] != 0 for p in phases):
            raise ValueError("Completed continuous pair required")
    return pair, continuous


def section(pair, continuous):
    a, b = pair["arms"]

    def table(en):
        return dict(
            headers=[
                "Metric" if en else "指标",
                "B0 · 0.18/0.18",
                "B1 · 0.28.1/0.25.1rc2",
            ],
            rows=[
                [
                    "Correct / tasks" if en else "正确/题数",
                    f"{a['correct']}/200",
                    f"{b['correct']}/200",
                ],
                [
                    "Length stops" if en else "长度终止",
                    str(a["truncated"]),
                    str(b["truncated"]),
                ],
                [
                    "Output token/s",
                    f"{a['output_tokens_per_second']:.4f}",
                    f"{b['output_tokens_per_second']:.4f}",
                ],
                *[
                    [
                        label,
                        f"{a['timing'][key]['mean']:.4f}",
                        f"{b['timing'][key]['mean']:.4f}",
                    ]
                    for key, label in (
                        ("ttft_ms", "Mean TTFT (ms)"),
                        ("tpot_ms", "Mean TPOT (ms/token)"),
                    )
                ],
                [
                    "CNY / million output tokens" if en else "元/百万输出token",
                    f"{a['cost']['cny_per_million_output_tokens']:.4f}",
                    f"{b['cost']['cny_per_million_output_tokens']:.4f}",
                ],
            ],
        )

    def window_table(en):
        return dict(
            headers=[
                "Arm / phase" if en else "方案/阶段",
                "Requests" if en else "请求",
                "token/s",
                "TTFT P95/P99 (ms)",
                "TPOT P95/P99 (ms)",
                "Failures" if en else "失败",
            ],
            rows=[
                [
                    key.upper() + " " + p["name"],
                    str(p["requests"]),
                    f"{p['cohort_output_tokens_per_second']:.4f}",
                    f"{p['timing']['ttft_ms']['p95']:.2f} / {p['timing']['ttft_ms']['p99']:.2f}",
                    f"{p['timing']['tpot_ms']['p95']:.2f} / {p['timing']['tpot_ms']['p99']:.2f}",
                    str(p["failures"]),
                ]
                for key in ("b0", "b1")
                for p in continuous[key]["phases"]
                if p["name"].endswith("formal")
            ],
        )

    return dict(
        id="full-answer-joint",
        title="完整回答联合补测 / Full-answer joint measurement",
        source_heading="V1.03 · 2026-10-30 · MMLU-Pro fixed200 / out8192",
        scope="同一轮响应联合评分、计时、采样利用率和核算成本；固定200题、输出上限8192、EOS开启、同两张910B2。B1采用逐状态对象锁页合并分配、每rank 4GiB主机State预算。旧短输出结果与失败保留；新质量不回填旧M01计数轮。",
        scope_en="Quality, timing, sampled utilization and cost use the same responses: fixed200, output cap8192, EOS enabled, the same two910B2 cards. B1 packs pinned allocation within each state object and uses a4GiB/rank host-State budget. Prior short-output records and failures remain; new quality does not qualify the old M01 counted run.",
        tables=[table(False), window_table(False)],
        tables_en=[table(True), window_table(True)],
        evidence=[
            dict(path=NEW + "/" + name, kind="blob", label=label)
            for name, label in (
                ("README.md", "配置与结论 / Configuration and findings"),
                (
                    "joint-summary.json",
                    "逐题标量与复算 / Per-task scalars and recomputation",
                ),
                ("continuous-summary.json", "连续窗口 / Continuous windows"),
                (
                    "original-telemetry.tar.gz",
                    "原始采样与回执 / Original telemetry and receipts",
                ),
                ("retained-r3-failure.json", "原失败 / Retained failure"),
                ("attempt-history.json", "全部尝试回执 / All attempt receipts"),
                ("manifest.json", "文件校验 / File hashes"),
            )
        ],
    )


def build(revision, read):
    data = json.loads(read("website-publication/data/midterm-v102.json"))
    pair, continuous = measurements(read)
    data.update(
        version="V1.03",
        source_commit=revision,
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        report_sha256=hashlib.sha256(read(REPORT)).hexdigest(),
        full_answer_joint=pair,
        packed_continuous=continuous,
    )
    data["historical_current_summary_v102"] = copy.deepcopy(data["current_summary"])
    a, b = pair["arms"]
    data["current_summary"]["quality"] = dict(
        b0=dict(correct=a["correct"], total=200, truncated=a["truncated"]),
        b1=dict(candidate_correct=b["correct"], candidate_truncated=b["truncated"]),
        output_limit=8192,
        evidence_path=NEW + "/joint-summary.json",
        scope="Joint full-answer measurement; not the historical out256 M01 configuration.",
    )
    data["current_summary"]["performance"] = dict(
        b0=dict(
            run=a["run"],
            output_tokens_per_second=a["output_tokens_per_second"],
            timing=a["timing"],
        ),
        b0_status="fixed018_same_response_joint_measurement",
        output_limit=8192,
        requests=200,
        b1_metrics=[
            dict(
                metric_id=metric,
                value=value,
                baseline_value=baseline,
                selected_candidate_id=b["run"],
                provenance=dict(
                    repository="vLLM-HUST/vllm-hust-benchmark",
                    repository_commit=revision,
                    run_id=b["run"],
                    evidence_path=NEW + "/joint-summary.json",
                ),
            )
            for metric, value, baseline in (
                (
                    "output_token_throughput",
                    b["output_tokens_per_second"],
                    a["output_tokens_per_second"],
                ),
                (
                    "ttft_mean",
                    b["timing"]["ttft_ms"]["mean"],
                    a["timing"]["ttft_ms"]["mean"],
                ),
                (
                    "tpot_mean",
                    b["timing"]["tpot_ms"]["mean"],
                    a["timing"]["tpot_ms"]["mean"],
                ),
            )
        ],
        evidence_path=NEW + "/joint-summary.json",
    )
    data["current_summary"]["b1_runtime"] = dict(
        vllm=b["runtime"]["vllm_version"],
        ascend=b["runtime"]["ascend_version"],
        host_state_bytes_per_rank=4 * 2**30,
    )
    data["sections"].insert(0, section(pair, continuous))
    historical = next(s for s in data["sections"] if s["id"] == "acceptance-completion")
    historical["title"] = (
        "V1.02补测与保留失败 / V1.02 measurements and retained failures"
    )
    next(s for s in data["sections"] if s["id"] == "corrected-quality")["title"] = (
        "纠正实现既有质量 / Prior corrected-implementation quality"
    )
    return data


def markdown_table(table):
    rows = [table["headers"], ["---"] * len(table["headers"]), *table["rows"]]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def evidence_readme(pair, continuous):
    a, b = pair["arms"]
    s = section(pair, continuous)
    paired = pair["paired_quality"]
    lines = []
    for arm in (a, b):
        normal = arm["quality_by_finish"]["stop"]
        length = arm["quality_by_finish"]["length"]
        lines.append(
            f"| {arm['role']} | {normal['correct']}/{normal['total']} | {length['correct']}/{length['total']} | "
            f"{arm['successful_output_tokens']} | {arm['phase_seconds']:.6f} | {arm['utilization']['mean_percent']:.4f}% |"
        )
    quality = "未下降" if b["correct"] >= a["correct"] else "下降，未通过质量比较"
    return f"""# 完整回答联合测量与连续运行复核

日期：2026-10-30。固定B0：vLLM 0.18.0 / vLLM-Ascend 0.18.0；优化候选：
vLLM {b["runtime"]["vllm_version"]} / Ascend {b["runtime"]["ascend_version"]}。

## 本轮结果

{markdown_table(s["tables"][0])}

固定200题子集正确数{quality}。逐题{paired["improved"]}题改善、{paired["regressed"]}题退步，差值{paired["difference_percentage_points"]:+.2f}个百分点；全部响应和原评分器fallback保留。长度终止不隐去，也不事后改判。
吞吐变化{pair["output_throughput_change_percent"]:+.4f}%，平均TTFT/TPOT分别改善{pair["latency_reduction_percent"]["ttft_ms"]:.4f}%/{pair["latency_reduction_percent"]["tpot_ms"]:.4f}%，每百万输出token租赁估算降低{pair["rental_cost_reduction_percent"]:.4f}%。

| 方案 | 正常结束正确/总数 | 长度终止正确/总数 | 完成输出token | 请求区间秒数 | 双卡AI Core采样均值 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(lines)}

这是一轮固定子集配对，不代表全集质量、统计非劣或所有负载。新完整回答质量和计时不回填旧out256的M01算量。

## 输入、资源与计量

- 原MMLU-Pro固定200题顺序及物化请求不变，服务端分词哈希全部核对；输出上限8192、temperature=0、seed=0、EOS开启。
- 同Qwen3.5-35B-A3B源权重、BF16、TP2、同物理5/6两张910B2，双方计费资源均为两卡。
- 同时提交200请求，无额外任务预热，服务模型加载和图捕获完成后开始。请求总超时及读超时均6600秒；不事后增加交互式SLO或质量容差。
- 两侧max-model-len=65536、max-num-seqs=24、max-num-batched-tokens=4096、MTP2、设备KV预算20GiB/rank。B0为FULL_AND_PIECEWISE、CANN8.5.1及原text-only兼容补丁；B1为FULL、CANN9.1.0、E24/R36、逐请求重算及Microflow组合。
- B1将同一状态对象内同dtype lane合并为一块锁页分配，不跨对象共享；主机State预算4GiB/rank。8GiB/rank失败记录保留，不将更低预算当成原配置。
- 每臂独立服务、新空运行/编译缓存、逐请求唯一salt；只读模型及OS文件页缓存共享。连续资格结束后先释放服务和卡，再启动联合轮，不导入前轮答案或状态。
- B0完成后，闲置B0运行时已移存并逐文件核验后删除本地副本；这会改变OS页缓存环境，与4GiB预算变化共同披露，不单独归因于预算。不声称全局冷缓存一致。
- 吞吐分母为首次提交至全部响应完成的区间；同一批成功响应的全部输出计入分母，包括答错和长度终止。TTFT到首非空正文，TPOT=(响应结束−首正文)/(输出token−1)，逐请求等权；P95/P99采用最近秩法。HTTP/SSE采集开销保留。
- AI Core利用率按同轮单调时钟中点、左保持采样作时长加权；逐卡及观测空档见JSON。采样均值不是物理精确值或置信区间。

## 连续窗口

{markdown_table(s["tables"][1])}

同原4115/16407实际输入token，C8闭环、out256，分别60秒预热、300秒正式发起并完整排空。
B0复用已封存原件；B1预热24/40、正式408/183，共655请求无失败，独立准入655、reused_tokens=0，两个rank实际活动均经核验。16K尾时延高于4K，完整分窗口记录保留；零失败不等于五任务全范围稳定性通过。

## 成本依据与范围

沿用[天翼云类似双卡910B报价](https://www.ctyun.cn/document/10029787/10047957)：pak3.8xlarge.8，36908元/月；[计费范围](https://www.ctyun.cn/document/10029787/10030140)。来源核验日期2026-10-29，30天为分析约定。
此报价包含对应SKU的CPU/主机内存/加速卡，云盘和公网另计；报价主机256GB内存不等同本机64GiB容器，本结果为类似机型同费率估算，不是本机账单。
费用=每卡月费×实际卡数×测量秒数/(30×24×3600)；单位成本=费用×10⁶/完成输出token。固定费用不按利用率打折。
另列包含初始化、预检及退出观测的完整控制器生命周期成本：B0 {a["cost"]["controller_lifecycle_cny"]:.6f}元、B1 {b["cost"]["controller_lifecycle_cny"]:.6f}元；不与请求区间混合，也不把生命周期差异用作新增否决条件。

## 原件、复算与历史

- [逐题标量及配对汇总](joint-summary.json)：完整200×2标量、每题原SSE哈希、实际版本、配置、双卡采样、报价与公式。公开标量由独立原SSE重建/原评分器复核产生，不包含题文、gold或全文回答。
- [连续逐请求/窗口统计](continuous-summary.json)：包括原B0、当前B1、两rank激活和P95/P99。
- [未改写的原始采样与回执](original-telemetry.tar.gz)、[成员校验](telemetry-manifest.json)、[文件校验](manifest.json)。这些是选定原件，不是完整私有响应包。
- [8GiB候选原失败](retained-r3-failure.json)：24预热成功、正式16成功8失败，rank1的32MiB锁页分配207001；16K未执行。关停后的EngineDeadError不替代原始错误。
- [本批次全部尝试](attempt-history.json)：保留r1监控保护中断、r2模型启动前端口拒绝、r3驱动分配失败和r4新配置；没有从多轮完整回答结果中择优。
- [V1.02原连续失败和out256评分](../acceptance-completion-20261029/README.md)。原21/200对19/200不能借本轮完整回答评分。
- 私有完整响应、输入及实现源码保存在授权存储；不公开其内容。冻结源码包、原响应和控制回执均有SHA绑定；公开部分可按下列命令复算，原内容复核须通过私有保管入口。

```sh
python scripts/recompute_full_answer_joint.py reports/full-answer-joint-20261030/joint-summary.json
```

## English summary

This is one same-response MMLU-Pro fixed200/full-answer pair, capped at8192 tokens with EOS enabled. Fixed B0 uses0.18/0.18; the candidate uses the disclosed0.28.1/0.25.1rc2 development stack, per-object pinned packing and a4GiB/rank host-State budget. Both reserve the same two910B2 cards. Quality, latency, utilization and cost come from the same responses, with all length stops and per-task regressions retained.
The4K/16K continuous formal cohorts completed153/73 requests for reused B0 and408/183 for the new candidate, with zero failures. This does not establish five-task stability. Prior allocation failures remain separately archived. Pricing is a public analogous-machine estimate, not an invoice. New full-answer measurements do not supply missing F/T/reference evidence for the historical M01 workload.
"""


def prepare():
    """Generate current prose from audited scalars, retaining the exact V1.02 report."""

    def read(path):
        return (ROOT / path).read_bytes()

    pair, continuous = measurements(read)
    (ROOT / NEW / "README.md").write_text(evidence_readme(pair, continuous))
    a, b = pair["arms"]
    added = section(pair, continuous)
    prior = ROOT / BASE / "中期测试报告-v102.md"
    if not prior.exists():
        prior.write_bytes(read(REPORT))
    report = prior.read_text()
    assert "报告版本：V1.02" in report
    report = report.replace(
        "报告版本：V1.02（固定B0连续补测、同预算质量与参考模型复核）",
        "报告版本：V1.03（固定B0完整回答联合测量与连续运行复核）",
        1,
    ).replace("编制日期：2026年10月29日", "编制日期：2026年10月30日", 1)
    formal = {
        key: [p for p in continuous[key]["phases"] if p["name"].endswith("formal")]
        for key in ("b0", "b1")
    }
    quality = (
        "本轮固定子集正确数未下降"
        if b["correct"] >= a["correct"]
        else "本轮固定子集正确数下降，该候选未通过质量比较"
    )
    summary = (
        f"**完整回答联合测量：B0 {a['correct']}/200、优化候选{b['correct']}/200，"
        f"长度终止分别{a['truncated']}/{b['truncated']}题，双方零请求失败；{quality}。"
        f"输出吞吐变化{pair['output_throughput_change_percent']:+.4f}%，"
        f"同费率租赁成本降低{pair['rental_cost_reduction_percent']:.4f}%。"
        "两侧4K/16K连续窗口均完成；本轮结果不回填旧M01计数工况。**"
    )
    paragraphs = report.split("\n\n")
    for index, paragraph in enumerate(paragraphs):
        if paragraph.startswith("**固定B0的4K/16K连续窗口已完成"):
            paragraphs[index] = summary
        elif paragraph.startswith("B0的正式服务窗口见3.3节"):
            paragraphs[index] = (
                f"本轮固定200题完整回答质量为B0 {a['correct']}/200、优化候选{b['correct']}/200，"
                "联合性能与成本见5.13节。5.11节的151/200等既有质量记录保留原配置与日期；"
                "3.3节另列既有服务窗口。其他版本的Native仅称版本对照，不称B0；开发栈对照收益不作为相对B0的收益。"
            )
    report = "\n\n".join(paragraphs)
    report = report.replace(
        "报告结论截至2026年10月26日（北京时间）",
        "报告结论截至2026年10月30日（北京时间）",
    )
    before, rest = report.split("### 当前证据与缺口清单", 1)
    _, after = rest.split("## 二、大纲指标对照表", 1)
    gaps = f"""### 当前证据与缺口清单

| 项目 | 当前结果 | 仍影响结论的条件 |
| --- | --- | --- |
| M01 | 同轮实测与独立标定模型诊断保留 | B0路径绑定、适用无插桩计时及原计数工况质量；不借用本轮8192回答 |
| 联合性能/成本 | B0 {a["correct"]}/200、候选{b["correct"]}/200；同轮计时、用量及双卡利用率已采齐 | {quality}；适用固定200题单次突发负载，报价是公开类似机型估算 |
| 连续运行 | B0 4K/16K分别{formal["b0"][0]["requests"]}/{formal["b0"][1]["requests"]}请求，B1分别{formal["b1"][0]["requests"]}/{formal["b1"][1]["requests"]}请求，零失败 | 合成连续窗口，不替代五任务全范围稳定性；旧分配失败保留 |
| 五任务 | MMLU本轮评分完成；HLE、FrontierScience及Agent既有记录保留 | HLE原退出状态缺失；SWE/Terminal部分任务及失败未由本轮消除 |

依据：[联合补测](../full-answer-joint-20261030/README.md)、[逐题标量](../full-answer-joint-20261030/joint-summary.json)、[连续窗口](../full-answer-joint-20261030/continuous-summary.json)、[M01参考方法](../acceptance-completion-20261029/reference-method.md)。

"""
    report = before + gaps + "## 二、大纲指标对照表" + after
    # Only replace the current outline table; archived ALIGNED_MMLU stays byte-bound.
    start, remainder = report.split("## 二、大纲指标对照表", 1)
    current_table, remainder = remainder.split("## 三、测试范围与环境", 1)
    replacements = {
        "M02": f"| M02 | 输出吞吐 | 提升≥10% | 本轮{a['output_tokens_per_second']:.4f}→{b['output_tokens_per_second']:.4f} token/s，变化{pair['output_throughput_change_percent']:+.4f}% | {quality}；数值及适用负载见5.13节 |",
        "M03": f"| M03 | 平均首字时延 | 降低≥5% | 本轮{a['timing']['ttft_ms']['mean']:.4f}→{b['timing']['ttft_ms']['mean']:.4f} ms，降低{pair['latency_reduction_percent']['ttft_ms']:.4f}% | 同一200题突发负载；质量与完整尾时延另列 |",
        "M04": f"| M04 | 平均逐token时延 | 降低≥5% | 本轮{a['timing']['tpot_ms']['mean']:.4f}→{b['timing']['tpot_ms']['mean']:.4f} ms/token，降低{pair['latency_reduction_percent']['tpot_ms']:.4f}% | 同一200题突发负载；不外推交互式SLO |",
        "M05": "| M05 | 数据集与长上下文稳定性 | 五任务异常核查、≥16K连续及可比4K窗口 | 新4K/16K两臂完成、零失败，统计见7.5节 | 连续合成窗口已测；五任务剩余项保留 |",
        "M06": f"| M06 | 单位输出成本及利用率 | 成本降低≥10%，高负载利用率改善 | 本轮{a['cost']['cny_per_million_output_tokens']:.4f}→{b['cost']['cny_per_million_output_tokens']:.4f}元/百万输出token | 同轮质量及同期双卡利用率见8.4节；公开类似机型报价估算 |",
        "O01": "| O01 | 长上下文运行统计 | 报告连续窗口吞吐及TTFT/TPOT P95、P99 | 固定B0与当前4GiB/rank候选的4K/16K窗口，见7.5节 | 逐请求及分窗口统计完成 |",
        "O02": f"| O02 | 高负载设备利用率 | 报告可比工况同期AI Core平均利用率 | 本轮双卡均值{a['utilization']['mean_percent']:.4f}%→{b['utilization']['mean_percent']:.4f}%，见8.4节 | 同轮采样覆盖完整；设备忙碌率不等于M01有效算力 |",
    }
    lines = []
    for line in current_table.splitlines():
        key = line.split("|")[1].strip() if line.startswith("|") else ""
        lines.append(replacements.get(key, line))
    report = (
        start
        + "## 二、大纲指标对照表"
        + "\n".join(lines)
        + "\n\n## 三、测试范围与环境"
        + remainder
    )
    joint_section = f"""### 5.13 完整回答联合测量（2026-10-30）

{added["scope"]}
固定原200题顺序、temperature=0、seed=0，200请求一次突发、无额外任务预热，服务完成模型加载及图捕获后开始。评分器及fallback规则不变；长度终止照原结束原因单列，不改判分。

{markdown_table(added["tables"][0])}

{quality}。这是固定子集单次配对，不作全集或统计非劣结论；原逐题退步、截断与未解析记录可复算。请求统一超时6600秒，不事后增加交互时延门槛；批量完整回答时延不外推在线交互SLO。
配置、完整P95/P99、质量分组及生命周期成本见[本轮结果](../full-answer-joint-20261030/README.md)。

"""
    marker = next(line for line in report.splitlines() if line.startswith("## 六、"))
    report = report.replace(marker, joint_section + marker, 1)
    continuous_section = (
        "### 7.5 4GiB/rank候选的连续复核（2026-10-30）\n\n"
        + markdown_table(added["tables"][1])
        + "\n\n同原输入4115/16407 token、C8闭环、out256，分别60秒预热、300秒正式发起并完整排空。B0复用已完成原件；B1独立服务和新缓存，逐请求准入及双rank活动经审计。锁页分配改为单对象合并分配，主机State预算为4GiB/rank；移出闲置B0运行时释放OS文件页缓存的环境变化一并披露，不能仅归因于预算。旧失败见7.4节及新归档。\n\n"
    )
    report = report.replace(
        "## 八、单位输出成本与设备利用率",
        continuous_section + "## 八、单位输出成本与设备利用率",
        1,
    )
    cost_section = f"""### 8.4 完整回答的同轮成本与利用率

本轮沿用天翼云类似双卡910B的36908元/月公开报价、30天分析约定，云盘/公网另计；不是本机账单。成功完成响应的全部输出token计入分母，包含答错与长度终止，语义质量单独评分。
两侧每百万输出token租赁估算分别{a["cost"]["cny_per_million_output_tokens"]:.4f}/{b["cost"]["cny_per_million_output_tokens"]:.4f}元，降低{pair["rental_cost_reduction_percent"]:.4f}%。双卡同期时长加权AI Core采样均值分别{a["utilization"]["mean_percent"]:.4f}%/{b["utilization"]["mean_percent"]:.4f}%，采样覆盖与逐卡统计见[同轮原始采样及复算](../full-answer-joint-20261030/joint-summary.json)。
完整控制器生命周期成本另列，不混入请求测量窗口；旧8.3节短输出结果保留原评分，不作为本轮完整回答成本。

"""
    report = report.replace("## 九、测试结论", cost_section + "## 九、测试结论", 1)
    report = report.replace(
        "见8.3节；同预算质量及精确配对利用率仍未闭合。",
        "见8.3节；该旧out256配置的质量不合格。新8192完整回答联合成本及同期利用率见8.4节。",
        1,
    ).replace(
        "下表保留历史开发栈资源系数。固定B0原件恢复后已按用户指定的公开类似机器报价另列租赁估算，见8.3节；不是本机账单或完整业务成本验收。",
        "下表保留历史开发栈资源系数。8.3节为固定B0原out256复算；8.4节列新完整回答联合成本与质量。报价均为公开类似机型估算，不是本机账单。",
        1,
    )
    report = report.replace(
        "### 5.12 当前纠正实现质量", "### 5.12 纠正实现既有质量（原配置）", 1
    )
    report = report.replace(
        "### 7.4 固定0.18连续补测", "### 7.4 固定0.18连续补测（原锁页分配失败保留）", 1
    )
    report = report.replace(
        "### 8.3 固定B0恢复复算与质量适用性",
        "### 8.3 固定B0原out256恢复复算与质量适用性",
        1,
    )
    before, conclusion = report.split("## 九、测试结论", 1)
    conclusion, tail = conclusion.split("## 十、", 1)
    conclusion = conclusion.replace(
        "当前纠正组合的新质量结果为156/200", "此前纠正组合原配置的质量结果为156/200", 1
    )
    conclusion = conclusion.replace(
        "此前146/200属于此前恢复配置，不代表当前纠正组合。",
        "该156/200不替代5.13节本轮4GiB/rank配置的实测质量。",
        1,
    )
    lines = []
    for line in conclusion.splitlines():
        key = line.split("|")[1].strip() if line.startswith("|") else ""
        if key == "数据集性能":
            line = f"| 数据集性能 | 本轮固定200题完整回答：{a['correct']}/200对{b['correct']}/200，同轮吞吐与逐请求时延已测 | {quality}；适用本次批量负载，不外推其他工况 |"
        elif key == "成本与利用率":
            line = f"| 成本与利用率 | 本轮完整回答费用用量、质量与双卡利用率同步采齐，成本降低{pair['rental_cost_reduction_percent']:.4f}% | 公开类似机型报价、30天约定；结果与质量和逐卡利用率共同解释 |"
        lines.append(line)
    report = (
        before
        + "## 九、测试结论\n\n"
        + summary
        + "\n\n下列文字保留既有配置的结果；本轮完整回答见5.13、7.5及8.4节。\n"
        + "\n".join(lines)
        + "\n\n## 十、"
        + tail
    )
    (ROOT / REPORT).write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    revision = subprocess.check_output(
        ["git", "rev-parse", args.revision], cwd=ROOT, text=True
    ).strip()

    def read(path):
        return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)

    (ROOT / OUTPUT).write_text(
        json.dumps(build(revision, read), ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
