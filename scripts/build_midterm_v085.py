"""Add existing C16 equal-request latency without changing task-quality results."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告.md"
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v084.md"
SOURCE = "reports/c16-request-latency-reuse-20261026/summary.json"
OUTPUT = ROOT / "website-publication/data/midterm-v085.json"


def table_rows(value):
    b0, b1 = value["arms"]
    rows = []
    for metric in ("ttft", "tpot"):
        for field, label in [
            ("mean_seconds", "mean"),
            ("p95_seconds", "P95"),
            ("p99_seconds", "P99"),
        ]:
            rows.append(
                [
                    metric.upper() + " " + label,
                    f"{b0['window'][metric][field]:.9f}",
                    f"{b1['window'][metric][field]:.9f}",
                    f"{value['descriptive_reduction_percent'][metric]:.6f}%"
                    if field == "mean_seconds"
                    else "-",
                ]
            )
    return rows


def render_report(old, value):
    assert "报告版本：V0.84" in old
    old = old.replace(
        "报告版本：V0.84（HLE指定0.18完整基线与SWE-bench-Pro单Agent实测）",
        "报告版本：V0.85（补齐既有C16指定基线逐请求时延）",
        1,
    )
    old = old.replace(
        "客户端版本、B1设备占用及正式验收适用性待核",
        "客户端记录版本已恢复；B1设备占用及质量/SLO适用性待核",
    )
    old = old.replace(
        "客户端运行时版本及B1设备占用原始采样仍待补核",
        "客户端记录版本已恢复，B1设备占用原始采样及质量/SLO适用性仍待核验",
    )
    lines = old.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if cells[1].strip() in ("M03", "M04"):
            metric = "ttft" if cells[1].strip() == "M03" else "tpot"
            b0, b1 = value["arms"]
            cells[4] = (
                f" C16固定0.18 B0 {b0['window'][metric]['mean_seconds']:.9f}秒、B1 {b1['window'][metric]['mean_seconds']:.9f}秒，等权均值降低{value['descriptive_reduction_percent'][metric]:.6f}%，见3.3.1节 "
            )
            cells[5] = (
                " C16时延数值门槛达到；全部占用资源及适用质量/SLO仍待核验，不外推其他工况 "
            )
            lines[index] = "|".join(cells)
        elif cells[1].strip() == "数据集性能":
            cells[2] = (
                " C16固定0.18吞吐及逐请求等权TTFT/TPOT已复算；客户端记录版本已恢复 "
            )
            cells[3] = (
                " B1全部占用资源及适用质量/SLO；B0未记录源码提交作为来源限制披露，不因此机械重跑 "
            )
            lines[index] = "|".join(cells)
    old = "\n".join(lines) + "\n"
    section = "\n".join(
        [
            "### 3.3.1 C16逐请求时延复算",
            "",
            "复用两侧原始请求，不新增推理。TTFT为首输出接收时间减请求开始时间；TPOT为末输出与首输出接收时间差除以输出token数减一，再按请求等权平均。P95/P99用(n-1)q线性插值。主表仅统计900秒内发起并完成的成功请求：B0 392条、B1 1280条；含drain为408/1296条，单列原始汇总。两侧失败均为0。",
            "",
            "| 指标 | B0（秒） | B1（秒） | 等权均值下降 |",
            "|---|---:|---:|---:|",
            *["| " + " | ".join(row) + " |" for row in table_rows(value)],
            "",
            "客户端元数据分别记录0.1.2与0.1.1；B1记录tool_revision为29136f1f481ebab8566014a05b7ed53bcf79dc84，B0未记录源码提交。时延数值不再列为缺失测量；运行时import不是本次离线重建的内容。B1全部占用资源及适用任务质量/SLO尚须核验。本表不是单MOD消融，也不是连续4K/16K稳定性配对。",
            "",
            "证据：[规范汇总及离线复算](../c16-request-latency-reuse-20261026/README.md)。所有原始档案、drain和历史判定保留。",
            "",
        ]
    )
    anchor = "### V1.5 既有证据复算"
    assert old.count(anchor) == 1
    return old.replace(anchor, section + "\n" + anchor, 1)


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    report = read(REPORT)
    assert "报告版本：V0.85" in report.decode()
    data = json.loads(read("website-publication/data/midterm-v084.json"))
    value = json.loads(read(SOURCE))
    assert value["new_inference_requests"] == 0
    data.update(
        version="V0.85",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        c16_equal_request_latency=value,
    )
    data["milestone_contract"]["decision_relevant_gaps"]["performance"] = [
        "C16 B1 full physical resource occupancy and applicable quality/SLO evidence",
        "C16 request means are recovered; other workloads are assessed within their own scope",
    ]
    section = dict(
        id="c16-request-latency",
        title="C16指定基线时延 / C16 fixed-baseline latency",
        scope="既有900秒SWE-prefix请求复算，无新增推理。B0 392、B1 1280条窗口内成功请求；含drain 408/1296单列。TTFT/TPOT按请求等权，P95/P99线性插值。记录客户端0.1.2/0.1.1；不是现场import证明。B1全部资源占用及任务质量/SLO仍须核验，不将描述性时延改善改写为完整验收通过。",
        scope_en="Reanalysis of existing 900-second SWE-prefix runs, no new inference. B0/B1 have 392/1280 successful in-window requests; 408/1296 including drain are reported separately. Equal-request TTFT/TPOT means and interpolated P95/P99. Recorded client versions are 0.1.2/0.1.1, not fresh import attestations. Full B1 resource occupancy and applicable task quality/SLO still require evidence; descriptive latency gains are not full acceptance.",
        source_heading="3.3.1 C16逐请求时延复算",
        tables=[
            dict(
                headers=["指标", "B0（秒）", "B1（秒）", "均值下降"],
                rows=table_rows(value),
            )
        ],
        tables_en=[
            dict(
                headers=["Metric", "B0 seconds", "B1 seconds", "Mean reduction"],
                rows=table_rows(value),
            )
        ],
        evidence=[
            dict(
                path=SOURCE,
                kind="blob",
                label="逐请求时延复算 / Request-latency reanalysis",
            )
        ],
    )
    index = next(i for i, s in enumerate(data["sections"]) if s["id"] == "baseline018")
    data["sections"].insert(index + 1, section)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision")
    parser.add_argument("--prepare-report", action="store_true")
    args = parser.parse_args()
    if args.prepare_report:
        (ROOT / REPORT).write_text(
            render_report(
                (ROOT / OLD_REPORT).read_text(), json.loads((ROOT / SOURCE).read_text())
            )
        )
    else:
        assert args.revision
        OUTPUT.write_text(
            json.dumps(build(args.revision), indent=2, ensure_ascii=False) + "\n"
        )
