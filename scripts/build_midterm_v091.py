"""Project one original Terminal answer with disclosed verifier-only recovery."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import build_midterm_v090 as previous

ROOT = Path(__file__).resolve().parents[1]
REPORT = previous.REPORT
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v090.md"
SOURCE = "reports/terminal-openssl-model-recovery-20261026/summary.json"
OUTPUT = ROOT / "website-publication/data/midterm-v091.json"


def scope(english=False):
    if english:
        return (
            "TerminalBench2.1 OpenSSL: one original Qwen3.5-35B-A3B native 0.18/0.18 Agent trial, "
            "5 model calls in 86.812555 seconds. Its original verifier timed out downloading uv after 900 seconds, reward null. "
            "Verifier-only recovery using pinned uv 0.9.5 transport caches passed all 6 original tests, reward 1, "
            "in 11.784143 seconds within the unchanged 900-second budget. No new Agent, model calls or controls. "
            "Four SSL files retain original snapshots; check_cert.py is deterministically reconstructed from the original command and echo, "
            "not an original file snapshot; its original mode is unknown. One task scored and passed out of one attempted; "
            "88 of 89 tasks remain unattempted. ARM64 derived image, not full-dataset or B0/B1 paired acceptance."
        )
    return (
        "TerminalBench2.1 OpenSSL采用Qwen3.5-35B-A3B官方0.18/0.18，同一次Agent共5次模型调用、86.812555秒。"
        "原评分因uv下载在900秒超时，原reward=null保留。固定uv0.9.5传输缓存后仅恢复评分，"
        "原6项测试全部通过、reward=1，耗时11.784143秒，仍使用原900秒预算；新增Agent、模型调用及控制实验均为0。"
        "四个SSL文件复用原快照；check_cert.py由原命令及终端回显确定性重建，非现场原文件快照，原模式未知。"
        "已执行1/89题，该题恢复评分通过，另88题未执行；独立ARM64派生镜像，不构成全量或B0/B1配对验收。"
    )


def render_report(old):
    assert "报告版本：V0.90" in old
    text = old.replace(
        "报告版本：V0.90（SWE增量实测与Terminal ARM64评分控制）",
        "报告版本：V0.91（Terminal原答案恢复评分）",
        1,
    )
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and line.split("|")[1].strip() == "TerminalBench 2.1":
            lines[i] = (
                "| TerminalBench 2.1 | " + scope() + " | 其余88题与B1同题配对尚未完成 |"
            )
    text = "\n".join(lines) + "\n"
    marker = "#### 0.18 Native与已有优化方案同题质量"
    assert marker in text
    section = (
        "#### TerminalBench2.1模型实测及恢复评分\n\n"
        + scope()
        + "\n\n原Harbor 0.24.0 Verifier及测试不变，资源为1CPU、2GiB、MTU1400。"
        "原日志中的env文件缺失警告保留；随后原uvx执行pytest 8.4.1，6项测试均通过。"
        "评分前后5个恢复文件的SHA256及模式一致，控制器退出0，评分容器及原模型服务已释放。"
        "\n\n证据：[规范结果与离线复核](../terminal-openssl-model-recovery-20261026/README.md)。\n\n"
    )
    return text.replace(marker, section + marker, 1)


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    data = previous.build(revision, lambda p: read(OLD_REPORT if p == REPORT else p))
    report = read(REPORT)
    summary = json.loads(read(SOURCE))
    assert "报告版本：V0.91" in report.decode()
    assert summary["recovery"]["passed"] == 6 and summary["model_task_attempts"] == 1
    data.update(
        version="V0.91",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        terminal_model_recovery=summary,
    )
    data["sections"].append(
        {
            "id": "terminal-model-recovery",
            "title": "Terminal原答案恢复评分 / Recovered grading",
            "scope": scope(),
            "scope_en": scope(True),
            "source_heading": "TerminalBench2.1模型实测及恢复评分",
            "tables": [
                {
                    "headers": ["执行", "模型调用", "测试通过", "奖励", "资源释放"],
                    "rows": [
                        [
                            "原Agent及原评分",
                            "5",
                            "未启动测试",
                            "null（900秒超时）",
                            "已释放",
                        ],
                        ["仅恢复原答案评分", "0", "6 / 6", "1", "已释放"],
                    ],
                }
            ],
            "tables_en": [
                {
                    "headers": [
                        "Execution",
                        "Model calls",
                        "Tests passed",
                        "Reward",
                        "Resources",
                    ],
                    "rows": [
                        [
                            "Original Agent and verifier",
                            "5",
                            "Tests not started",
                            "null (900s timeout)",
                            "Released",
                        ],
                        ["Verifier-only recovery", "0", "6 / 6", "1", "Released"],
                    ],
                }
            ],
            "evidence": [
                {
                    "path": SOURCE,
                    "kind": "blob",
                    "label": "原答案恢复评分 / Recovered original answer",
                }
            ],
        }
    )
    remaining = next(x for x in data["sections"] if x["id"] == "v15-remaining")
    remaining["evidence"].append(data["sections"][-1]["evidence"][0].copy())
    for key, english in (("tables", False), ("tables_en", True)):
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] == "TerminalBench 2.1":
                    row[1] = scope(english)
                    if len(row) > 2:
                        row[2] = (
                            "88 tasks and paired B1 remain."
                            if english
                            else "其余88题与B1同题配对尚未完成。"
                        )
    return data


def prepare():
    (ROOT / REPORT).write_text(render_report((ROOT / OLD_REPORT).read_text()))
    directory = ROOT / "reports/terminal-openssl-model-recovery-20261026"
    (directory / "README.md").write_text(
        "# TerminalBench2.1 OpenSSL原答案恢复评分\n\n"
        + scope()
        + "\n\n## 配置与方法\n\nHarbor 0.24.0 / Terminus2，原任务、测试及评分器不变，"
        "1CPU、2GiB、MTU1400。原Agent最大100轮、900秒、max_tokens=4096；"
        "原Harbor及Terminus内层重试策略各保留3次上限，SDK重试0，不套用SWE零重试合同。"
        "恢复评分仅一次，预算900秒。固定官方uv安装器及ARM64包经哈希核验，"
        "仅更换下载传输为本地缓存；不声称传递依赖wheel与历史完全相同。"
        "\n\n## 结果与证据\n\n评分前后5文件SHA及模式一致；原env文件缺失警告保留，"
        "随后原uvx/pytest执行成功。所有模型服务和评分容器均已释放。"
        "原超时/null见original-trial.json；恢复结果见summary.json、tests.json和recovery-receipt.json。"
        "独立复核见independent-review.json及profiling-review.json；受限原件索引见restricted-evidence-index.json。"
        "私有题目、答案、证书私钥和原始轨迹未公开。"
        "\n\n离线校验：\n\n```sh\npython3 verify.py\n```\n\n"
        "离线入口复核公开聚合及绑定，不声称可脱离受控原件重跑原始审计。\n"
    )
    path = ROOT / "reports/four-primary-datasets-20261025/README.md"
    if not path.exists():
        paths = list((ROOT / "reports").glob("*/README.md"))
        paths = [p for p in paths if "four-primary" in str(p)]
        assert len(paths) == 1
        path = paths[0]
    lines = path.read_text().splitlines()
    found = 0
    for i, line in enumerate(lines):
        if line.startswith("|") and line.split("|")[1].strip() in (
            "TerminalBench 2.1",
            "TerminalBench2.1",
        ):
            lines[i] = (
                "| TerminalBench 2.1 | " + scope() + " | 88题及B1同题配对尚未完成。 |"
            )
            found += 1
    assert found == 1
    path.write_text("\n".join(lines) + "\n")
    index = ROOT / "reports/vllm-hust-midterm-20261019/README.md"
    text = index.read_text()
    link = "- Terminal原答案恢复评分：[规范结果](../terminal-openssl-model-recovery-20261026/README.md)。"
    if link not in text:
        index.write_text(text.rstrip() + "\n\n" + link + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-report", action="store_true")
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.prepare_report:
        prepare()
    else:
        assert args.revision
        OUTPUT.write_text(
            json.dumps(build(args.revision), ensure_ascii=False, indent=2) + "\n"
        )
