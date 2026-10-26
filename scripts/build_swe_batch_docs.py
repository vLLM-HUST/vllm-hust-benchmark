"""Refresh current SWE evidence navigation from canonical aggregates."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/swe-native018-batch-20261026"


def render(value):
    rows = []
    for entry in value["entries"]:
        scope = (
            "应用证据不足"
            if entry["patch_application_evidence_limited"]
            else "确定应用失败"
            if entry["confirmed_patch_application_failure"]
            else "已应用补丁"
            if entry.get("patch_applied") is True
            else "原官方评分"
        )
        rows.append(
            f"| {entry['run_id']} | {entry['official_resolved']}/1 | "
            f"{entry['model_calls']} | {entry['elapsed_seconds']:.6f} | {scope} |"
        )
    return "\n".join(
        [
            "# SWE-bench-Pro V1 指定基线批次实测",
            "",
            (
                "模型Qwen3.5-35B-A3B，vLLM 0.18.0 / vLLM-Ascend 0.18.0。"
                "本目录追加此前未执行任务，前三题历史原件及结果直接复用。"
                "每题仅一次Agent，100步、1200秒、max_tokens=4096、temperature=0、seed=0、thinking=false、零自动重试。"
                "任务与独立评分各使用新容器，保留固定镜像、源码起点和原有fixture，不继承上题补丁。"
            ),
            "",
            "| 运行 | 官方原记录 | 模型调用 | Agent秒 | 应用证据 |",
            "|---|---:|---:|---:|---|",
            *rows,
            "",
            (
                f"累计执行{value['attempted_unique_tasks']}/731个不同任务，官方原记录"
                f"{value['recorded_official_resolved']}/{value['recorded_official_denominator']}，未决0；"
                f"{value['unattempted_tasks']}题尚未执行。补丁应用证据不足"
                f"{value['application_evidence_limited_tasks']}题，已确认应用失败"
                f"{value['confirmed_patch_application_failure_tasks']}题。"
                "保留完整已执行分母及原官方分数，不外推731题通过率，不宣称B0/B1配对完成。"
            ),
            "",
            "## 执行与判分范围",
            "",
            (
                "本批五题087/0c8/0f7/18c/ans共用一个正常驻留的独占模型窗口，APC关闭；"
                "每题输出和客户端缓存独立。9000秒为模型服务生命周期上限，不改变单题1200秒预算，"
                "不声明逐题冷启动或吞吐比较。此前049使用另一个独立模型窗口。"
                "五项既有资格在实际运行模块、架构与启动参数一致后复用，没有新增资格请求。"
            ),
            "",
            (
                "评分使用原固定官方规则，且不调用模型。增加的取证只记录单次git apply退出码和"
                "受影响文件在应用前、应用后及测试前（实际待评分）的状态，不增加第二次应用、不改补丁或测试。"
                "原始测试条目与官方按测试名集合计算的F2P/P2P分别保存，重名条目不改变任务分母。"
            ),
            "",
            (
                "049的原补丁含两个镜像已有Redis文本文件的新建段，git apply退出1；"
                "全部受影响文件前后相同，官方测试继续执行并判false。"
                "这是确定的补丁应用失败，不能表述为候选代码功能已被测试。"
                "此前36次调用任务未采集该退出码，仍只标应用证据不足，不事后追补为同样结论。"
            ),
            "",
            "## 证据与复算",
            "",
            "18c的F2P为0/7：4项已解析失败、3项预期测试名未出现，P2P为418/418。官方仍明确判false、评分未决0；不能把评分未决0解释为所有预期测试齐全。Ansible题应用成功，F2P为1/1、P2P为15/15，官方判true。",
            "",
            "- catalog.json及inputs.json：历史规范源与新汇总的SHA绑定、唯一任务分母及分类。",
            "- summary-*.json：实际运行身份、预算、原官方成绩、测试集合覆盖、应用取证、释放及受限原件SHA。",
            "- 049-independent-review.json：storage对原补丁、官方评分、应用失败的独立复核投影。",
            "- 049-independent-release.json：原服务14份封存文件、退出及设备释放的独立复核。",
            "- derive_private.py.source：本批派生程序原字节；需要受控CPU原件和固定评分源码，不是公开独立重判入口。",
            "- verify.py及public-files.json：公开文件大小/SHA和汇总算术检查；不重执行私有任务。",
            "",
            (
                "受限原件保存在授权CPU持久卷swe-pro-shuhao-20261025的runs目录，各run_id对应原始轨迹、"
                "补丁、请求时间、独立评分、运行身份和SHA256SUMS-batch-r1。"
                "公开材料不包含私有题目、答案、补丁内容、完整轨迹或凭据。"
            ),
            "",
            "```bash",
            "python3 reports/swe-native018-batch-20261026/verify.py",
            "```",
            "",
        ]
    )


def update_index(value):
    path = ROOT / "reports/four-primary-datasets-20261025/README.md"
    text = path.read_text()
    lines = text.splitlines()
    matches = 0
    for i, line in enumerate(lines):
        if line.startswith("|") and line.split("|")[1].strip() == "SWE-bench-Pro":
            matches += 1
            lines[i] = (
                f"| SWE-bench-Pro | 指定0.18 B0累计执行{value['attempted_unique_tasks']}个不同任务，"
                f"官方原记录{value['recorded_official_resolved']}/{value['recorded_official_denominator']}、未决0；"
                f"{value['application_evidence_limited_tasks']}题应用证据不足，"
                f"{value['confirmed_patch_application_failure_tasks']}题确定应用失败，原分均保留。 | "
                f"剩余{value['unattempted_tasks']}题未执行，不构成全量通过率或B0/B1配对。 |"
            )
    assert matches == 1
    text = "\n".join(lines) + "\n"
    text = text.replace("当前指定0.18三项任务", "当前指定0.18任务")
    old = "- SWE指定0.18增量：[两项新增任务及评分限制](../swe-native018-incremental-20261026/README.md)、[累计三项规范目录](../swe-native018-incremental-20261026/catalog.json)。"
    new = (
        "- SWE指定0.18当前结果：[批次实测及应用取证](../swe-native018-batch-20261026/README.md)、"
        "[可复算任务目录](../swe-native018-batch-20261026/catalog.json)；"
        "[前三题历史归档](../swe-native018-incremental-20261026/README.md)。"
    )
    assert old in text or "../swe-native018-batch-20261026/catalog.json" in text
    text = text.replace(old, new)
    path.write_text(text)


if __name__ == "__main__":
    value = json.loads((FOLDER / "catalog.json").read_text())
    (FOLDER / "README.md").write_text(render(value))
    update_index(value)
