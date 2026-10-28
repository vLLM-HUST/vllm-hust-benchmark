"""Refresh the current four-dataset index from the repository-expansion catalog."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/swe-native018-repository-20261026/catalog.json"


def main():
    value = json.loads(SOURCE.read_text())
    p = ROOT / "reports/four-primary-datasets-20261025/README.md"
    lines = p.read_text().splitlines()
    matches = 0
    for i, line in enumerate(lines):
        if line.startswith("|") and line.split("|")[1].strip() == "SWE-bench-Pro":
            matches += 1
            lines[i] = (
                f"| SWE-bench-Pro | 指定0.18 B0已执行{value['attempted_unique_tasks']}题，官方原记录"
                f"{value['recorded_official_resolved']}/{value['recorded_official_denominator']}，未决0；"
                f"{value['application_evidence_limited_tasks']}题应用证据不足，"
                f"{value['confirmed_patch_application_failure_tasks']}题确定应用失败。element-web已应用补丁且实际测试失败，保留原解析器覆盖限制。 | "
                f"{value['unattempted_tasks']}题未执行；不是全量通过率或B0/B1配对。 |"
            )
    assert matches == 1
    text = "\n".join(lines) + "\n"
    text = text.replace(
        "[批次实测及应用取证](../swe-native018-batch-20261026/README.md)",
        "[跨仓库实测及评分覆盖](../swe-native018-repository-20261026/README.md)",
    )
    text = text.replace(
        "[可复算任务目录](../swe-native018-batch-20261026/catalog.json)",
        "[可复算任务目录](../swe-native018-repository-20261026/catalog.json)",
    )
    p.write_text(text)


if __name__ == "__main__":
    main()
