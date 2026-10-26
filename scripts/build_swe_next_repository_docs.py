"""Generate the next public evidence index from canonical observations."""

import hashlib
import json
from pathlib import Path

from build_midterm_v089 import coverage_note

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/swe-native018-next-repository-20261026"
PRIOR = ROOT / "reports/swe-native018-repository-20261026"


def main():
    catalog = json.loads((FOLDER / "catalog.json").read_text())
    inputs = json.loads((FOLDER / "inputs.json").read_text())
    count, resolved = (
        catalog["attempted_unique_tasks"],
        catalog["recorded_official_resolved"],
    )
    lines = [
        "# SWE-bench-Pro跨仓库增量实测",
        "",
        "## 测试对象与方法",
        "",
        "Qwen3.5-35B-A3B，vLLM 0.18.0 / vLLM-Ascend 0.18.0，TP2。各任务沿用冻结的官方Agent、输入与独立评分器；100步、1200秒、max_tokens=4096、temperature=0、seed=0、thinking=false，零自动重试。任务和评分使用不同容器，原始补丁不改写。",
        "",
        "## 结果",
        "",
        "| 运行 | 官方原记录 | 调用数 | Agent秒 | 应用退出码 | F2P | P2P |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in inputs["observations"]:
        v = json.loads((FOLDER / item["summary"]).read_text())
        g, a = v["grading"], v["agent"]
        f, p = (
            g["official_test_coverage"]["fail_to_pass"],
            g["official_test_coverage"]["pass_to_pass"],
        )
        lines.append(
            f"| {v['run_id']} | {int(g['official_result'])}/1 | {a['model_calls']} | {a['elapsed_seconds']:.6f} | {g['apply_exit_code']} | {f['passed']}/{f['expected']} | {p['passed']}/{p['expected']} |"
        )
    lines += [
        "",
        f"累计已执行{count}/731题，官方原记录{resolved}/{count}，尚未执行{731 - count}题。此前1题补丁应用证据不足、2题确定应用失败及全部有效负结果保留；不代表全量通过率或B0/B1配对。",
        "",
        "## 证据",
        "",
        "公开材料仅含聚合结果、源身份和证据SHA256，不含题目、回答、请求标识、补丁或轨迹。受控原件按各任务custody文件索引；原失败记录不覆盖。",
        "",
        "运行 python3 verify.py 可离线复核目录合并、原官方判分聚合及独立审计绑定。derive_private_*.py.source 是原派生源码，需要受控原件，不能仅靠公开包重新执行完整评分。before/after/graded 分别为应用前、应用后及测试前的实际待评分状态。",
        "",
        "完整目录：[catalog.json](catalog.json)；公开文件清单：[public-files.json](public-files.json)。",
        "",
    ]
    notes = [
        coverage_note(entry).strip()
        for entry in catalog["entries"]
        if entry.get("raw_test_coverage")
    ]
    position = lines.index("## 证据")
    lines[position:position] = ["\n\n".join(notes), ""] if notes else []
    (FOLDER / "README.md").write_text("\n".join(lines))
    index = ROOT / "reports/four-primary-datasets-20261025/README.md"
    text = index.read_text().splitlines()
    for i, line in enumerate(text):
        if line.startswith("|") and line.split("|")[1].strip() == "SWE-bench-Pro":
            text[i] = (
                f"| SWE-bench-Pro | 指定0.18 B0已执行{count}题，官方原记录{resolved}/{count}，未决0；原1题应用证据不足、2题确定应用失败均保留。新增各题的实际应用状态和完整F2P/P2P分别归档。 | {731 - count}题未执行；不是全量通过率或B0/B1配对。 |"
            )
    index.write_text(
        "\n".join(text).replace(
            "../swe-native018-repository-20261026/",
            "../swe-native018-next-repository-20261026/",
        )
        + "\n"
    )
    paths = {
        entry["path"] for entry in json.loads((PRIOR / "public-files.json").read_text())
    }
    paths.add(str((PRIOR / "public-files.json").relative_to(ROOT)))
    names = [
        "README.md",
        "catalog.json",
        "inputs.json",
        "derive_catalog.py",
        "verify.py",
    ]
    for item in inputs["observations"]:
        names.extend(
            item[key]
            for key in ("summary", "review", "release", "custody", "derivation_source")
        )
    paths.update(str((FOLDER / name).relative_to(ROOT)) for name in names)
    rows = []
    for name in sorted(paths):
        raw = (ROOT / name).read_bytes()
        rows.append(
            {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    (FOLDER / "public-files.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
