"""Project a whitelisted independent review and build the incremental evidence index."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reports/swe-native018-tutanota-20261027"
PRIOR = ROOT / "reports/swe-native018-protonmail-qutebrowser-20261027"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n")


def project_review(s):
    g, a, e = s["official_grading"], s["agent"], s["evidence"]
    assert s["new_review_model_or_grading_calls"] == 0
    assert g["test_counts"] == {"ERROR": 1}
    assert g["fail_to_pass"] == {"expected": 1, "passed": 0, "failed": 0, "missing": 1}
    assert g["resolved"] is False and g["unresolved"] == []
    assert a["agent_submitted"] is False and a["status"] == "LimitsExceeded"
    assert (
        s["request_audit"]["attempts"]
        == s["request_audit"]["successful_returns"]
        == a["model_calls"]
        == 100
    )
    assert s["request_audit"]["rejected"] == s["request_audit"]["http_retries"] == 0
    return {
        "status": s["status"],
        "projection_source": "storage-review.json",
        "official_result": g["resolved"],
        "unresolved": len(g["unresolved"]),
        "request_calls": a["model_calls"],
        "agent_seconds": a["elapsed_seconds"],
        "patch_application": {
            "patch_sha256": a["patch_sha256"],
            "exit_code": s["partial_patch"]["apply_exit_code"],
        },
        "fail_to_pass": g["fail_to_pass"],
        "pass_to_pass": g["pass_to_pass"],
        "seal": {"cpu_files_verified": e["CPU_files"], "service_files_verified": 14},
        "test_coverage": {
            "target_test_build_failed": True,
            "target_tests_executed": False,
            "target_test_assertions_failed": False,
            "compiler_errors": s["compiler_errors"],
            "official_parser_records": 1,
            "source_boundary": s["test_execution_boundary"],
        },
        "no_reruns": True,
        "new_model_calls": s["new_review_model_or_grading_calls"],
        "original_independent_review_sha256": e["storage_final_review_sha256"],
    }


def prepare():
    s = json.loads((HERE / "storage-review.json").read_text())
    v = json.loads((HERE / "summary-tutanota.json").read_text())
    assert (
        sha(HERE / "summary-tutanota.json")
        == s["evidence"]["public_CPU_summary_sha256"]
    )
    save("review-tutanota.json", project_review(s))
    save(
        "custody-tutanota.json",
        {
            "run_id": v["run_id"],
            "controlled_location": "Authorized CPU persistent workspace; runs directory by run_id",
            "seal": "SHA256SUMS-batch-r1",
            "seal_sha256": s["evidence"]["CPU_seal_sha256"],
            "files_verified": s["evidence"]["CPU_files"],
            "private_contents_included": False,
            "old_seals_preserved": True,
        },
    )
    save(
        "inputs.json",
        {
            "prior_catalog_sha256": sha(PRIOR / "catalog.json"),
            "observations": [
                {
                    "summary": "summary-tutanota.json",
                    "summary_sha256": sha(HERE / "summary-tutanota.json"),
                    "review": "review-tutanota.json",
                    "review_sha256": sha(HERE / "review-tutanota.json"),
                    "release": "release-tutanota.json",
                    "custody": "custody-tutanota.json",
                    "derivation_source": "derive_private_tutanota.py.source",
                }
            ],
        },
    )


def documents():
    import build_midterm_v096 as b

    v = json.loads((HERE / "catalog.json").read_text())
    recovery = json.loads((ROOT / b.previous.RECOVERY).read_text())
    a = b.aggregate(v, recovery)
    save("current-aggregate.json", a)
    text = (
        "# SWE-bench-Pro Tutanota增量实测\n\n"
        "## 对象与方法\n\n"
        "Qwen3.5-35B-A3B，vLLM 0.18.0 / vLLM-Ascend 0.18.0，TP2。原任务、Agent、官方评分器及预算不变："
        "100步、1200秒、max_tokens=4096、temperature=0、seed=0、thinking=false、零自动重试；Agent与评分使用不同新容器。\n\n"
        "## 结果\n\n" + b.note() + "\n\n"
        "累计18/731个不同任务，官方原记录2/18；保留同一Protonmail答案依赖恢复后为3/18，未执行713题。"
        "恢复评分没有增加Agent尝试；不是全量通过率或B0/B1配对验收。旧应用失败、应用证据不足、未提交部分补丁、编译失败和测试依赖限制均保留。\n\n"
        "## 证据\n\n"
        "[规范目录](catalog.json)、[当前聚合](current-aggregate.json)、[本次原派生摘要](summary-tutanota.json)、"
        "[独立复核白名单](storage-review.json)、[释放回执](release-tutanota.json)。"
        "公开材料不含题目、答案、补丁或轨迹正文；受控原件以SHA256索引。应用前、应用后及测试前实际状态分别保留。\n\n"
        "`python3 verify.py`可离线复核公开聚合和绑定；`derive_private_tutanota.py.source`需受控原件，不能仅凭公开摘要重跑完整评分。\n"
    )
    (HERE / "README.md").write_text(text)
    index = ROOT / "reports/four-primary-datasets-20261025/README.md"
    lines = index.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and line.split("|")[1].strip() == "SWE-bench-Pro":
            lines[i] = (
                "| SWE-bench-Pro | 指定0.18 B0已尝试18/731题，官方原记录2/18；同一答案依赖恢复后3/18。Tutanota未提交部分补丁应用成功但编译失败，功能测试未执行；旧失败及限制保留。 | 713题及B1同题配对未完成；[规范目录](../swe-native018-tutanota-20261027/catalog.json)。 |"
            )
    index.write_text("\n".join(lines) + "\n")
    files = {x["path"] for x in json.loads((PRIOR / "public-files.json").read_text())}
    files.add(str((PRIOR / "public-files.json").relative_to(ROOT)))
    files.add(b.previous.RECOVERY)
    files.update(
        str(p.relative_to(ROOT))
        for p in HERE.iterdir()
        if p.is_file() and p.name != "public-files.json"
    )
    save(
        "public-files.json",
        [
            {"path": p, "bytes": (ROOT / p).stat().st_size, "sha256": sha(ROOT / p)}
            for p in sorted(files)
        ],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "documents"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else documents()
