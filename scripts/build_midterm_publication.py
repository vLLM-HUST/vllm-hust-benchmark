"""Build a report-pinned website view; audit archived KV controls without inference."""

import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "f5f39e035582e26dee5f46cbbc055519ad66194f"  # pragma: allowlist secret
REPORT_DIR = "reports/vllm-hust-midterm-20261019"
REPORT = REPORT_DIR + "/中期测试报告.md"
OUTPUT = ROOT / "website-publication/data/midterm-v050.json"


def pinned(path):
    return subprocess.check_output(["git", "show", f"{REVISION}:{path}"], cwd=ROOT)


def section(text, heading):
    start = text.index(heading)
    level = len(heading.split(" ", 1)[0])
    tail = text[start + len(heading) :]
    boundary = re.search(r"(?m)^#{1," + str(level) + r"} ", tail)
    return tail[: boundary.start()] if boundary else tail


def tables(text):
    found = []
    for block in re.findall(r"(?m)(?:^\|.*\|\s*\n)+", text):
        rows = [
            [c.strip() for c in row.strip().strip("|").split("|")]
            for row in block.strip().splitlines()
        ]
        if len(rows) > 2 and all(
            re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in rows[1]
        ):
            found.append({"headers": rows[0], "rows": rows[2:]})
    return found


def kv_audit():
    results = []
    all_salts = set()
    for name in ("b1-kvmat-screen", "b1-kvmat-e24-screen"):
        summary = json.loads((ROOT / REPORT_DIR / f"{name}.json").read_text())
        archive = ROOT / REPORT_DIR / summary["archive"]["file"]
        assert (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            == summary["archive"]["sha256"]
        )
        with tarfile.open(archive) as tar:
            contents = {}
            for member in tar.getmembers():
                if member.isfile():
                    relative = member.name.split("/", 1)[1]
                    raw = tar.extractfile(member).read()
                    assert (
                        hashlib.sha256(raw).hexdigest()
                        == summary["archive"]["files"][relative]
                    )
                    contents[relative] = raw
        run = summary["run_id"]
        policy = [
            json.loads(line)
            for line in contents[f"{run}/kvmat-policy.jsonl"].splitlines()
        ]
        engine = [
            json.loads(line)
            for line in contents[f"{run}/kvmat-engine.jsonl"].splitlines()
        ]
        salts = {
            hashlib.sha256(row["runtime_cache_salt"].encode()).hexdigest()
            for row in policy
        }
        assert len(policy) == len(engine) == len(salts) == 200
        assert salts == {row["cache_salt_sha256"] for row in engine}
        assert not salts & all_salts
        all_salts.update(salts)
        assert all(
            row["runtime_effective_decision"] == "recompute"
            and row["runtime_target_reuse_tokens"] == 0
            for row in policy
        )
        rates = re.findall(
            r"Prefix cache hit rate: ([0-9.]+)%",
            contents[f"{run}/service.log"].decode(),
        )
        assert rates and all(float(rate) == 0 for rate in rates)
        receipt = json.loads(contents[f"{run}/receipt.json"])
        assert receipt["all_devices_released"] and receipt["bench_exit_code"] == 0
        before = contents[f"{run}/npu-before.txt"].decode()
        assert all(
            f"No running processes found in NPU {card}" in before
            for card in range(4, 8)
        )
        results.append(
            {
                "run_id": run,
                "requests": len(policy),
                "unique_salts": len(salts),
                "prefix_hit_samples": len(rates),
                "max_logged_prefix_hit_percent": max(map(float, rates)),
                "service_pid": receipt["service_pid"],
                "devices_released": True,
                "archive_sha256": summary["archive"]["sha256"],
            }
        )
    return results


def build():
    raw = pinned(REPORT)
    text = raw.decode()
    assert "报告版本：V0.50" in text
    url = f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{REVISION}/{REPORT}"
    specs = [
        (
            "candidates",
            "同工况性能候选 / Performance candidates",
            "### 5.4 E24/R36与KV materialization同工况补测",
            "200请求、out256。各指标分别选择B1，不将不同配置的最优指标合并为单一部署结果。",
        ),
        (
            "quality",
            "质量回归 / Quality regression",
            "### 5.7 MMLU-Pro性能输入的任务质量回归",
            "同一200题，输出上限8192；固定评分器、完整分母。达到输出上限仍可被提取器判对，不等于自然完成；不是全集评分或统计非劣性结论。",
        ),
        (
            "long-context",
            "长上下文 / Long context",
            "## 七、长上下文结果",
            "四组共160条探针，其中72条实际输入≥16K均回答正确；长度32,781、131,085、261,901 token，每条仅输出7 token。探针没有完整流式时序，不能推导TTFT/TPOT分位数。",
        ),
        (
            "long-diagnostic",
            "长输入连续运行诊断 / Sustained long-input diagnostic",
            "### 10.2 异配置候选诊断（不参与原方案验收）",
            "实际16,384 token、C8、300秒测量，固定输入温缓存工况，无语义评分。此处B1是异运行栈的诊断候选，不是中期MMLU-Pro最优B1；吞吐下降27.5086%，窗口资源成本增加37.9474%。资格预热及排空另计，不能将此组和数据集配对拼接为同一结论。",
        ),
        (
            "resources",
            "资源成本与尾时延 / Resource cost and tail latency",
            "### 5.2 MMLU-Pro同机配对补测",
            "此表为原始E16/R36配对，不是当前Microflow最优配置。主成本包含4张预约卡，另列2卡服务口径；请求区间成本下降14.63%，完整生命周期成本增加2.68%，P99 TPOT增加57.07%。没有人民币计费依据。",
        ),
        (
            "m01",
            "M01 有效算力进展 / Effective compute progress",
            "#### r21 设备映射与同轮计量",
            "V0.50：r21设备映射与同轮计量已完成；完整跨流依赖、算术覆盖和匹配参考容量尚未闭合。完整有效F和M01≥50%达成率未判定，中期整体验收待定。诊断计量不替换正式B0/B1性能结果。",
        ),
    ]
    sections = []
    for ident, title, heading, note in specs:
        selected = tables(section(text, heading))
        assert selected, heading
        sections.append(
            {
                "id": ident,
                "title": title,
                "scope": note,
                "tables": selected,
                "source_heading": heading.lstrip("# "),
            }
        )
    return {
        "schema": "midterm-publication/v1",
        "version": "V0.50",
        "report_url": url,
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": REVISION,
        "sections": sections,
        "kv_audit": {
            "status": "historical-controls-verified-not-a-bug-exclusion",
            "scope": "仅核对E16/E24逐请求重算归档；每轮独立服务、运行前四卡无进程、请求盐唯一且跨两轮不重复、已记录前缀命中率均0%。这些证据不支持常规前缀缓存跨轮命中的解释，但不能排除Mamba/GDN状态残留或底层错误。未进行新的设备复测，未将隔离性或收益因果归因判为通过。Microflow与其他配置不自动继承本审计结论。",
            "runs": kv_audit(),
        },
    }


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n")
