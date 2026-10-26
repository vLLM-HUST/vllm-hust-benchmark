"""Recompute equal-request client latency from the existing sealed C16 archives."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
B0_RUN = "dab8ed7c4ee74f2d958dab5528514294"  # pragma: allowlist secret
B1_RUN = "fa0e01a32eef456bab4bebe51e4aa7dd"  # pragma: allowlist secret


def sha(data):
    return hashlib.sha256(data).hexdigest()


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def stats(values):
    return dict(
        n=len(values),
        mean_seconds=statistics.mean(values) if values else None,
        p95_seconds=quantile(values, 0.95),
        p99_seconds=quantile(values, 0.99),
    )


def summarize(configuration, records, include_drain=False):
    selected = [
        r
        for r in records
        if r["success"]
        and 0 <= r["start"] < configuration["duration"]
        and (include_drain or r["end"] <= configuration["duration"])
    ]
    ttft, tpot = [], []
    undefined = single_chunk = 0
    for row in selected:
        count = len(row["token_ids"])
        assert count == sum(chunk[1] for chunk in row["chunks"]) and count > 0
        start, first, last, end = (
            row[k] for k in ("start", "first_token", "last_token", "end")
        )
        assert all(math.isfinite(x) for x in (start, first, last, end))
        assert start <= first <= last <= end
        delay = first - start
        assert math.isclose(delay, row["ttft_seconds"], abs_tol=1e-9)
        ttft.append(delay)
        if count > 1:
            tpot.append((last - first) / (count - 1))
        else:
            undefined += 1
        single_chunk += len(row["chunks"]) == 1
    lengths = [r["prompt_tokens"] for r in selected]
    return dict(
        requests=len(selected),
        ttft=stats(ttft),
        tpot=stats(tpot),
        undefined_tpot=undefined,
        single_chunk_requests=single_chunk,
        input_tokens_min=min(lengths) if lengths else None,
        input_tokens_max=max(lengths) if lengths else None,
        actual_ge16k_requests=sum(x >= 16384 for x in lengths),
    )


def build(root=ROOT):
    native_root = root / "reports/c16-baseline018-reuse-20261022"
    native_manifest = json.loads((native_root / "manifest.json").read_text())
    native = b"".join(
        (native_root / p).read_bytes() for p in native_manifest["native_parts"]
    )
    assert sha(native) == native_manifest["native_archive_sha256"]
    optimized_root = root / "reports/frontier-raw-recovery-20261020"
    manifest = json.loads((optimized_root / "manifest.json").read_text())
    run = next(r for r in manifest["runs"] if r["run_id"] == B1_RUN)
    optimized = b"".join(
        (optimized_root / p["path"]).read_bytes() for p in run["archive_parts"]
    )
    assert sha(optimized) == run["archive_sha256"]
    result = dict(
        schema="c16-equal-request-latency/v1",
        new_inference_requests=0,
        method=dict(
            ttft="first_token-start",
            tpot="(last_token-first_token)/(output_tokens-1)",
            weighting="equal weight per successful request",
            quantiles="linear interpolation at (n-1)*q",
            primary_cohort="starts and completes in [0,900]; starts strictly before 900",
            drain_cohort="all in-window starts, including post-window completions",
            timestamp_scope="client stream receive times; not per-token device timestamps",
        ),
        arms=[],
    )
    for label, blob, identifier in [
        ("B0", native, B0_RUN),
        ("B1", optimized, B1_RUN),
    ]:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
            configuration = json.load(archive.extractfile("config.json"))
            rows = [json.loads(line) for line in archive.extractfile("requests.jsonl")]
        assert (
            configuration["run_id"] == identifier and configuration["duration"] == 900
        )
        result["arms"].append(
            dict(
                arm=label,
                run_id=identifier,
                archive_sha256=sha(blob),
                recorded_client_tool_version=configuration["tool_version"],
                recorded_tool_revision=configuration["server_metadata"].get(
                    "tool_revision"
                ),
                measurement_seconds=configuration["duration"],
                declared_chips=configuration["chips"],
                records=len(rows),
                failed=sum(not r["success"] for r in rows),
                window=summarize(configuration, rows),
                including_drain=summarize(configuration, rows, True),
            )
        )
    b0, b1 = result["arms"]
    result["descriptive_reduction_percent"] = {
        name: 100
        * (1 - b1["window"][name]["mean_seconds"] / b0["window"][name]["mean_seconds"])
        for name in ("ttft", "tpot")
    }
    result["limitations"] = [
        "Recorded client version is not a fresh import attestation.",
        "B1 physical occupancy sampling and applicable task quality/SLO still require evidence.",
        "Mixed long inputs are not a controlled continuous 4K/16K stability pair.",
        "These are existing deployment observations, not a new run or single-MOD causal effect.",
    ]
    return result


def render(data):
    a, b = data["arms"]
    rows = []
    for metric in ("ttft", "tpot"):
        x, y = a["window"][metric], b["window"][metric]
        rows.append(
            f"| {metric.upper()}等权均值 | {x['mean_seconds']:.9f} | {y['mean_seconds']:.9f} | {data['descriptive_reduction_percent'][metric]:.6f}% |"
        )
        for p in ("p95_seconds", "p99_seconds"):
            rows.append(
                f"| {metric.upper()} {p.split('_')[0].upper()} | {x[p]:.9f} | {y[p]:.9f} | 单独报告 |"
            )
    return "\n".join(
        [
            "# C16逐请求时延复用复算",
            "",
            "对象为Qwen3.5-35B-A3B、TP2、C16、900秒SWE-prefix-reuse窗口。B0采用vLLM 0.18/Ascend 0.18及已披露兼容补丁；B1为原vLLM-HUST E16/R20 resident均衡注意力完整部署。未新增推理。",
            "",
            "TTFT取首输出接收时间减请求开始时间；TPOT取末输出与首输出接收时间差除以输出token数减一。先逐请求计算，再等权平均；P95/P99按(n-1)q线性插值。主表只统计区间内发起并完成的成功请求，drain另列在summary.json，不混入主表。",
            "",
            "| 指标（秒） | B0 | B1 | 均值下降 |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            f"B0主表{a['window']['requests']}条、含drain共{a['records']}条；B1主表{b['window']['requests']}条、含drain共{b['records']}条。两侧失败均为0，TPOT未定义及单chunk请求均为0。",
            "",
            "客户端记录版本分别为0.1.2与0.1.1；B1元数据记录tool_revision为29136f1f481ebab8566014a05b7ed53bcf79dc84，B0没有对应提交字段。该记录不是运行时import证明。原始档案不修改。",
            "",
            "时延数值已可复算，不再作为缺失测量字段。不同栈、图模式、驻留策略和回复反馈产生的后续输入差异沿用原复用审计披露。B1全部占用资源及适用任务质量/SLO仍须核验，不据此宣布完整吞吐时延验收通过。混合长输入也不替代连续4K/16K稳定性配对。",
            "",
            "来源：[既有0.18复用审计](../c16-baseline018-reuse-20261022/README.md)、[B1原始档案](../frontier-raw-recovery-20261020/README.md)。summary.json保存逐臂原始档案SHA256、请求数、分母、drain和配置来源。",
            "",
            "```bash",
            "python reports/c16-request-latency-reuse-20261026/derive.py --check",
            "```",
            "",
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build()
    if args.check:
        assert json.loads((HERE / "summary.json").read_text()) == value
        prose = (HERE / "README.md").read_text()
        for arm in value["arms"]:
            for metric in ("ttft", "tpot"):
                assert f"{arm['window'][metric]['mean_seconds']:.9f}" in prose
    else:
        (HERE / "summary.json").write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        )
        (HERE / "README.md").write_text(render(value))
