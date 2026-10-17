#!/usr/bin/env python3
"""Everything we know how to show about one closed-loop experiment, from one
command, published as a Feishu document. Run it after every experiment instead
of drawing figures one by one.

    python experiments/workload/experiment_report.py \\
        data/from_ascend86/workload/<experiment> --feishu \\
        [--window-start S --window-end S] [--skip-run-figures]

Input: <experiment>/concurrency_<C>/<configuration>/ as concurrency_sweep.sh
writes it (rounds.jsonl, summary.json, engine_metrics_raw.jsonl.gz,
device.jsonl.gz, thunderagent.log ...). Runs without summary.json (stopped
early) are left out and listed.

Output:
    Feishu    one document per experiment in the "实验分析" folder: settings,
              definitions, every table, the main figures inline, and the path of
              every per-run figure. The first run creates it; later runs update
              the same document through scripts/sync_document_to_feishu.py, which
              refuses to overwrite a copy someone changed. Experiment -> document
              is recorded in data/feishu_reports.json.
    local     <experiment>/report/tables/*.csv and report/figures/*.png; per run
              and per point figures stay next to their data:
                per run    <run>/figures/A1-A4, B1, C1-C3, D1-D2 (plot_observability.py)
                per point  concurrency_<C>/figures/F1 (plot_program_phases.py),
                           G1 (plot_program_arrivals.py), E1-E2
                           (plot_observability_compare.py)
                sweep      figures/concurrency_sweep.png (plot_concurrency_sweep.py),
                           report/figures/figure1a_standard_steps.png and
                           step_size_bias_*.png (plot_step_size.py)
              No Markdown is kept: it exists only while uploading.
Without --feishu the document text is printed to report/preview.md instead.

The window defaults to the last two thirds of the run (the first third is the
start-up burst of the closed loop). A figure script that fails is reported in
the document; the rest still runs.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import experiment_tables as tables

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
REGISTRY = REPOSITORY / "data" / "feishu_reports.json"
# the "实验分析" folder in the user's Feishu Drive (created 2026-09-30)
FEISHU_FOLDER = "EmQVfmiL5lrmtmd4CdYcyzO2npd"

DEFINITIONS = (
    "\n".join(
        [
            "- **程序**：一个 agent 做一道题的全过程，由很多轮组成。",
            (
                "- **一步 = 一轮**：一次发给模型的请求 + 之后的工具时间（Thunde"
                "rAgent 论文 §5.1 的定义）。"
            ),
            (
                "- **一步的 prompt**：这次请求发送的全部上下文（历史 + 本轮新输入"
                "），取引擎的 `usage.prompt_tokens`。"
            ),
            (
                "- **窗口**：表格统计的时间段；一步最后一个 token 到达的时刻在窗口内"
                "，就算这一步属于窗口。"
            ),
            (
                "- **步/分钟**：窗口内完成的步数 ÷ 窗口分钟数；每一步都算 1，不论大小"
                "（论文的指标；论文没有公开怎么数）。"
            ),
            ("- **标准步**：workload 里所有程序每一轮都跑完时，一步 prompt 的平均值。"),
            (
                "- **步/分钟（按标准步折算）**：窗口内完成的步的 prompt 之和 ÷ "
                "分钟 ÷ 标准步。窗口只数到程序早期小步时也可比"
                "（本报告自定义，不是论文的指标）。"
            ),
            (
                "- **可避免的重算**：没有命中前缀缓存的 prompt token − 每轮"
                "必须新算的输入；卡时间按下面写的 prefill 速度估算。"
            ),
            (
                "- **闭环**：t=0 时 C 个程序同时到达，之后只有一个程序完成才到达一个"
                "新程序；不是流式（开环）到达。"
            ),
        ]
    )
    + "\n"
)

COLUMNS = {
    "concurrency": "并发",
    "configuration": "配置",
    "steps per minute": "步/分钟",
    "prompt tokens of those steps per minute": "这些步的 prompt token/分钟",
    "output tokens per second": "生成 token/秒",
    "mean prompt per step": "每步平均 prompt",
    "mean round index of those steps": "这些步的平均轮次",
    "steps per minute, standard-size steps": "步/分钟（按标准步折算）",
    "seconds": "时间段（秒）",
    "mean round index": "平均轮次",
    "prompt tokens per minute": "prompt token/分钟",
    "mean context per step": "每步平均上下文",
    "concurrency x context / KV capacity": "并发 × 上下文 ÷ KV 容量",
    "from cache GiB per minute": "命中缓存 GiB/分钟",
    "not from cache GiB per minute": "未命中缓存 GiB/分钟",
    "avoidable recompute tokens per second": "可避免重算 token/秒",
    "avoidable recompute card seconds per minute (estimate)": (
        "可避免重算占卡秒/分钟（估算）"
    ),
    "preemptions": "抢占次数",
    "engine requests running (mean)": "引擎运行中请求（平均）",
    "engine requests waiting (mean)": "引擎排队请求（平均）",
    "engine seconds with nothing running": "引擎完全空闲秒数",
    "tool seconds per step": "工具 秒/步",
    "engine and held seconds per step": "引擎 + 被扣 秒/步",
    "predicted steps per minute": "公式预测 步/分钟",
    "measured steps per minute": "实测 步/分钟",
    "arrived at t=0": "t=0 到达",
    "arrived later": "之后到达",
    "finished": "完成",
    "first batch finished share": "第一批完成比例",
    "first batch median waiting s": "第一批等首 token 总时间中位数（秒）",
    "later arrivals finished share": "后来者完成比例",
    "later arrivals median waiting s": "后来者等首 token 总时间中位数（秒）",
    f"programs with a wait > {tables.LONG_WAIT_SECONDS} s": (
        f"有一次等待 >{tables.LONG_WAIT_SECONDS} 秒的程序"
    ),
    "longest single wait s": "最长一次等待（秒）",
    "admitted programs": "放进来的程序",
    "reasoning": "推理中",
    "acting": "执行工具中",
    "held": "被扣着",
    "ThunderAgent counts as occupied (tokens)": "ThunderAgent 记为占用（token）",
    "engine KV held by running requests (tokens)": "引擎运行中请求实际占用 KV（token）",
    "engine requests running": "引擎运行中请求",
    "steps from small programs (<8k)": "来自小程序（<8k）的步",
    "steps from large programs (>16k)": "来自大程序（>16k）的步",
}


def run_script(script: str, *arguments: str) -> str | None:
    """Run one figure script with this interpreter; None on success, else the
    last line of its error."""
    completed = subprocess.run(
        [sys.executable, str(HERE / script), *arguments],
        capture_output=True,
        text=True,
        cwd=HERE,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    if completed.returncode == 0:
        return None
    return (completed.stderr or completed.stdout).strip().splitlines()[-1][:300]


def repository_path(path: Path) -> str:
    return os.path.relpath(path, REPOSITORY)


def image(path: Path, caption: str) -> list[str]:
    """An inline figure; the upload turns the repository path into an image."""
    if not path.exists():
        return [f"（缺图：`{repository_path(path)}`）", ""]
    return [f"![{caption}]({repository_path(path)})", ""]


def chinese_table(rows: list[dict]) -> str:
    return tables.to_markdown(
        [{COLUMNS.get(key, key): value for key, value in row.items()} for row in rows]
    )


def build_document(
    experiment,
    points,
    first,
    start,
    end,
    prefill,
    step_figures,
    sections,
    incomplete,
    failures,
) -> list[str]:
    manifest = first.summary.get("workload_manifest", {})
    engine = first.summary["engines"][0]
    configurations = sorted({c for runs in points.values() for c in runs})
    lines = [
        f"# 实验报告：{experiment.name}",
        "",
        f"由 `experiments/workload/experiment_report.py` 生成于 "
        f"{datetime.datetime.now():%Y-%m-%d %H:%M}；请重新生成，不要手改。"
        f"数据目录：`{repository_path(experiment)}`。",
        "",
        "## 设置",
        "",
        f"- 并发点：{', '.join(str(c) for c in points)}",
        f"- 配置：{', '.join(configurations)}",
        f"- 统计窗口：每次运行的第 {start:.0f}–{end:.0f} 秒",
        f"- workload：`{Path(first.summary['parameters']['workload']).name}`",
        "- workload 旋钮：`"
        + json.dumps(manifest.get("knobs", {}), ensure_ascii=False)
        + "`",
        f"- 标准步：{tables.standard_step_tokens(first):,.0f} prompt token",
        f"- 引擎：`{engine.get('model')}`，每个引擎 KV "
        f"{engine.get('kv_capacity_tokens_device'):,} token，"
        f"max_model_len {engine.get('max_model_len')}，"
        f"{len(first.summary['engines'])} 个引擎",
        f"- 估算卡时间用的 prefill 速度：{prefill:,.0f} token/秒（实测，估算用）",
        f"- 证据等级：`{first.summary.get('evidence_type')}`",
        "",
        "## 定义",
        "",
        DEFINITIONS,
        "## 吞吐随并发变化",
        "",
        *image(
            experiment / "figures" / "concurrency_sweep.png",
            "四格：步/分钟、这些步的 prompt token/分钟、单步延迟、prompt KV",
        ),
        *image(step_figures[0], "步/分钟：虚线为原始测量，实线为按标准步折算"),
    ]
    for path in step_figures[1:]:
        lines += image(path, f"步大小偏差：{path.stem.removeprefix('step_size_bias_')}")
    for key, title, rows in sections:
        lines += [
            f"## {title}",
            "",
            f"CSV：`{repository_path(experiment)}/report/tables/{key}.csv`",
            "",
            chinese_table(rows),
        ]
    lines += ["## 每个并发点：程序到达与执行、各阶段程序数", ""]
    for concurrency, runs in points.items():
        point = experiment / f"concurrency_{concurrency}" / "figures"
        lines += [f"### 并发 {concurrency}", ""]
        lines += image(
            point / "G1_program_arrivals.png",
            f"并发 {concurrency}：程序到达与每个程序的执行过程",
        )
        lines += image(
            point / "F1_program_phases.png", f"并发 {concurrency}：每一秒各阶段的程序数"
        )
        for run in runs.values():
            run_figures = sorted((run.directory / "figures").glob("*.png"))
            lines += [
                f"- {run.configuration} 的全部随时间观测图（A 引擎、B 客户端、C 程序、"
                f"D ThunderAgent 决策）：`{repository_path(run.directory)}/figures/`"
                f"（{len(run_figures)} 张）"
            ]
        if (point / "E1_configurations.png").exists():
            lines += [
                f"- 两种配置对比图：`{repository_path(point)}/E1_configurations.png`"
            ]
        lines.append("")
    lines += ["## 问题", ""]
    lines += [f"- 未完成、未计入的运行：`{i}`" for i in incomplete]
    lines += [f"- 画图失败：{f}" for f in failures]
    if not incomplete and not failures:
        lines.append("- 无")
    return lines


def publish(experiment: Path, text: str) -> str:
    """Create the experiment's document in the folder once, then write through the
    guarded sync script. Returns the document URL."""
    registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    documents = registry.setdefault("documents", {})
    key = repository_path(experiment)
    url = documents.get(key)
    if url is None:
        created = subprocess.run(
            [
                "lark-cli",
                "docs",
                "+create",
                "--as",
                "user",
                "--parent-token",
                registry.get("folder_token", FEISHU_FOLDER),
                "--title",
                f"实验报告：{experiment.name}",
                "--doc-format",
                "markdown",
                "--content",
                "（生成中）",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            cwd=REPOSITORY,
        )
        body = created.stdout[created.stdout.find("{") :]
        reply = json.loads(body) if body else {}
        if not reply.get("ok"):
            raise SystemExit(
                f"creating the Feishu document failed: {created.stdout[-500:]}"
            )
        url = reply["data"]["document"]["url"]
        documents[key] = url
        registry.setdefault("folder_token", FEISHU_FOLDER)
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY.write_text(json.dumps(registry, indent=1, ensure_ascii=False) + "\n")
        print(f"[report] created {url}")
    staged = REPOSITORY / f".feishu_report_{experiment.name}.md"
    staged.write_text(text)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY / "scripts" / "sync_document_to_feishu.py"),
                str(staged),
                url,
            ],
            capture_output=True,
            text=True,
            cwd=REPOSITORY,
        )
    finally:
        staged.unlink(missing_ok=True)
    print(completed.stdout.strip()[-1500:])
    if completed.returncode != 0:
        raise SystemExit(f"Feishu sync stopped: {completed.stderr.strip()[-800:]}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--window-start", type=float)
    parser.add_argument("--window-end", type=float)
    parser.add_argument(
        "--prefill-tokens-per-second",
        type=float,
        default=tables.PREFILL_TOKENS_PER_SECOND,
    )
    parser.add_argument(
        "--skip-run-figures",
        action="store_true",
        help="do not redraw the per-run and per-point figures (faster)",
    )
    parser.add_argument(
        "--feishu", action="store_true", help="publish to the 实验分析 folder in Feishu"
    )
    arguments = parser.parse_args()
    experiment = arguments.experiment.resolve()
    report = experiment / "report"
    (report / "tables").mkdir(parents=True, exist_ok=True)
    (report / "figures").mkdir(parents=True, exist_ok=True)

    points = tables.load(experiment)
    if not points:
        raise SystemExit(f"no complete runs under {experiment}")
    first = next(iter(next(iter(points.values())).values()))
    start, end = tables.default_window(first)
    start = arguments.window_start if arguments.window_start is not None else start
    end = arguments.window_end if arguments.window_end is not None else end
    incomplete = sorted(
        str(p.parent.relative_to(experiment))
        for p in experiment.glob("concurrency_*/*/rounds.jsonl")
        if not (p.parent / "summary.json").exists()
    )
    failures: list[str] = []

    # ---- per run and per point figures (existing scripts, unchanged)
    if not arguments.skip_run_figures:
        for concurrency, runs in points.items():
            for run in runs.values():
                print(f"[report] figures for {run.directory.relative_to(experiment)}")
                error = run_script("plot_observability.py", str(run.directory))
                if error:
                    failures.append(
                        f"plot_observability {run.directory.name} "
                        f"C={concurrency}: {error}"
                    )
            point = experiment / f"concurrency_{concurrency}"
            for script in (
                "plot_program_phases.py",
                "plot_program_arrivals.py",
                "plot_observability_compare.py",
            ):
                error = run_script(script, str(point))
                if error:
                    failures.append(f"{script} C={concurrency}: {error}")
    error = run_script(
        "plot_concurrency_sweep.py",
        str(experiment),
        "--window-start",
        str(start),
        "--window-end",
        str(end),
    )
    if error:
        failures.append(f"plot_concurrency_sweep.py: {error}")

    # ---- step size figures
    import matplotlib

    matplotlib.use("Agg")
    import plot_step_size
    from matplotlib import pyplot

    step_figures = [
        plot_step_size.figure1a(pyplot, points, start, end, report / "figures")
    ]
    for configuration in sorted({c for runs in points.values() for c in runs}):
        step_figures.append(
            plot_step_size.step_size_bias(
                pyplot, points, configuration, start, end, report / "figures"
            )
        )

    # ---- tables
    prefill = arguments.prefill_tokens_per_second
    sections = [
        ("throughput", "吞吐：各种口径并排", tables.throughput(points, start, end)),
        (
            "steps_over_time",
            "步/分钟随时间变化（有没有到稳态）",
            tables.steps_over_time(points),
        ),
        (
            "kv_and_recompute",
            "KV 装不装得下、缓存命中、可避免的重算",
            tables.kv_fit_and_recompute(points, start, end, prefill),
        ),
        (
            "littles_law",
            "闭环验算：步/分钟 = 程序数 ÷ 每步时间",
            tables.littles_law(points, start, end),
        ),
        (
            "arrivals_and_fairness",
            "程序怎么到达；第一批 vs 后来者；长时间等待",
            tables.arrivals_and_fairness(points),
        ),
        (
            "thunderagent_admission",
            "ThunderAgent：放进来多少 vs 引擎实际占用",
            tables.thunderagent_admission(points, start, end),
        ),
    ]
    for key, _title, rows in sections:
        tables.to_csv(rows, report / "tables" / f"{key}.csv")
    (report / "README.md").unlink(missing_ok=True)  # replaced by the Feishu document

    text = (
        "\n".join(
            build_document(
                experiment,
                points,
                first,
                start,
                end,
                prefill,
                step_figures,
                sections,
                incomplete,
                failures,
            )
        )
        + "\n"
    )
    if arguments.feishu:
        url = publish(experiment, text)
        print(
            f"[report] published {url} ({len(failures)} figure failures, "
            f"{len(incomplete)} incomplete runs)"
        )
    else:
        (report / "preview.md").write_text(text)
        print(f"[report] no --feishu: wrote {report / 'preview.md'}")


if __name__ == "__main__":
    main()
