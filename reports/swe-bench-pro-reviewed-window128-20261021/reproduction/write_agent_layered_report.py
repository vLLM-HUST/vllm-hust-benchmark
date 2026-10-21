"""Compose a readable report from canonical derived evidence; no new measurements."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_run import BLUE, ORANGE, GRID, style


def build(run, layered, devices, audit_path, output):
    output.mkdir(parents=True, exist_ok=False)
    figures = output / 'figures'
    figures.mkdir()
    data = json.loads((layered / 'data.json').read_text())
    progress = json.loads((run / 'program-progress-summary-private.json').read_text())
    terminal = json.loads((run / 'terminal-summary-private.json').read_text())
    audit = json.loads(audit_path.read_text())
    window = data['window_seconds']
    whole = f'0-{window:g}'
    n, c = [data['arms'][a]['phases'][whole] for a in ['native', 'controller']]
    agents = data['agent_count_per_arm']
    budgets = data['arms']['native']['settings']
    runtime_path = run / 'terminal-extract/native-evidence/before-runtime-private.json'
    engine = json.loads(runtime_path.read_text())['g7_shared_runtime_v1']['results'][0]['effective_engine']
    for path in layered.glob('*.png'):
        shutil.copy2(path, figures / path.name)
    for suffix in ['png', 'svg', 'pdf']:
        shutil.copy2(devices / ('engine-device.' + suffix), figures / ('engine-device.' + suffix))
    values = sorted(audit['programs'].values(), key=lambda x: x['task'])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5), gridspec_kw={'width_ratios': [1, 2]})
    with (layered / 'normal-client-returns.csv').open() as stream:
        counts = {'native': {}, 'controller': {}}
        for row in csv.DictReader(stream):
            arm, task = row['arm'], row['program']
            counts[arm][task] = counts[arm].get(task, 0) + 1
    for arm, color in [('native', BLUE), ('controller', ORANGE)]:
        reply_counts = sorted([*counts[arm].values(), *([0] * (agents - len(counts[arm])))])
        axes[0].plot(range(1, agents + 1), reply_counts, color=color, label=arm.title())
    axes[0].set_xlabel('Agents sorted by their own reply count')
    style(axes[0], 'Normally returned model calls / Agent')
    axes[0].legend(frameon=False)
    for i, row in enumerate(values):
        if row['first_seen_t'] is not None:
            axes[1].barh(i, row['last_seen_t'] - row['first_seen_t'], left=row['first_seen_t'], color=ORANGE)
    style(axes[1], 'Agents with no normal reply')
    axes[1].set_yticks(range(len(values)), [x['task'] for x in values], fontsize=8)
    axes[1].set_xlim(0, window)
    axes[1].set_xlabel('Seconds since Controller group started')
    axes[1].set_title('All recorded states: paused; no assigned backend', loc='left', fontsize=10)
    fig.suptitle('CONTROLLER / PROGRAM | First-call access and progress distribution', fontsize=13)
    fig.text(.5, .02, 'Bars span recorded queue observations; sequential HTTP snapshots are not atomic scheduler decisions. Missing initial time is not filled.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .94))
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(figures / ('controller-first-call.' + suffix), dpi=160)
    plt.close(fig)
    ratio = c['client']['output_tokens'] / n['client']['output_tokens'] - 1
    zero = progress['controller']['zero_normal_return_agents']
    seconds = min((x['queued_seconds'] for x in values), default=0)
    pending_n, pending_c = [data['arms'][a]['pending'] for a in ['native', 'controller']]
    lines = [
        '# SWE-bench Pro真实Agent对照：配置审查后的结果与问题', '',
        '## 结论先说清楚', '',
        f'本次比较了Native组和Controller组，各{agents}个Agent，整批同时到达，不在客户端限并发；每组观察{window / 60:g}分钟。模型及两组预算一致，Controller组额外启用已明确标注的暂停容量修正。', '',
        f'Controller组在窗口内返回了更多输出token（多{ratio * 100:.1f}%），已正常返回调用的平均延迟也更低；但它有{zero}个Agent连第一次模型调用都没有正常返回。两组最终都没有完成提交。因此本轮不能宣布Controller整体更好，更不能把输出量变化当作相同工作量加速。', '',
        '读下面的图时，蓝色表示Native，橙色表示Controller。先看任务是否推进，再沿请求链路解释等待转移到了哪里。', '',
        '## 系统链路与实验边界', '',
        'Agent产生第一次模型请求 → Controller决定放行或等待 → vLLM引擎排队、处理输入、生成输出 → 请求返回 → Agent执行工具并继续下一轮。Controller等待、引擎排队、模型计算和工具执行是不同阶段。', '',
        f'本轮两组各{agents}个独立Agent；每个Agent保留多轮模型与工具依赖。两组模型采用相同{engine["max_num_batched_tokens"]} token调度预算，async_scheduling={engine["async_scheduling"]}，上下文{engine["max_model_len"]} token；Agent上限{budgets["step_limit"]}步、{budgets["wall_budget_seconds"]}秒、单次输出{budgets["max_tokens"]} token、工具{budgets["tool_timeout_seconds"]}秒。一次Native→Controller顺序比较，不足以证明稳定因果收益。', '',
        '## 第一层：Controller——哪些任务在引擎外等候？', '',
        f'最需要关注的是{zero}个没有正常回复的Agent。已有记录中，它们约{seconds:.1f}秒持续出现在Controller等待队列，状态为paused，尚无分配后端，step_count始终为1。这里的step_count是代理收到请求后的计数，不代表第一次模型调用已经执行成功。', '',
        '![首次调用准入与全部Agent进展分布](figures/controller-first-call.png)', '',
        '左图把全部Agent按正常回复次数排序，避免平均数掩盖完全没推进的任务；右图只展示没有正常回复的Agent及其被观测到的排队时段。记录开始前的缺口没有补零，也没有当作完整逐事件追踪。', '',
        '![Controller状态与准入队列](figures/controller.png)', '',
        '这张图展示Controller登记的活动、暂停和等待状态。活动program数不是引擎实际执行数；逐次动作采样文件缺失的面板保持缺失，不能解释为没有暂停或恢复。', '',
        '固定源码的恢复策略优先选择已经进入多轮的请求，新任务优先级较低；兜底放行等待是1800秒，超过本轮Agent的1200秒总时限。这套规则没有保证新任务在本轮窗口内得到机会。快照重建仍需进一步核对每次选择，不能把非原子快照说成每个调度时刻的完整判断。', '',
        '## 第二层：引擎和设备——队列减少，是否真的处理得更好？', '',
        '![引擎队列、缓存池和输入输出处理](figures/engine-load.png)', '',
        f'Native的引擎内部平均排队时间为{n["histograms"]["request_queue_time_seconds"]["mean"]:.2f}秒，Controller为{c["histograms"]["request_queue_time_seconds"]["mean"]:.2f}秒。引擎里的队列明显变小，但第一层表明部分等待留在了Controller，不能把“引擎不排队”理解为“所有任务都不等”。', '',
        '缓存图表示每个rank缓存池的占用，不是设备总HBM。输入计算、输出生成和缓存命中分别画在各自面板，不能互相替代。', '',
        '![引擎输入处理、生成阶段与批次观测](figures/engine-latency-and-batches.png)', '',
        '![引擎首token和每token耗时](figures/engine-token-latency.png)', '',
        '上述阶段延迟来自各窗口内直方图sum/count增量；不是逐kernel耗时。分位桶只给区间，iteration指标也尚未被证明等同每一次真实调度步。', '',
        '![两张物理NPU卡的设备观测](figures/engine-device.png)', '',
        '设备图来自本轮实际两张卡的记录，不用缓存占用推测设备利用率。AI Core用实际scrape时间去重，其他指标保留查询时观测点；纵轴保留exporter原生读数，不猜测未核实的单位。这些曲线没有提供kernel或通信profile，仍不能据此宣称算力、内存带宽或通信已经饱和。', '',
        '## 第三层：Agent——每个任务是否都在推进？', '',
        '![全部Agent的推进过程和时间分配](figures/program.png)', '',
        f'Native每个Agent获得{progress["native"]["min_normal_returns_per_agent"]}–{progress["native"]["max_normal_returns_per_agent"]}次正常回复；Controller为{progress["controller"]["min_normal_returns_per_agent"]}–{progress["controller"]["max_normal_returns_per_agent"]}次，其中{zero}个为0。它反映出两组进展分布差异很大。Controller的平均返回延迟不能代表没有获得返回的这些Agent。', '',
        f'终态原件显示，两组各{agents}个Agent均以RetryError退出，完整异常均包含总时限到达。评分保持未决，不填成0分。任务完成数不能被引擎输出量替代。', '',
        '## 第四层：请求——调用者实际拿到了多少，等了多久？', '',
        '![正常返回数量、输出吞吐和调用延迟](figures/request.png)', '',
        f'Native正常返回{n["client"]["returns"]}次、输出{n["client"]["output_tokens"]} token；Controller正常返回{c["client"]["returns"]}次、输出{c["client"]["output_tokens"]} token。Controller返回次数更少、输出量更多，说明实际回复长度与后续生成轨迹不同，不能直接解释为同工作量完成得更快。', '',
        f'正常返回调用的平均端到端延迟为{n["client"]["mean_latency_s"]:.2f}/{c["client"]["mean_latency_s"]:.2f}秒，p95为{n["client"]["p95_latency_s"]:.2f}/{c["client"]["p95_latency_s"]:.2f}秒（Native/Controller）。窗口末各{pending_n}/{pending_c}个未返回调用不在正常返回延迟分布中。', '',
        '![不同阶段的平均延迟与长尾](figures/request-latency-by-interval.png)', '',
        '要同时看前段、后段和完整窗口。总体均值下降不代表后段长尾没有恶化，也不代表未获得第一次返回的任务被照顾到了。', '',
        '## 当前判读和下一步', '',
        '本轮观察到了更多输出交付、更小的引擎内部队列，以及更不均衡的Agent进展；没有证明完整批次完成更快。下一步先用已有记录核对新任务准入和恢复选择，并把首轮推进、等待上界、失败归档、报告交付纳入标准工作流门禁，再考虑新实验。', '',
        '两组十分钟现场UI截图未按时取得，缺口明确保留；自动状态和设备记录仍在原包。逐请求client→proxy→engine没有完整一一关联，原子选择理由和kernel profile仍缺失。', '',
        '## 复算与来源', '',
        '本文数字来自规范四层报告data.json、normal-client-returns.csv、逐Agent终态、零回复审计和设备适配器；生成脚本不启动任何推理或配置变更。来源哈希、图文件哈希和生成入口记录在manifest.json。', '',
    ]
    markdown = '\n'.join(lines)
    (output / 'report.md').write_text(markdown)
    # Keep the same prose and hierarchy in a standalone, directly readable HTML.
    blocks = []
    for line in lines:
        if line.startswith('!['):
            label, target = line[2:].split('](', 1)
            raw = (output / target[:-1]).read_bytes()
            blocks.append('<figure><img alt="' + html.escape(label) + '" src="data:image/png;base64,' + base64.b64encode(raw).decode() + '"><figcaption>' + html.escape(label) + '</figcaption></figure>')
        elif line.startswith('## '):
            blocks.append('<h2>' + html.escape(line[3:]) + '</h2>')
        elif line.startswith('# '):
            blocks.append('<h1>' + html.escape(line[2:]) + '</h1>')
        elif line:
            blocks.append('<p>' + html.escape(line) + '</p>')
    (output / 'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实Agent分层实验报告</title><style>body{margin:40px auto;max-width:1080px;padding:0 24px;color:#17212d;font:17px/1.85 system-ui}h1{font-size:32px;line-height:1.45}h2{margin-top:52px;border-top:1px solid #dce2e8;padding-top:22px;font-size:24px}figure{margin:28px 0}img{width:100%;height:auto}figcaption{color:#52616e;font-size:14px}p{margin:18px 0}@media print{h2{break-before:page}figure{break-inside:avoid}}</style>' + ''.join(blocks) + '</html>')
    source_paths = [layered / 'data.json', layered / 'normal-client-returns.csv', audit_path,
                    devices / 'device-data.json', runtime_path, run / 'terminal-summary-private.json',
                    run / 'program-progress-summary-private.json']
    manifest = {'generator': 'benchmarks/write_agent_layered_report.py', 'new_inference_requests': 0,
                'layers': ['controller', 'engine', 'program', 'request'],
                'source_sha256': {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                'output_sha256': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'report': str(output / 'report.md'), 'html': str(output / 'report.html'), 'figures': len(list(figures.glob('*.png')))}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ['run-dir', 'layered-dir', 'devices-dir', 'audit', 'output-dir']:
        p.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args()
    build(a.run_dir, a.layered_dir, a.devices_dir, a.audit, a.output_dir)


if __name__ == '__main__':
    main()
