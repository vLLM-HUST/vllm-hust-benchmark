"""Layered figures and embedded Chinese report from finite-work originals only."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_fixed_work import analyze
from agent_report_data import parse_metrics
from plot_run import BLUE, ORANGE, GRID, style
from report_agent_devices import METRICS, read_device_rows
from write_agent_layered_report import number

COLORS = {'native': BLUE, 'controller': ORANGE, 'thunderagent_original': '#387f62',
          'controller_original': ORANGE, 'controller_corrected': '#8558a1'}
LABELS = {'native': 'Native', 'controller': 'Controller',
          'thunderagent_original': 'ThunderAgent original',
          'controller_original': 'Controller original', 'controller_corrected': 'Controller corrected'}


def cdf(axis, values, name):
    if values:axis.step([0, *sorted(values)], range(len(values) + 1), where='post',
                        color=COLORS[name], label=LABELS.get(name, name))


def build(originals, output):
    originals, output = Path(originals).resolve(), Path(output).resolve()
    if output.is_relative_to(originals) or originals.is_relative_to(output):
        raise ValueError('Report output must be separate from originals')
    data = analyze(originals)
    output.mkdir(parents=True, exist_ok=False)
    groups = data['groups']
    raw = json.loads((originals / 'ablation-result-private.json').read_text())
    sections, figures, markdown_sections = [], [], []
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'savefig.facecolor': 'white'})

    def save(figure, filename, title, caption):
        figure.tight_layout()
        for suffix in ['png', 'svg', 'pdf']:figure.savefig(output / (filename + '.' + suffix), dpi=150)
        plt.close(figure)
        figures.append(filename)
        markdown_sections.extend(['## ' + title, '', f'![{title}]({filename}.png)', '', caption, ''])
        encoded = base64.b64encode((output / (filename + '.png')).read_bytes()).decode()
        sections.append('<h2>' + html.escape(title) + '</h2><figure><img alt="' + html.escape(title)
                        + '" src="data:image/png;base64,' + encoded + '"><figcaption>'
                        + html.escape(caption) + '</figcaption></figure>')

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    sampled, devices = {}, {}
    for block in raw['blocks']:
        name, index = block['group'], block['index']
        group = groups[name]
        cdf(axes[0], [row['admission_seconds'] for row in group['request_rows'] if 'admission_seconds' in row], name)
        path = originals / f'block-{index}-observations' / 'observations-private.jsonl'
        sampled[name] = []
        if path.exists():
            data['source_sha256'][str(path.relative_to(originals))] = hashlib.sha256(path.read_bytes()).hexdigest()
            for line in path.read_text().splitlines():
                row = json.loads(line)
                row['relative_seconds'] = row['epoch'] - block['epoch_start']
                sampled[name].append(row)
            points = [(row['relative_seconds'], len(row['scheduler']['waiting'])) for row in sampled[name]
                      if 'scheduler' in row and 'waiting' in row['scheduler']]
            if points:axes[1].plot(*zip(*points), color=COLORS[name], label=LABELS.get(name, name))
        path = originals / f'block-{index}-observations' / 'device-observations-private.jsonl'
        if path.exists():
            devices[name] = read_device_rows(path, block['epoch_start'], block.get('makespan_seconds') or 0)
            data['source_sha256'][str(path.relative_to(originals))] = devices[name]['source_sha256']
    for axis, ylabel in zip(axes, ['Admitted requests (count)', 'Waiting programs (count)']):
        style(axis, ylabel);axis.set_xlabel('Seconds')
        if axis.get_legend_handles_labels()[0]:axis.legend(frameon=False)
    if not any(sampled.values()):
        axes[1].text(.5, .5, 'Time series not recorded', ha='center', transform=axes[1].transAxes)
        axes[1].set_axis_off()
    if not any(group['matched_requests'] for group in groups.values()):
        axes[0].text(.5, .5, 'Admission phases not recorded', ha='center', transform=axes[0].transAxes)
        axes[0].set_axis_off()
    fig.suptitle('CONTROLLER | Matched admission time and sampled queue')
    save(fig, 'controller', '第一层：Controller——等待移到了哪里？',
         '左图按请求关联准入开始和结束，右图为有记录时的待执行会话采样。正常排队不是故障；快照不代表每个调度时刻的原子决策。缺失的时间序列明确保留缺失。')

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    width = min(.32, .8 / len(groups))
    center = width * (len(groups) - 1) / 2
    for offset, (name, group) in enumerate(groups.items()):
        counts = group['engine_counters']
        values = [counts.get(key) for key in ['request_prefill_kv_computed_tokens_sum', 'prefix_cache_hits_total']]
        axes[0].bar([x + offset * width for x in range(2)],
                    [value / 1e6 if value is not None else float('nan') for value in values],
                    width=width, color=COLORS[name], label=LABELS.get(name, name))
        values = [group['engine_histograms'].get(key, {}).get('mean')
                  for key in ['request_queue_time_seconds', 'request_prefill_time_seconds', 'request_decode_time_seconds']]
        axes[1].bar([x + offset * width for x in range(3)],
                    [value if value is not None else float('nan') for value in values],
                    width=width, color=COLORS[name], label=LABELS.get(name, name))
    axes[0].set_xticks([center, 1 + center], ['Input actually computed', 'Prefix tokens found'])
    axes[1].set_xticks([center, 1 + center, 2 + center], ['Engine queue', 'Prefill', 'Decode'])
    style(axes[0], 'Tokens (millions, complete group)');style(axes[1], 'Mean seconds / finished request')
    for axis in axes:axis.legend(frameon=False)
    if not any(group['engine_counters'] for group in groups.values()):
        for axis in axes:
            axis.text(.5, .5, 'Engine counters not recorded', ha='center', transform=axis.transAxes)
            axis.set_axis_off()
    fig.suptitle('ENGINE | Prefix reuse, actual input computation and engine phases')
    save(fig, 'engine', '第二层：引擎——是否少算了历史？',
         '各组输入和输出一致后，比较实际输入计算量、缓存命中与引擎阶段均值。引擎排队不含全部Controller等待；计数器和直方图是组内增量，不冒充逐请求kernel关联。')

    if any(points for device in devices.values() for cards in device['points'].values() for points in cards.values()):
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.5))
        for axis, (metric, label) in zip(axes.flat, METRICS.items()):
            for name, device in devices.items():
                for index, (card, points) in enumerate(sorted(device['points'][metric].items())):
                    points.sort()
                    if points:axis.plot(*zip(*points), color=COLORS[name], ls='-' if index == 0 else '--',
                                        label=LABELS.get(name, name) + ' card ' + card)
            style(axis, 'Exporter reading (native units)');axis.set_title(label, loc='left')
            axis.set_xlabel('Seconds since each group started');axis.legend(frameon=False, fontsize=8)
        fig.suptitle('ENGINE / DEVICE | Physical TP2 cards, recorded observations')
        save(fig, 'device', '第二层补充：两张卡的实际设备观测',
             'AI Core使用真实scrape时间去重；其他指标保留查询观测点。单位沿用exporter，不猜换算，不补零。利用率、缓存池占用和总HBM不是同一概念，也不足以单独证明算力或带宽瓶颈。')
    else:
        sections.append('<p>设备限制：本轮没有完整设备采样，不以旧实验或缺失值补齐。</p>')
    data['device_observations'] = devices

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    for name, group in groups.items():
        cdf(axes[0], group['program_completion_seconds'], name)
        cdf(axes[1], group['program_first_reply_seconds'], name)
    for axis, title in zip(axes, ['All rounds completed', 'First normal reply received']):
        axis.set_title(title, loc='left');axis.set_xlabel('Seconds since finite-batch arrival')
        style(axis, 'Programs (count)');axis.legend(frameon=False)
        axis.set_xlim(0, max((group['observed_return_span_seconds'] or 0 for group in groups.values()), default=0) * 1.02 or 1)
    fig.suptitle('PROGRAM | Complete-work progress versus first reply')
    save(fig, 'program', '第三层：会话——完整工作是否更早完成？',
         '左图是完成全部冻结轮次的会话数，是本次性能目标；右图只描述第一次回复。晚拿到首回复不自动算故障。会话开始时间相同，不能从队列中删除晚启动的会话。')

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    for index, (name, group) in enumerate(groups.items()):
        cdf(axes[0], [row['elapsed'] for row in group['request_rows']], name)
        left = 0
        for key, color, label in [('admission_seconds', '#52514e', 'Controller admission'),
                                  ('backend_seconds', '#9aa4b2', 'Backend queue + execution'),
                                  ('post_backend_seconds', '#dce2e8', 'Proxy response')]:
            value = group['matched_phase_means'][key]
            if value is not None:axes[1].barh(index, value, left=left, color=color, label=label if index == 0 else None);left += value
    axes[0].set_xlabel('Client response latency (seconds)');style(axes[0], 'Returned requests (count)');axes[0].legend(frameon=False)
    axes[1].set_yticks(range(len(groups)), [LABELS.get(name, name) for name in groups]);axes[1].set_xlabel('Mean matched-proxy time (seconds)')
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(frameon=False, fontsize=8)
    else:
        axes[1].text(.5, .5, 'Proxy phases not recorded', ha='center', transform=axes[1].transAxes)
        axes[1].set_axis_off()
    axes[1].grid(axis='x', color=GRID)
    fig.suptitle('REQUEST | Client tails and matched proxy phases')
    save(fig, 'request', '第四层：请求——平均改善是否伴随长尾代价？',
         '左图保留全部正常返回的客户端延迟，不能用均值掩盖长尾。右图灰色分段表示阶段，只使用成功匹配的同一请求、同一代理时钟；后端时间包含引擎排队、执行与传输，不是纯计算时间。失败与未完成数量另列。')

    reduction = data['makespan_reduction_fraction']
    verdict = ('相同完整固定工作耗时下降' + number(reduction * 100) + '%' if reduction is not None and reduction >= 0 else
               '相同完整固定工作耗时增加' + number(-reduction * 100) + '%' if reduction is not None else
               '多组相同完整固定工作已完成，分别核对接入差异与修正效果' if data['comparable_complete_work'] else '完整可比条件未通过，不计算加速')
    lines = ['# 完整固定工作对照报告', '', verdict, '',
             '这是冻结历史的实机回放，不是真实Agent解题完成；一次配对不证明稳定或普遍收益。各组身份见图例，原版与修正候选分开。', '',
             '请求链路：会话按依赖发出请求 → Controller准入或等待 → 引擎排队与执行 → 客户端收到回复 → 下一轮。', '']
    plan = json.loads((originals / 'fixed-work-plan-private.json').read_text())
    lines.append(f'每组{plan["batch_size"]}个会话、各{plan["rounds_per_program"]}轮、每次输出{plan["output_tokens"]} token；整批同时到达，客户端不设并发上限。')
    for name, group in groups.items():
        lines.append(f'{LABELS.get(name, name)}：完整会话{group["complete_programs"]}，返回{group["normal_returns"]}/{group["expected_returns"]}，错误{group["errors"]}；完整工作耗时{number(group["complete_work_makespan_seconds"])}秒，客户端p95 {number(group["client_p95_seconds"])}秒。')
        actions = group['controller_action_events']
        if actions is not None:
            lines.append(f'标记/暂停/恢复事件次数：{actions["Marked"]}/{actions["Paused"]}/{actions["Resumed"]}；'
                         '这是所存服务日志内的事件次数，不是独立program数，也不自动等同于测量窗口事件。')
    for comparison in data['comparisons'].values():
        delta = comparison['makespan_reduction_fraction']
        if delta is not None:
            lines.append(f'{LABELS.get(comparison["candidate"], comparison["candidate"])} 相对 '
                         f'{LABELS.get(comparison["baseline"], comparison["baseline"])}：'
                         f'完整工作耗时{"下降" if delta >= 0 else "增加"}{number(abs(delta) * 100)}%。')
    lines.append('未记录逐请求代理阶段的组保留缺失，不从引擎平均排队推算准入时间。')
    limits = ['一次配对回放不能证明稳定或普遍加速。', '后端时间包含引擎排队、执行与传输，不是纯kernel时间。',
              '设备和调度器快照不是原子决策，缺失数据不代表零使用。', '即使完整工作耗时改善，也必须保留请求排队长尾。']
    (output / 'report.md').write_text('\n'.join(lines + [''] + markdown_sections + ['## 证据限制', ''] + limits) + '\n')
    intro = '<h1>完整固定工作对照报告</h1>' + ''.join('<p>' + html.escape(line) + '</p>' for line in lines[2:] if line and not line.startswith('#'))
    css = 'body{margin:40px auto;max-width:1080px;padding:0 24px;color:#17212d;font:17px/1.85 system-ui}h1{font-size:32px;line-height:1.45}h2{margin-top:52px;border-top:1px solid #dce2e8;padding-top:22px;font-size:24px}figure{margin:28px 0}img{width:100%;height:auto}figcaption{color:#52616e;font-size:14px}p{margin:18px 0}'
    (output / 'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整固定工作对照</title><style>' + css + '</style>' + intro + ''.join(sections) + '<h2>证据限制</h2>' + ''.join('<p>' + html.escape(value) + '</p>' for value in limits) + '</html>')
    (output / 'data.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    with (output / 'requests.csv').open('w', newline='') as stream:
        writer = csv.writer(stream);writer.writerow(['group', 'request_id', 'prompt_tokens', 'output_tokens', 'elapsed_seconds', 'admission_seconds', 'backend_seconds'])
        for name, group in groups.items():
            for row in group['request_rows']:writer.writerow([name, row['request_id'], row['usage']['prompt_tokens'], row['usage']['completion_tokens'], row['elapsed'], row.get('admission_seconds'), row.get('backend_seconds')])
    sources = [Path(__file__).with_name(name) for name in ['report_fixed_work.py', 'analyze_fixed_work.py',
               'plot_run.py', 'report_agent_devices.py', 'agent_report_data.py', 'write_agent_layered_report.py']]
    manifest = {'figures': figures, 'generator_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                'files': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir() if path.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals', type=Path, required=True);parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args();data = build(args.originals, args.output_dir)
    print(json.dumps({'report': str(args.output_dir / 'report.html'), 'same_complete_work': data['same_complete_work']}))


if __name__ == '__main__':main()
