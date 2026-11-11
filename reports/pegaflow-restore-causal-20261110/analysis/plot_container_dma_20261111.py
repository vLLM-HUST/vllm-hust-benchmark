"""Plot reversible native interventions and isolated DMA preparation costs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--font', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding='utf-8'))
    plt.switch_backend('Agg')
    plt.rcParams.update(**{'font.family': FontProperties(fname=args.font).get_name(),
                           'font.size': 11, 'axes.spines.top': False,
                           'axes.spines.right': False, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    blue, orange, gray = '#2563eb', '#ea580c', '#b6c2d2'
    for axis, name, title in zip(axes[0],
                               ['container1111-native-shadow-toggle-device1',
                                'container1111-native-shadow-crossover-device0'],
                               ['A｜91 物理卡 7', 'B｜91 物理卡 2'], strict=True):
        run = next(v for k, v in data['native'].items()
                   if '/hust-zsx-91/' in k and k.endswith('/' + name))
        chosen = [r for r in run['restores'] if r['phase'] in ('stage-4', 'stage-5', 'stage-6')]
        for i, row in enumerate(chosen):
            residual, total = row['residual_restore_s'], row['complete_s']
            axis.bar(i, residual, color=blue if row['policy'] == 29 else orange, width=.58)
            axis.bar(i, total - residual, bottom=residual, color=gray, width=.58)
            axis.text(i, total + .18, f'{total:.3f}s', ha='center', fontweight='bold')
        axis.set(title=title + '：同一进程、相同源和目的地址', ylabel='一次恢复完成时间（秒）',
                 xticks=range(3), xticklabels=['DMA 预注册', '关闭预注册', '重新预注册'], ylim=(0, 12.7))
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    axis = axes[1, 0]
    for i, (host, device) in enumerate([('81', 0), ('81', 1), ('91', 7), ('91', 2)]):
        runs = [v for k, v in data['native'].items()
                if 'shadow' in k and v['host'] == 'hust-zsx-' + host
                and v['physical_device'] == device]
        for delta, registered, color, label in [(-.14, False, orange, '未预注册'),
                                                (.14, True, blue, '已预注册')]:
            values = [r['complete_s'] for run in runs for r in run['restores']
                      if r['condition'] == ('shadow_dma_registered' if registered
                                           else 'shadow_unregistered_after')
                      or (not registered and r['condition'] == 'shadow_unregistered_before')]
            jitter = np.linspace(-.055, .055, len(values))
            axis.scatter(i + delta + jitter, values, s=35, color=color, alpha=.8,
                         label=label if i == 0 else None)
    axis.set(title='C｜独立进程复核：固定影子缓冲区的逐次结果', ylabel='恢复完成时间（秒，对数轴）',
             xticks=range(4), xticklabels=['81 卡0', '81 卡1', '91 卡7', '91 卡2'], yscale='log')
    axis.grid(axis='y', alpha=.2)
    axis.legend(frameon=False)
    axis.text(.01, -.18, '含开关本次成本；首次分配/5GiB 初始化及最终释放单列，不入此图。',
              transform=axis.transAxes, fontsize=10)
    axis = axes[1, 1]
    for i, (host, trial) in enumerate([('81', 0), ('91', 0), ('81', 1), ('91', 1)]):
        run = next(v for k, v in data['controls'].items()
                   if '/hust-zsx-' + host + '/' in k and k.endswith('/container1111-dma-reuse'))
        row = next(r for r in run['cases'] if r['trial'] == trial)
        bottom = 0
        for key, color, label in [('create_ns', gray, '准备 DMA 地址'),
                                  ('submit_ns', blue, '同步提交及拷贝'),
                                  ('destroy_ns', orange, '释放 DMA 地址')]:
            value = row[key] / 1e9
            axis.bar(i, value, bottom=bottom, color=color, width=.6,
                     label=label if i == 0 else None)
            bottom += value
        axis.text(i, bottom + .018, f"{row['wall_ns']/1e9:.3f}s", ha='center')
    axis.set(title='D｜独立 HAL 对照：1GiB，4,096 次同步拷贝', ylabel='循环耗时（秒）',
             xticks=range(4), xticklabels=['81 即时准备', '91 即时准备', '81 复用准备', '91 复用准备'],
             ylim=(0, .92))
    axis.legend(frameon=False, fontsize=9, loc='upper right')
    axis.grid(axis='y', alpha=.15)
    axis.set_axisbelow(True)
    axis.text(.01, -.18, '复用组的一次性准备/释放：81 0.118/0.077s；91 0.312/0.347s。',
              transform=axis.transAxes, fontsize=10)
    fig.suptitle('本地 KV 恢复慢尾：DMA 准备路径的可逆干预', fontsize=21, x=.06, ha='left', y=.98)
    fig.text(.06, .935, '原生恢复：3.773GiB、15,456 描述符；每次全部描述符首字标记通过。灰色段为注册/注销成本。',
             fontsize=11, color='#475569')
    fig.subplots_adjust(top=.87, bottom=.1, left=.07, right=.98, hspace=.46, wspace=.22)
    fig.savefig(args.output, dpi=160, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
