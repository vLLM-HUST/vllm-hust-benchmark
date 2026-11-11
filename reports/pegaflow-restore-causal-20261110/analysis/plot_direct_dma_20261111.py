"""Plot same-address direct-HAL diagnostics, including preparation costs."""

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
    plt.rcParams.update({'font.family': FontProperties(fname=args.font).get_name(),
                         'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.7))
    for i, (host, device) in enumerate([('81', 0), ('81', 1), ('91', 2), ('91', 7)]):
        key, run = next((k, v) for k, v in data['runs'].items()
                        if 'paired' in k and v['host'] == 'hust-zsx-' + host
                        and v['physical_device'] == device)
        for offset, mode, color, label in [(-.15, 29, '#2563eb', 'CANN 异步'),
                                            (.15, 32, '#16a34a', '预准备描述符＋HAL 同步')]:
            values = [r['residual_s'] for r in run['restores']
                      if r['policy'] == mode and not r['initial_shadow']]
            left.scatter(i + offset + np.linspace(-.05, .05, len(values)), values,
                         s=36, alpha=.85, color=color, label=label if i == 0 else None)
        ratios = [r['direct_over_async'] for r in data['pairs'] if r['run'] == key]
        right.scatter(i + np.linspace(-.09, .09, len(ratios)), ratios, s=55,
                      color='#475569' if host == '81' else '#dc2626')
    labels = ['81 卡0', '81 卡1', '91 卡2', '91 卡7']
    left.set(title='A｜相同源和目的地址：逐次残余恢复', ylabel='恢复耗时（秒）',
             xticks=range(4), xticklabels=labels, ylim=(0, 2.55))
    left.legend(frameon=False, fontsize=10, loc='upper left')
    left.text(.02, -.2, '扣除实际记录的注册、描述符准备及释放；每组 8 次。',
              transform=left.transAxes, fontsize=10)
    right.axhline(1, color='#64748b', linestyle='--', linewidth=1)
    right.set(title='B｜完整配对时间：直接提交 ÷ CANN 异步', ylabel='每组两次完整恢复均值之比',
              xticks=range(4), xticklabels=labels, ylim=(0, 2.15))
    right.text(.02, -.2, '包含描述符准备/释放成本；比值＞1 表示直接提交更慢。',
               transform=right.transAxes, fontsize=10)
    for axis in (left, right):
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle('同地址旁路对照：动态准备与异步推进的开销', fontsize=19, x=.07, ha='left')
    fig.text(.07, .88, '每次 3.773GiB、15,456 个标记通过；两组均使用同一已预注册 CPU 影子缓冲区。',
             fontsize=11, color='#475569')
    fig.subplots_adjust(top=.75, bottom=.23, left=.07, right=.97, wspace=.25)
    fig.savefig(args.output, dpi=160, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
