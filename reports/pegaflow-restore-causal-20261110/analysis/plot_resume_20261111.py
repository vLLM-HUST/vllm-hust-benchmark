"""Plot accepted versus kernel-forwarded entries from independently timed records."""
import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary')
    parser.add_argument('--output', required=True)
    parser.add_argument('--font', default='C:/Windows/Fonts/msyh.ttc')
    args = parser.parse_args()
    data = json.load(open(args.summary))
    font = FontProperties(fname=args.font)
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                         'font.size': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, host, name in zip(axes, ['81', '91'], ['正常路径', '慢路径']):
        key = f'hust-zsx-{host}/resume1111-native-kernel-a-device' + (
            '0' if host == '81' else '1')
        item = data['native'][key]
        curve = item['curves']['stage-3']
        x, y = zip(*curve['points'])
        ax.plot(x, y, color='#e87a2b', label='用户接口返回：驱动已接收')
        points = [p for p in item['kernel_points'] if p['phase'] == 'stage-3']
        ax.plot([p['relative_s'] for p in points],
                [p['forwarded_entries'] - p['phase_base'] for p in points],
                'o-', color='#2465c2', label='内核通道：已下发到硬件')
        ax.set_title(f'{host} 节点 · {name} · 完成 {curve["complete_s"]:.3f} 秒', pad=14)
        ax.set_xlabel('本次恢复开始后（秒）')
        ax.set_ylim(0, 16500)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(0, .85 if host == '81' else 10.5)
    axes[0].set_ylabel('本次恢复累计任务数（共 15,456 个）')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[1].text(.98, .11, '后半段：驱动有积压，硬件队列采样为空\n'
                '下一任务在驱动中驻留约 3.9～5.0 秒',
                transform=axes[1].transAxes, ha='right', fontsize=10)
    fig.suptitle('相同 3.773GiB 恢复：慢在驱动接收之后、硬件执行之前', fontsize=16)
    fig.text(.5, .02, '两图横轴尺度不同。蓝点为快照读取时刻，完成后快照晚于恢复返回；'
             '连接线辅助观察。“内核下发”不等于 stream 完成。',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .06, 1, .93))
    fig.savefig(args.output, dpi=170)


if __name__ == '__main__':
    main()
