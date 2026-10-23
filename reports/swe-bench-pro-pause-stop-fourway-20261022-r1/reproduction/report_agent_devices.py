"""Plot recorded per-card exporter observations; no network or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_run import BLUE, ORANGE, style

METRICS = {
    'npu_chip_info_utilization': 'AI Core utilization',
    'npu_chip_info_hbm_bandwidth_utilization': 'HBM bandwidth utilization',
    'npu_chip_info_cube_utilization': 'Cube utilization',
    'npu_chip_info_vector_utilization': 'Vector utilization',
}


def read_device_rows(path, origin, window):
    points = {metric: {} for metric in METRICS}
    ages = []
    seen = set()
    missing = 0
    for line in path.open():
        row = json.loads(line)
        stamps = {x['metric']['id']: float(x['value'][1]) for x in
                  row.get('source_timestamps', {}).get('data', {}).get('result', [])}
        for metric in METRICS:
            response = row.get(metric, {})
            if response.get('status') != 'success':
                missing += 1
                continue
            for item in response['data']['result']:
                card = item['metric']['id']
                timestamp, value = float(item['value'][0]), float(item['value'][1])
                if not 0 <= timestamp - origin < window:
                    continue
                if metric == 'npu_chip_info_utilization':
                    if card not in stamps:
                        missing += 1
                        continue
                    ages.append(timestamp - stamps[card])
                    # Actual scrape time is available only for this metric.
                    timestamp = stamps[card]
                    if not 0 <= timestamp - origin < window:
                        continue
                key = metric, card, timestamp
                if key not in seen:
                    points[metric].setdefault(card, []).append([timestamp - origin, value])
                    seen.add(key)
    return {'points': points, 'missing_responses': missing,
            'ai_core_sample_age_seconds': {'max': max(ages) if ages else None,
                                          'median': statistics.median(ages) if ages else None},
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--window-seconds', type=float, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(exist_ok=False, parents=True)
    data = {}
    for arm in ['native', 'controller']:
        root = args.run_dir / 'terminal-extract'
        arrivals = [json.loads(x.read_text())['epoch'] for x in (root / ('pair-1-' + arm)).glob('task-*/arrival.json')]
        data[arm] = read_device_rows(root / (arm + '-evidence') / 'device-observations-private.jsonl', min(arrivals), args.window_seconds)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.8), sharex=True)
    summary = {}
    for ax, (metric, label) in zip(axes.flat, METRICS.items()):
        summary[metric] = {}
        for arm, color in [('native', BLUE), ('controller', ORANGE)]:
            summary[metric][arm] = {}
            for index, (card, points) in enumerate(sorted(data[arm]['points'][metric].items())):
                points.sort()
                ax.plot([x[0] for x in points], [x[1] for x in points], color=color,
                        ls='-' if index == 0 else '--', lw=1.4, label=arm.title() + ' card ' + card)
                values = [x[1] for x in points]
                summary[metric][arm][card] = {'samples': len(values), 'sample_mean_native_units': statistics.mean(values),
                                             'min_native_units': min(values), 'max_native_units': max(values)}
        ax.set_title(label, loc='left')
        ax.set_ylabel('Exporter reading (native units)')
        ax.set_xlabel('Seconds since each group started')
        ax.set_xlim(0, args.window_seconds)
        style(ax, 'Exporter reading (native units)')
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('ENGINE / DEVICE | Physical TP2 cards, recorded utilization observations', fontsize=14)
    fig.text(0.5, 0.015, 'AI Core: unique actual scrape timestamps. Other metrics: query-observed points. Raw exporter units; no smoothing or zero-filling. Not a saturation proof.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(args.output_dir / ('engine-device.' + suffix), dpi=160)
    plt.close(fig)
    result = {'window_seconds': args.window_seconds, 'data': data, 'summary': summary,
              'scope': 'Native exporter units retained; sample means are not time-weighted; utilization is not kernel/communication profiling or saturation proof.'}
    (args.output_dir / 'device-data.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'figure': str(args.output_dir / 'engine-device.png'), 'summary': summary}))


if __name__ == '__main__':
    main()
