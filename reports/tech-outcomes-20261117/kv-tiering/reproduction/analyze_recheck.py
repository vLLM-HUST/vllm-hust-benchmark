"""Audit all finite-workload arms, including negative and incomplete results."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def percentile(numbers, q):
    numbers = sorted(numbers)
    if not numbers:
        return None
    position = (len(numbers) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(numbers) - 1)
    return numbers[lower] + (numbers[upper] - numbers[lower]) * (position - lower)


def metric_sum(metrics, name):
    return sum(value for key, value in metrics.items() if key.split('{')[0] == name)


def analyze(root):
    result = {'scope': 'full-HBM specialized diagnostic; not official dataset score or live-agent quality',
        'arms': {}, 'pairs': {}, 'all_complete_inputs_identical': False}
    signatures = {}
    counters = ('vllm:prefix_cache_hits_total', 'vllm:external_prefix_cache_hits_total',
        'vllm:request_prefill_kv_computed_tokens_sum', 'vllm:prompt_tokens_total',
        'vllm:kv_offload_store_bytes_total', 'vllm:kv_offload_load_bytes_total',
        'vllm:num_preemptions_total')
    for path in sorted(root.glob('*-client/requests.jsonl')):
        directory = path.parent
        label = directory.name[:-7]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows:
            continue
        phases = [json.loads(p.read_text()) for p in sorted(directory.glob('*-summary.json'))]
        plan = json.loads((directory / 'plan.json').read_text())
        expected = sum(len(p['requests']) for p in plan['phases'])
        wall = sum(p['wall_s'] for p in phases)
        output = sum(p['output_tokens'] for p in phases)
        signatures[label] = {r['key']: [r['input_sha256'], r['usage']['prompt_tokens'],
            r['usage']['completion_tokens']] for r in rows}
        arm = {'requests': len(rows), 'expected_requests': expected,
            'completed_all': (directory / 'complete.json').exists() and len(rows) == expected,
            'invalid': sum(not r['valid'] for r in rows),
            'usage_prompt_mismatches': sum(r['usage']['prompt_tokens'] != r.get('prompt_tokens', r['usage']['prompt_tokens']) for r in rows),
            'output_tokens': output,
            'output_tokens_received_so_far': sum(r['usage']['completion_tokens'] for r in rows),
            'logical_prompt_tokens': sum(r['usage']['prompt_tokens'] for r in rows),
            'phase_wall_s_sum': wall, 'output_tok_s': output / wall if wall else None,
            'ttft_mean_s': statistics.mean(r['ttft_s'] for r in rows),
            'ttft_p50_s': percentile([r['ttft_s'] for r in rows], .5),
            'ttft_p95_s': percentile([r['ttft_s'] for r in rows], .95),
            'ttft_p99_s': percentile([r['ttft_s'] for r in rows], .99),
            'tpot_mean_s': statistics.mean(r['tpot_s'] for r in rows),
            'tpot_p95_s': percentile([r['tpot_s'] for r in rows], .95),
            'golden_count': sum('expected' in r for r in rows),
            'golden_passed': sum(r['output_text'].strip().splitlines()[0].strip() == r['expected'] for r in rows if 'expected' in r),
            'counters': {name: sum(metric_sum(p['metrics_delta'], name) for p in phases) for name in counters},
            'prompt_by_source_tokens': {},
            'phases': {}}
        for phase in phases:
            for key, value in phase['metrics_delta'].items():
                if key.split('{')[0] == 'vllm:prompt_tokens_by_source_total':
                    arm['prompt_by_source_tokens'][key] = arm['prompt_by_source_tokens'].get(key, 0) + value
        receipt = root / (label + '-client-wall.json')
        if receipt.exists():
            client_wall = json.loads(receipt.read_text())['seconds']
            arm['whole_client_wall_s'] = client_wall
            arm['whole_client_output_tok_s'] = output / client_wall
        for phase in phases:
            sub = [r for r in rows if r['phase'] == phase['name']]
            arm['phases'][phase['name']] = {
                'requests': len(sub), 'wall_s': phase['wall_s'],
                'ttft_mean_s': statistics.mean(r['ttft_s'] for r in sub),
                'tpot_mean_s': statistics.mean(r['tpot_s'] for r in sub),
                'counters': {name: metric_sum(phase['metrics_delta'], name) for name in counters}}
        log = root / (label + '-server/server.log')
        if log.exists():
            text = log.read_text(errors='replace')
            arm['reported_kv_gib_per_card'] = sorted(set(re.findall(r'Available KV cache memory: ([0-9.]+) GiB', text)))
            arm['reported_cache_tokens'] = sorted(set(re.findall(r'GPU KV cache size: ([0-9,]+) tokens', text)))
            arm['cannot_store_warning_count'] = text.count('Cannot store')
        exit_path = root / (label + '-server/exit.json')
        if exit_path.exists():
            arm['server_exit'] = json.loads(exit_path.read_text())
        result['arms'][label] = arm
    complete = [name for name, arm in result['arms'].items() if arm['completed_all']]
    if len(complete) > 1:
        result['all_complete_inputs_identical'] = all(signatures[n] == signatures[complete[0]] for n in complete)
    result['input_signatures_sha256'] = {name: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        for name, value in signatures.items()}
    for repetition in (1, 2, 3):
        native = result['arms'].get(f'r{repetition}-native')
        candidate = result['arms'].get(f'r{repetition}-working-set')
        if not native or not candidate or not native['completed_all'] or not candidate['completed_all']:
            continue
        assert signatures[f'r{repetition}-native'] == signatures[f'r{repetition}-working-set']
        pair = {field + '_change_pct': 100 * (candidate[field] / native[field] - 1)
            for field in ('output_tok_s', 'ttft_mean_s', 'ttft_p95_s', 'tpot_mean_s', 'whole_client_output_tok_s')}
        baseline_compute = native['counters']['vllm:request_prefill_kv_computed_tokens_sum']
        pair['computed_prefill_saved_tokens'] = baseline_compute - candidate['counters']['vllm:request_prefill_kv_computed_tokens_sum']
        pair['computed_prefill_saved_pct'] = 100 * pair['computed_prefill_saved_tokens'] / baseline_compute if baseline_compute else 0
        result['pairs'][str(repetition)] = pair
    if result['pairs']:
        result['pair_summary'] = {field: {'median': statistics.median(p[field] for p in result['pairs'].values()),
            'min': min(p[field] for p in result['pairs'].values()), 'max': max(p[field] for p in result['pairs'].values())}
            for field in next(iter(result['pairs'].values()))}
    samples = root / 'monitor.jsonl'
    peaks = {}
    if samples.exists():
        with samples.open() as handle:
            for line in handle:
                try:
                    sample = json.loads(line)
                except ValueError:
                    continue
                arm = sample.get('status', {}).get('arm')
                if not arm:
                    continue
                current = peaks.setdefault(arm, {})
                for line in sample.get('metrics', []):
                    key, raw = line.rsplit(' ', 1)
                    if any(word in key for word in ('kv_cache_usage_perc', 'device_cache_usage',
                        'cpu_cache_usage', 'candidate_working_set_', 'num_requests_running',
                        'num_requests_waiting')):
                        current[key] = max(current.get(key, 0), float(raw))
    result['sampled_peaks'] = peaks
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for label, arm in result['arms'].items():
        print(label, json.dumps({k: v for k, v in arm.items() if k not in ('phases', 'server_exit')}))
    print('pairs', json.dumps(result['pairs']))
