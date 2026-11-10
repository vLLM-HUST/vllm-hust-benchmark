"""Recompute host-return diagnostic summaries from a frozen extracted capsule."""

import argparse
import collections
import json
from pathlib import Path


def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def cpu_delta(row):
    before = {x['tid']: x for x in row['native_threads_before']}
    result = []
    for after in row['native_threads_after']:
        old = before.get(after['tid'])
        if old is None:
            continue
        first = list(map(int, old['schedstat'].split()))
        last = list(map(int, after['schedstat'].split()))
        if any(s in after['name'] for s in ['-load', '-save', 'RT_RECYCLE']):
            result.append(dict(name=after['name'], tid=after['tid'],
                               cpu_ms=(last[0]-first[0])/1e6,
                               runqueue_wait_ms=(last[1]-first[1])/1e6,
                               slices=last[2]-first[2]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capsule', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(native={}, counters={}, raw_controls={})
    for path in sorted(args.capsule.glob('hust-zsx*/**/operations.jsonl')):
        key = str(path.parent.relative_to(args.capsule)).replace('\\', '/')
        operations = []
        for row in jsonl(path):
            entry = {k: row[k] for k in ['event', 'phase', 'policy', 'layout',
                                       'blocks', 'bytes', 'descriptors', 'complete_s',
                                       'elapsed_s', 'all_tags', 'verified_tag_count']
                     if k in row}
            if row.get('event') == 'restore':
                entry['verified_tag_count'] = sum(map(len, row['verified_tags']))
                entry['threads'] = cpu_delta(row)
                def stat(name):
                    return {a: int(b) for a, b in
                            (s.split() for s in row[name]['cpu.stat'].splitlines())}
                first, last = stat('resources_before'), stat('resources_after')
                entry['throttled_usec_delta'] = last['throttled_usec']-first['throttled_usec']
            operations.append(entry)
        result['native'][key] = dict(operations=operations,
                                    completed=(path.parent/'done.json').exists(),
                                    failure=json.loads((path.parent/'failure.json').read_text())
                                    if (path.parent/'failure.json').exists() else None)
        api = jsonl(path.parent/'raw/api.jsonl')
        totals = collections.defaultdict(collections.Counter)
        for row in api:
            if row.get('event') not in ['retry_backoff', 'contended_unlock_control',
                                        'postwake_backoff', 'rw_batch', 'rw_slow_wait',
                                        'native_pack', 'synchronous_pieces', 'piece_api_control']:
                continue
            inferred = {'native_pack': 20, 'synchronous_pieces': 21}
            mode = row.get('mode', inferred.get(row['event'], 0))
            key2 = f"{row['event']}/mode{mode}"
            group = totals[key2]
            group['calls'] += 1
            for field in ['seen_unlocks', 'waiter_unlocks', 'contended_unlocks',
                          'queries', 'zero', 'one', 'above_one', 'pauses', 'pause_wall_ns',
                          'reads', 'writes', 'samples', 'sample_wait_ns',
                          'input_descriptors', 'output_descriptors', 'bytes',
                          'init_ns', 'pack_ns', 'api_ns', 'sync_ns', 'wall_ns', 'cpu_ns']:
                if field in row:
                    group[field] += row[field]
            if 'max_sample_wait_ns' in row:
                group['max_sample_wait_ns'] = max(group['max_sample_wait_ns'],
                                                  row['max_sample_wait_ns'])
        if totals:
            result['counters'][key] = dict(totals)
    for host in ['hust-zsx-81', 'hust-zsx-91']:
        for filename in ['contention-control.jsonl', 'chain.jsonl', 'multistream.jsonl']:
            path = args.capsule/host/filename
            if path.exists():
                result['raw_controls'][f'{host}/{filename}'] = jsonl(path)
    path = args.capsule/'device-versus-host-profile-summary.json'
    if path.exists():
        profiles = json.loads(path.read_text())
        for host in profiles.values():
            for row in host['phases']:
                row['expected_memcpy_tasks'] = 16384
                row['complete_task_coverage'] = row['memcpy_tasks'] == 16384
        result['device_profiles'] = profiles
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(native_groups=len(result['native']),
                          counter_groups=len(result['counters']))))


if __name__ == '__main__':
    main()
