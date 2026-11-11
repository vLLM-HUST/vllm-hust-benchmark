"""Audit native identical-address CANN/explicit-prepared-HAL crossover trials."""

import argparse
import bisect
import collections
import hashlib
import json
from pathlib import Path
import statistics

from analyze_resume_20261111 import cpu_stat, jsonl
from analyze_kernel_view import channel


def analyze(directory):
    config = json.loads((directory / 'config.json').read_text())
    ops = [r for r in jsonl(directory / 'operations.jsonl') if r['event'] == 'restore']
    assert json.loads((directory / 'done.json').read_text())['restores'] == len(ops)
    assert not (directory / 'failure.json').exists()
    events = jsonl(directory / 'raw/api.jsonl')
    offset = statistics.median(r['epoch_ns'] - r['start_ns'] for r in events
                               if 'epoch_ns' in r and 'start_ns' in r)
    lifecycle_names = {'direct_cache_prepare', 'direct_cache_destroy'}
    lifecycle = [r for r in events if r.get('event', '').startswith('shadow_')
                 or r.get('event') in lifecycle_names]
    assert all(r['error'] == 0 for r in lifecycle)
    result = []
    initialized = False
    for op in ops:
        descriptors = op['blocks'] * 96
        assert op['all_tags'] and op['verified_tag_count'] == op['descriptors'] == descriptors
        assert op['bytes'] == descriptors * 262144
        assert op['verified_tags'] == [[(i + plane * 521 + 1000) % 30000
                                       for i in range(op['blocks'])] for plane in range(96)]
        start = op['start_epoch_ns'] - offset
        end = start + op['complete_s'] * 1e9
        in_window = [r for r in events if start <= r.get('start_ns', 0) < end]
        mode = op['policy']
        event_name = 'direct_cached_copy' if mode == 32 else 'batch'
        copies = sorted([r for r in in_window if r.get('event') == event_name],
                        key=lambda r: r['start_ns'])
        assert sum(r['copies'] for r in copies) == descriptors
        assert sum(r['bytes'] for r in copies) == op['bytes']
        assert all(r.get('error', r.get('result')) == 0 for r in copies)
        accepted = [r for r in in_window if r.get('event') == 'hal_send'
                    and r.get('kind') == 'task' and r.get('sqe_type6') == 17
                    and r['error'] == 0]
        entries = sum(r['count'] for r in accepted)
        assert entries == (0 if mode == 32 else descriptors), (directory, op['phase'], entries)
        components = collections.defaultdict(float)
        for r in lifecycle:
            overlap = max(0, min(end, r['end_ns']) - max(start, r['start_ns']))
            if overlap:
                components[r['event']] += overlap / 1e9
        setup_s = sum(components.values())
        assert setup_s <= op['complete_s'] + .002
        signature = [(r['src_first'], r['dst_first'], r['copies'], r['bytes']) for r in copies]
        fence_s = sum(r['end_ns'] - r['start_ns'] for r in in_window
                      if r.get('event') == 'direct_stream_fence') / 1e9
        initial_shadow = mode in (29, 32) and not initialized
        initialized |= mode in (29, 32)
        before = cpu_stat(op['resources_before']['cpu.stat'])
        after = cpu_stat(op['resources_after']['cpu.stat'])
        result.append(dict(
            phase=op['phase'], policy=mode, blocks=op['blocks'], descriptors=descriptors,
            bytes=op['bytes'], verified_tags=op['verified_tag_count'], complete_s=op['complete_s'],
            transport='prepared_hal_sync' if mode == 32 else 'cann_async',
            initial_shadow=initial_shadow, lifecycle_s=dict(components),
            residual_s=op['complete_s'] - setup_s, direct_stream_fence_s=fence_s,
            copy_api_wall_s=sum(r['end_ns'] - r['start_ns'] for r in copies) / 1e9,
            copy_api_cpu_s=sum(r.get('thread_cpu_ns', 0) for r in copies) / 1e9,
            driver_queue_payload_entries=entries,
            signature_sha256=hashlib.sha256(json.dumps(signature).encode()).hexdigest(),
            throttled_usec_delta=after['throttled_usec'] - before['throttled_usec']))
    signatures = {r['signature_sha256'] for r in result if r['policy'] in (29, 32)}
    assert len(signatures) == 1, 'source, destination, or batch shape changed'
    hal = [r for r in events if r.get('event') == 'hal_send'
           and r.get('kind') == 'task' and r['error'] == 0]
    queue_key = collections.Counter((r['sq'], r['tid']) for r in hal
                                    if r.get('sqe_type6') == 17).most_common(1)[0][0]
    accepted = sorted([r for r in hal if (r['sq'], r['tid']) == queue_key],
                      key=lambda r: r['end_ns'])
    assert len(accepted) == sum(r['descriptors'] for r in ops if r['policy'] != 32)
    assert all(r['count'] == 1 and r['position'] == i % 2048
               for i, r in enumerate(accepted))
    ends = [r['end_ns'] for r in accepted]
    kernel_points = []
    for frame in jsonl(directory / 'kernel-queue.jsonl'):
        for path, content in frame['files'].items():
            if not path.startswith('/proc/trs_chan/'):
                continue
            state = channel(content.get('text', ''))
            if state is None or state['sq'] != queue_key[0]:
                continue
            forwarded = state['forwarded_entries']
            lower = bisect.bisect_right(ends, content['begin_ns'])
            upper = bisect.bisect_right(ends, content['end_ns'])
            next_wait = ((content['begin_ns'] - accepted[forwarded]['end_ns']) / 1e9
                         if forwarded < len(accepted)
                         and accepted[forwarded]['end_ns'] < content['begin_ns'] else None)
            kernel_points.append(dict(phase=frame['phase'], when=frame['when'],
                                      pending_min=lower - forwarded, pending_max=upper - forwarded,
                                      next_task_driver_wait_s=next_wait, **state))
    return dict(host=config['host'], logical_device=config['device'],
                physical_device=json.loads((directory / 'kernel-resources.json').read_text())[
                    'physical_device'], restores=result, kernel_points=kernel_points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capsule', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = {p.relative_to(args.capsule).as_posix(): analyze(p)
            for p in sorted(args.capsule.glob('hust-zsx-*/container1111-native-direct-*-device*'))}
    pairs = []
    for path in sorted(args.capsule.glob('hust-zsx-*/container1111-native-direct-paired-controller/plan.json')):
        plan = json.loads(path.read_text())
        assert json.loads((path.parent / 'done.json').read_text())['devices'] == 2
        for entry in plan['plan']:
            status = json.loads((path.parent / f"device{entry['device']}-exit.json").read_text())
            assert status['exit_code'] == 0
            key = path.parent.parent.name + f"/container1111-native-direct-paired-device{entry['device']}"
            run = runs[key]
            assert len(run['restores']) == entry['expected_restores']
            for block in entry['blocks']:
                arms = {}
                for index, mode in enumerate(block['order']):
                    stages = [f"stage-{block['start_stage'] + index * 2 + i}" for i in range(2)]
                    chosen = [r for r in run['restores'] if r['phase'] in stages]
                    assert len(chosen) == 2 and all(r['policy'] == mode for r in chosen)
                    arms[str(mode)] = dict(
                        mean_complete_s=statistics.mean(r['complete_s'] for r in chosen),
                        mean_residual_s=statistics.mean(r['residual_s'] for r in chosen),
                        second_complete_s=chosen[1]['complete_s'], stages=stages)
                pairs.append(dict(run=key, block=block['block'], order=block['order'],
                                  seed=entry['seed'], arms=arms,
                                  direct_over_async=arms['32']['mean_complete_s'] / arms['29']['mean_complete_s']))
    all_ops = [r for run in runs.values() for r in run['restores']]
    result = dict(schema='native-direct-dma/v1', runs=runs, pairs=pairs,
                  totals=dict(restores=len(all_ops), descriptors_checked=sum(r['verified_tags'] for r in all_ops),
                              full_size_restores=sum(r['blocks'] == 161 for r in all_ops),
                              pilot_restores=sum(r['blocks'] == 1 for r in all_ops),
                              throttled_usec_delta=sum(r['throttled_usec_delta'] for r in all_ops)))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['totals']))


if __name__ == '__main__':
    main()
