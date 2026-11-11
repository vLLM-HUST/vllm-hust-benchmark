"""Recompute the resumed four-card driver-forwarding diagnosis from frozen data."""

import argparse
import bisect
import collections
import json
from pathlib import Path
import statistics

from analyze_kernel_view import channel


def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def cpu_stat(text):
    return dict((key, int(value)) for key, value in
                (line.split() for line in text.splitlines()))


def analyze_native(directory, recovered_api=None):
    ops = [r for r in jsonl(directory / 'operations.jsonl') if r['event'] == 'restore']
    api = directory / 'raw/api.jsonl'
    if not api.exists() and recovered_api is not None:
        api = recovered_api
    rows = jsonl(api)
    hal = [r for r in rows if r.get('event') == 'hal_send'
           and r.get('kind') == 'task' and r['error'] == 0]
    groups = collections.Counter((r['sq'], r['tid']) for r in hal)
    sq, tid = groups.most_common(1)[0][0]
    accepted = sorted([r for r in hal if r['sq'] == sq and r['tid'] == tid],
                      key=lambda r: r['end_ns'])
    assert len(accepted) == sum(r['descriptors'] for r in ops)
    assert all(r['count'] == 1 and r['position'] == i % 2048
               for i, r in enumerate(accepted))
    ends = [r['end_ns'] for r in accepted]
    points = []
    curves = {}
    for frame in jsonl(directory / 'kernel-queue.jsonl'):
        op = next(r for r in ops if r['phase'] == frame['phase'])
        base = sum(r['descriptors'] for r in ops[:ops.index(op)])
        offset = frame['epoch_ns'] - frame['begin_ns']
        start = op['start_epoch_ns'] - offset
        for path, data in frame['files'].items():
            if not path.startswith('/proc/trs_chan/'):
                continue
            parsed = channel(data.get('text', ''))
            if parsed is None or parsed['sq'] != sq:
                continue
            forwarded = parsed['forwarded_entries']
            before = bisect.bisect_right(ends, data['begin_ns'])
            after = bisect.bisect_right(ends, data['end_ns'])
            wait = None
            if forwarded < len(accepted) and accepted[forwarded]['end_ns'] < data['begin_ns']:
                wait = (data['begin_ns'] - accepted[forwarded]['end_ns']) / 1e6
            points.append(dict(phase=frame['phase'], when=frame['when'],
                               relative_s=(data['begin_ns'] - start) / 1e9,
                               accepted_before=before, accepted_after=after,
                               pending_min=before - forwarded, pending_max=after - forwarded,
                               next_task_driver_wait_ms=wait, phase_base=base, **parsed))
        if frame['phase'] not in curves:
            phase_tasks = accepted[base:base + op['descriptors']]
            # Only the derived figure curve is decimated; every raw driver call is retained.
            sampled = phase_tasks[::64]
            if sampled[-1] is not phase_tasks[-1]:
                sampled.append(phase_tasks[-1])
            curves[frame['phase']] = dict(
                complete_s=op['complete_s'], descriptors=op['descriptors'],
                points=[[(r['end_ns'] - start) / 1e9,
                         bisect.bisect_right(ends, r['end_ns']) - base] for r in sampled])
    summary = dict(completed=read_json(directory / 'done.json'), sq=sq,
                   api_provenance='supplement_recovered_original' if recovered_api == api
                   else 'original_capsule',
                   driver_accepted=len(accepted), kernel_points=points, curves=curves,
                   restores=[])
    for op in ops:
        first = cpu_stat(op['resources_before']['cpu.stat'])
        last = cpu_stat(op['resources_after']['cpu.stat'])
        summary['restores'].append(dict(
            phase=op['phase'], complete_s=op['complete_s'],
            verified_tags=op['verified_tag_count'], descriptors=op['descriptors'],
            throttled_usec_delta=last['throttled_usec'] - first['throttled_usec']))
    return summary


def analyze_pause(directory):
    rows = jsonl(directory / 'queue.jsonl')
    result = dict(controller_failure=read_json(directory / 'failure.json'),
                  controller_done=read_json(directory / 'done.json'), interventions=[])
    for event in jsonl(directory / 'interventions.jsonl'):
        stopped = [r for r in rows if r['kind'] == 'native_stopped'
                   and r['stage'] == event['stage']]
        previous = event['running_history']
        rate = (previous[-1]['forwarded'] - previous[0]['forwarded']) / (
            (previous[-1]['start_ns'] - previous[0]['start_ns']) / 1e9)
        drained = next((r for r in stopped if r['forwarded'] >= r['accepted_at_read']), None)
        result['interventions'].append(dict(
            stage=event['stage'], pause_ms=(event['end_ns'] - event['begin_ns']) / 1e6,
            pre_pause_forwarded_per_s=rate,
            accepted_while_stopped=stopped[-1]['accepted_at_read'],
            first_stopped_pending=stopped[0]['accepted_at_read'] - stopped[0]['forwarded'],
            final_stopped_pending=stopped[-1]['accepted_at_read'] - stopped[-1]['forwarded'],
            forwarded_first_to_last=stopped[-1]['forwarded'] - stopped[0]['forwarded'],
            observation_ms=(stopped[-1]['start_ns'] - stopped[0]['start_ns']) / 1e6,
            observed_drained_by_ms=(drained['end_ns'] - event['begin_ns']) / 1e6
            if drained else None))
    return result


def analyze_control(directory):
    rows = [r for r in jsonl(directory / 'results.jsonl')
            if all(key in r for key in ['wall_ns', 'start_ns', 'end_ns', 'descriptors'])]
    hal = sorted([r for r in jsonl(directory / 'api.jsonl')
                  if r.get('event') == 'hal_send' and r.get('kind') == 'task'
                  and not r['error']], key=lambda r: r['end_ns'])
    ends = [r['end_ns'] for r in hal]
    cases = []
    for row in rows:
        calls = hal[bisect.bisect_left(ends, row['start_ns']):
                    bisect.bisect_right(ends, row['end_ns'])]
        entries = sum(r['count'] for r in calls)
        assert entries == row['descriptors'], (directory.name, row, entries)
        cases.append({**row, 'driver_entries': entries})
    grouped = collections.defaultdict(list)
    for row in cases:
        key = (row.get('kind_name') or row.get('history') or
               ('stream_' + row['stream_kind'] if 'stream_kind' in row else
                'allocation_' + str(row['mode'])))
        grouped[key].append(row['wall_ns'] / 1e9)
    return dict(cases=cases, distributions={k: dict(
        seconds=v, median=statistics.median(v), minimum=min(v), maximum=max(v))
        for k, v in grouped.items()})


def summarize(root, supplement=None, large=None):
    result = dict(native={}, pause={}, controls={}, schema='restore-resume/v1')
    for directory in sorted(root.glob('hust-zsx*/resume1111-*-device*')):
        if (directory / 'operations.jsonl').exists():
            recovered = (supplement / directory.parent.name /
                         (directory.name + '-log-recovery') / 'api.jsonl') if supplement else None
            result['native'][directory.relative_to(root).as_posix()] = analyze_native(
                directory, recovered)
    for directory in sorted(root.glob('hust-zsx*/resume1111-pause-native-a-controller')):
        result['pause'][directory.relative_to(root).as_posix()] = analyze_pause(directory)
    for name in ['resume1111-command-types', 'resume1111-source-history']:
        for directory in sorted(root.glob('hust-zsx*/' + name)):
            result['controls'][directory.relative_to(root).as_posix()] = analyze_control(directory)
    if supplement:
        for name in ['resume1111-stream-scope', 'resume1111-host-lock']:
            for directory in sorted(supplement.glob('hust-zsx*/' + name)):
                result['controls'][directory.relative_to(supplement).as_posix()] = (
                    analyze_control(directory))
    if large:
        for host, name in [('81', 'resume1111-host-lock-large-v3'),
                           ('91', 'resume1111-host-lock-large')]:
            directory = large / ('hust-zsx-' + host) / name
            item = analyze_control(directory)
            item['exit'] = read_json(directory / 'exit.json')
            item['completion_marker'] = any(r.get('event') == 'completed'
                                            for r in jsonl(directory / 'results.jsonl'))
            result['controls'][directory.relative_to(large).as_posix()] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capsule', type=Path)
    parser.add_argument('--supplement', type=Path)
    parser.add_argument('--large', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.capsule, args.supplement, args.large),
                                      indent=2) + '\n')


if __name__ == '__main__':
    main()
