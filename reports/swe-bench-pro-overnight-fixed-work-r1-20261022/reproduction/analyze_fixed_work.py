"""Read-only, request-matched analysis of archived finite-work replays.

Replay completion is not fresh Agent task completion. No network, submission,
policy changes or inference is performed by this analysis entry.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

from agent_report_data import EvidenceError, parse_metrics, counter_delta, histogram_delta


def percentile(values, fraction):
    if not values:return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def matched_phases(path, identities):
    events = collections.defaultdict(dict)
    if path.exists():
        for line in path.read_text().splitlines():
            if 'DELIVERY_TRACE ' not in line:continue
            value = json.loads(line.split('DELIVERY_TRACE ', 1)[1])
            identity = value['request_id']
            if identity not in identities:continue  # Exclude other runs on retained services.
            if value['event'] in events[identity]:
                raise EvidenceError('Duplicate matched event: ' + identity)
            events[identity][value['event']] = value['monotonic']
    result = {}
    for identity, values in events.items():
        names = ['request_received', 'admission_start', 'admission_end',
                 'backend_body_received', 'request_finished']
        if not all(name in values for name in names):continue
        ordered = [values[name] for name in names]
        if any(last < first for first, last in zip(ordered, ordered[1:])):
            raise EvidenceError('Matched proxy timestamps are not ordered')
        result[identity] = {'admission_seconds': values['admission_end'] - values['admission_start'],
                            'backend_seconds': values['backend_body_received'] - values['admission_end'],
                            'post_backend_seconds': values['request_finished'] - values['backend_body_received']}
    return result


def analyze(root):
    root = Path(root)
    result_path = root / 'ablation-result-private.json'
    plan_path = root / 'fixed-work-plan-private.json'
    raw, plan = json.loads(result_path.read_text()), json.loads(plan_path.read_text())
    input_path = root / plan['input_file']
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != plan['input_sha256']:
        raise EvidenceError('Frozen input checksum mismatch')
    if plan['arrival_time'] != 0 or plan['client_concurrency_limit'] is not None:
        raise EvidenceError('Not a time-zero finite batch without client slots')
    expected_workers = plan['expected_workers']
    runtime_verified = bool(expected_workers)
    for label in ['runtime_before', 'runtime_after']:
        rows = {str(row['rank']): row for row in raw.get(label, {}).get('results', [])}
        runtime_verified &= set(rows) == set(expected_workers)
        runtime_verified &= all(rows.get(rank, {}).get(key) == value
                                for rank, expected_worker in expected_workers.items()
                                for key, value in expected_worker.items())
    expected = plan['batch_size'] * plan['rounds_per_program']
    report = {'run_id': raw['run_id'], 'scope': 'Frozen-history real-engine replay, not fresh Agent completion',
              'groups': {}, 'source_sha256': {}}
    sources = [result_path, plan_path, input_path]
    for block in raw['blocks']:
        name, index = block['group'], block['index']
        if name in report['groups']:raise EvidenceError('Duplicate group label')
        programs = block.get('programs', [])
        records = [record for program in programs for record in program['records']]
        identities = {record['request_id'] for record in records}
        if len(identities) != len(records):raise EvidenceError('Duplicate request identity')
        trace = root / (name + '-service-ray.log')
        phases = matched_phases(trace, identities)
        origin = min((record['started'] for record in records), default=None)
        durations = [record['elapsed'] for record in records]
        complete_programs = [program for program in programs
                             if len(program['records']) == plan['rounds_per_program']
                             and 'error' not in program and program.get('release', {}).get('released') is True]
        complete = (len(complete_programs) == plan['batch_size'] and len(records) == expected
                    and block.get('errors') == 0 and all(record['usage']['completion_tokens'] == plan['output_tokens']
                                                       for record in records))
        span = max((record['ended'] for record in records), default=0) - origin if origin is not None else None
        if records and abs(span - block['makespan_seconds']) > .001:
            raise EvidenceError('Stored makespan does not match complete request records')
        group = {'complete_work': complete, 'normal_returns': len(records), 'expected_returns': expected,
                 'complete_programs': len(complete_programs), 'errors': block.get('errors'),
                 'output_tokens': sum(record['usage']['completion_tokens'] for record in records),
                 'prompt_tokens': sum(record['usage']['prompt_tokens'] for record in records),
                 'complete_work_makespan_seconds': span if complete else None,
                 'observed_return_span_seconds': span,
                 'client_mean_seconds': statistics.mean(durations) if durations else None,
                 'client_p95_seconds': percentile(durations, .95),
                 'matched_requests': len(phases), 'all_returns_have_matched_phases': len(phases) == len(records),
                 'matched_phase_means': {key: statistics.mean(row[key] for row in phases.values()) if phases else None
                                         for key in ['admission_seconds', 'backend_seconds', 'post_backend_seconds']},
                 'request_rows': [{**record, **phases.get(record['request_id'], {})} for record in records],
                 'program_completion_seconds': sorted(max(record['ended'] for record in program['records']) - origin
                                                       for program in complete_programs),
                 'program_first_reply_seconds': sorted(program['records'][0]['ended'] - origin
                                                        for program in programs if program['records']),
                 'engine_counters': {}, 'engine_histograms': {}}
        before, after = root / f'block-{index}-engine-before.txt', root / f'block-{index}-engine-after.txt'
        if before.exists() and after.exists():
            a, b = [parse_metrics(path.read_text()) for path in [before, after]]
            a, b = [{'m': value[0], 'b': value[1]} for value in [a, b]]
            for metric in ['prefix_cache_queries_total', 'prefix_cache_hits_total',
                           'request_prefill_kv_computed_tokens_sum', 'num_preemptions_total']:
                group['engine_counters'][metric] = counter_delta(a, b, metric)
            for metric in ['request_queue_time_seconds', 'request_prefill_time_seconds',
                           'request_decode_time_seconds', 'e2e_request_latency_seconds']:
                group['engine_histograms'][metric] = histogram_delta(a, b, metric)
            sources += [before, after]
        report['groups'][name] = group
        if trace.exists():sources.append(trace)
    groups = report['groups']
    same_work = (set(groups) == {'native', 'controller'} and all(row['complete_work'] for row in groups.values())
                 and len({(row['prompt_tokens'], row['output_tokens'], row['normal_returns'])
                          for row in groups.values()}) == 1)
    report['same_complete_work'] = same_work
    report['runtime_verified'] = runtime_verified
    comparable = same_work and runtime_verified and raw.get('status') == 'COMPLETE_FIXED_WORK'
    report['comparable_complete_work'] = comparable
    report['makespan_reduction_fraction'] = (1 - groups['controller']['complete_work_makespan_seconds']
                                              / groups['native']['complete_work_makespan_seconds']) if comparable else None
    report['limits'] = ['Single paired replay is not stable or generalized speedup.',
                        'Backend time includes engine queue, execution and transport; it is not kernel time.',
                        'Device and scheduler snapshots are sampled and non-atomic; absence is not zero.',
                        'Request waiting tails must be reported even when complete-work makespan improves.']
    report['source_sha256'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = analyze(args.originals)
    with args.output.open('x') as stream:stream.write(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'same_complete_work': data['same_complete_work'],
                      'makespan_reduction_fraction': data['makespan_reduction_fraction']}))


if __name__ == '__main__':main()
