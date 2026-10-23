"""Explain zero-return Agents from recorded Controller queues; never runs inference."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def audit(run, returns_csv, arm, window):
    root = run / 'terminal-extract'
    batch = root / ('pair-1-' + arm)
    folders = sorted(batch.glob('task-*'))
    arrivals = [json.loads((p / 'arrival.json').read_text()) for p in folders]
    origin = min(x['epoch'] for x in arrivals)
    counts = collections.Counter()
    with returns_csv.open() as stream:
        for row in csv.DictReader(stream):
            if row['arm'] == arm and 0 <= float(row['t']) < window:
                counts[row['program']] += 1
    reports = {p.name: json.loads((p / 'report-private.json').read_text()) for p in folders}
    identity = json.loads((root / (arm + '-evidence') / 'identity-before-private.json').read_text())
    buffer = identity['capacity_evidence']['per_program_buffer_tokens']
    zero = {name: report['program_id'] for name, report in reports.items() if not counts[name]}
    evidence = {pid: {'task': name, 'samples': 0, 'state_counts': collections.Counter(),
                     'backend_counts': collections.Counter(), 'step_counts': collections.Counter(),
                     'queued_seconds': 0.0, 'eligible_ignoring_priority_seconds': 0.0,
                     'first_seen_t': None, 'last_seen_t': None, 'tokens_min': None, 'tokens_max': None}
                for name, pid in zero.items()}
    trace_path = root / (arm + '-evidence') / 'observations-private.jsonl'
    previous = None
    sample_count = 0
    for line in trace_path.open():
        row = json.loads(line)
        t = row['epoch'] - origin
        if previous is not None:
            before, begin = previous
            dt = max(0.0, min(t, window) - max(begin, 0))
            scheduler = before.get('scheduler') or {}
            waiting = {p['program_id']: p for p in scheduler.get('waiting', [])}
            backends = list(scheduler.get('backends', {}).values())
            remaining = backends[0]['remaining_capacity'] if len(backends) == 1 else None
            programs = before.get('programs') or {}
            if dt:
                sample_count += 1
                for pid, result in evidence.items():
                    if pid in waiting:
                        result['queued_seconds'] += dt
                        required = waiting[pid]['program_tokens'] + buffer
                        if remaining is not None and remaining >= required:
                            result['eligible_ignoring_priority_seconds'] += dt
                    state = programs.get(pid)
                    if state is not None:
                        result['samples'] += 1
                        result['first_seen_t'] = max(begin, 0) if result['first_seen_t'] is None else result['first_seen_t']
                        result['last_seen_t'] = min(t, window)
                        result['state_counts'][state['state']] += 1
                        result['backend_counts'][str(state['backend'])] += 1
                        result['step_counts'][str(state['step_count'])] += 1
                        n = state['total_tokens']
                        result['tokens_min'] = n if result['tokens_min'] is None else min(n, result['tokens_min'])
                        result['tokens_max'] = n if result['tokens_max'] is None else max(n, result['tokens_max'])
        previous = row, t
    for name, pid in zero.items():
        events = [json.loads(line) for line in (batch / name / 'agent-events-private.jsonl').read_text().splitlines()]
        evidence[pid]['events'] = dict(collections.Counter(x['event'] for x in events))
        query = [x for x in events if x['event'] == 'agent_query_start']
        evidence[pid]['first_query_t'] = query[0]['epoch'] - origin if query else None
        evidence[pid]['report_status'] = reports[name]['status']
        evidence[pid]['wall_budget_literal'] = 'Frozen Agent wall budget reached' in (batch / name / 'agent-exception-private.txt').read_text()
    source_paths = [returns_csv, trace_path, root / (arm + '-evidence') / 'identity-before-private.json']
    return {'arm': arm, 'window_seconds': window, 'agents': len(folders),
            'zero_return_agents': len(zero), 'buffer_tokens': buffer,
            'observed_intervals': sample_count, 'programs': evidence,
            'source_sha256': {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in source_paths},
            'scope': 'Queue/state snapshots are sequential HTTP reads, not atomic decision witnesses; eligibility ignores other candidates and scheduler tick timing.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--returns-csv', type=Path, required=True)
    p.add_argument('--arm', choices=['native', 'controller'], required=True)
    p.add_argument('--window-seconds', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.run_dir, args.returns_csv, args.arm, args.window_seconds)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    values = list(result['programs'].values())
    print(json.dumps({'zero_return_agents': len(values), 'queue_seconds_min_max':
                      [min(x['queued_seconds'] for x in values), max(x['queued_seconds'] for x in values)] if values else [],
                      'eligible_seconds_min_max': [min(x['eligible_ignoring_priority_seconds'] for x in values),
                                                   max(x['eligible_ignoring_priority_seconds'] for x in values)] if values else [],
                      'observed_states': sorted({s for x in values for s in x['state_counts']}),
                      'observed_backends': sorted({s for x in values for s in x['backend_counts']}),
                      'observed_steps': sorted({s for x in values for s in x['step_counts']})}))


if __name__ == '__main__':
    main()
