"""Read-only initial membership audit; no inference or policy changes.

Snapshots are non-atomic. Require the full batch at step one, all REASONING,
and a following sample before the first client return to bound collection.
"""
import argparse
import hashlib
import json
from pathlib import Path


def initial_snapshot(rows, block):
    tasks = {p['task'] for p in block['programs']}
    first_return = min(q['ended'] for p in block['programs'] for q in p['records'])
    origin = min(q['started'] for p in block['programs'] for q in p['records'])
    for row, following in zip(rows, rows[1:]):
        programs = row.get('programs', {})
        normalized = {pid.rsplit(':', 1)[-1]: p for pid, p in programs.items()}
        if len(normalized) != len(programs):
            raise ValueError('Ambiguous program identity')
        if (set(normalized) != tasks or following['monotonic'] >= first_return
                or not all(p.get('step_count') == 1 and p.get('status') == 'reasoning'
                           for p in normalized.values())):
            continue
        active = sorted(task for task, p in normalized.items() if p['backend'] is not None)
        return {'collection_lower_seconds': row['monotonic'] - origin,
                'collection_upper_seconds': following['monotonic'] - origin,
                'first_client_return_seconds': first_return - origin,
                'active_count': len(active), 'active_tasks': active,
                'tracked_program_count': len(programs),
                'scope': 'Full-batch step-one sampled membership, not atomic admission decisions'}
    raise ValueError('No full-batch step-one snapshot bounded before first client return')



def budget_snapshot(row, buffer_tokens):
    """Check sequential scheduler/program samples agree; never call them atomic."""
    programs = row.get('programs', {})
    scheduler = row.get('scheduler', {})
    backends = scheduler.get('backends', {})
    if (not programs or len(backends) != 1
            or not all(p.get('step_count') == 1 and p.get('status') == 'reasoning'
                       and type(p.get('total_tokens')) is int for p in programs.values())):
        return None
    backend = next(iter(backends.values()))
    required = ['active_program_tokens', 'remaining_capacity', 'capacity',
                'shared_tokens', 'future_paused_tokens']
    if any(type(backend.get(key)) is not int for key in required):
        return None
    waiting = scheduler.get('waiting')
    if not isinstance(waiting, list):
        return None
    active = {pid: p for pid, p in programs.items() if p['backend'] is not None}
    queued = {pid: p for pid, p in programs.items() if p['backend'] is None}
    queue = {item['program_id']: item['queue_tokens'] for item in waiting}
    tokens = sum(p['total_tokens'] for p in active.values())
    remaining = backend['capacity'] - tokens - len(active) * buffer_tokens + backend['shared_tokens']
    coherent = (len(queue) == len(waiting) and set(queue) == set(queued)
                and all(queue[pid] == p['total_tokens'] for pid, p in queued.items())
                and tokens == backend['active_program_tokens']
                and remaining == backend['remaining_capacity'])
    minimum = min((value + buffer_tokens for value in queue.values()), default=None)
    return {'cross_sample_accounting_consistent': coherent,
            'active_tasks': sorted(pid.rsplit(':', 1)[-1] for pid in active),
            'active_count': len(active), 'waiting_count': len(queue),
            'active_program_tokens': tokens, 'remaining_capacity_tokens': backend['remaining_capacity'],
            'minimum_queued_requirement_tokens': minimum,
            'minimum_queue_fits': None if minimum is None or not coherent else minimum <= remaining,
            'shared_tokens': backend['shared_tokens'], 'future_paused_tokens': backend['future_paused_tokens'],
            'estimated_tokens_by_task': {pid.rsplit(':', 1)[-1]: p['total_tokens']
                                         for pid, p in programs.items()},
            'scope': 'Scheduler then programs are separate HTTP samples; arithmetic agreement is not atomicity'}


def initial_budget_transition(rows, block, buffer_tokens):
    initial = initial_snapshot(rows, block)
    origin = min(q['started'] for p in block['programs'] for q in p['records'])
    first_return = min(q['ended'] for p in block['programs'] for q in p['records'])
    index = next(i for i, row in enumerate(rows)
                 if row['monotonic'] - origin == initial['collection_lower_seconds'])
    before = budget_snapshot(rows[index], buffer_tokens)
    # The following sample also requires its own collection upper bound.
    after = None
    if (index + 2 < len(rows) and rows[index + 2]['monotonic'] < first_return
            and set(rows[index + 1].get('programs', {})) == set(rows[index]['programs'])):
        after = budget_snapshot(rows[index + 1], buffer_tokens)
    transition = None
    if (before is not None and after is not None
            and before['cross_sample_accounting_consistent']
            and after['cross_sample_accounting_consistent']):
        old, new = set(before['active_tasks']), set(after['active_tasks'])
        added, removed = new - old, old - new
        delta = before['remaining_capacity_tokens'] - after['remaining_capacity_tokens']
        cost = sum(before['estimated_tokens_by_task'][pid] + buffer_tokens for pid in added)
        unchanged_estimates = before['estimated_tokens_by_task'] == after['estimated_tokens_by_task']
        transition = {'added_tasks': sorted(added), 'removed_tasks': sorted(removed),
                      'remaining_capacity_decrease_tokens': delta,
                      'added_program_requirement_tokens': cost,
                      'unchanged_token_estimates': unchanged_estimates,
                      'added_cost_explains_budget_delta': (not removed and unchanged_estimates
                           and before['shared_tokens'] == after['shared_tokens'] and delta == cost)}
    return {'before': before, 'following': after, 'transition': transition,
            'buffer_tokens_per_program': buffer_tokens,
            'following_collection_upper_seconds': (rows[index + 2]['monotonic'] - origin
                if after is not None else None),
            'limit': 'No recorded scheduler tick/admission event time; not a causal performance attribution'}


def audit(root):
    root = Path(root)
    manifest = json.loads((root / 'evidence-files-private.json').read_text())
    sources = {}

    def read(path):
        raw = (root / path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if manifest.get(path) != digest:
            raise ValueError('Archived source checksum mismatch: ' + path)
        sources[path] = digest
        return raw.decode()

    capacity = json.loads(read('capacity-evidence.json'))
    buffer_tokens = capacity['per_program_buffer_tokens']
    result = json.loads(read('ablation-result-private.json'))
    groups = {}
    for block in result['blocks']:
        rows = [json.loads(line) for line in read(
            f'block-{block["index"]}-observations/observations-private.jsonl').splitlines()]
        groups[block['group']] = initial_snapshot(rows, block)
        groups[block['group']]['budget_transition'] = initial_budget_transition(rows, block, buffer_tokens)
    reference = set(groups['thunderagent_original']['active_tasks'])
    product = set(groups['controller_original']['active_tasks'])
    return {'groups': groups, 'reference_product_initial_membership': {
        'common_active_count': len(reference & product),
        'different_membership_count': len(reference ^ product),
        'different_tasks': sorted(reference ^ product)},
        'reference_product_initial_estimates_equal': (
            groups['thunderagent_original']['budget_transition']['before']['estimated_tokens_by_task']
            == groups['controller_original']['budget_transition']['before']['estimated_tokens_by_task']
        ) if all(groups[g]['budget_transition']['before'] is not None
                 for g in ['thunderagent_original', 'controller_original']) else None,
        'source_sha256': sources,
        'conclusion': 'Initial active membership already differs before any recorded client return; '
                      'same workload is not the same ordered admission trace. '
                      'This does not establish the cause or size of the cache/performance difference.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.originals)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['reference_product_initial_membership']))


if __name__ == '__main__':
    main()
