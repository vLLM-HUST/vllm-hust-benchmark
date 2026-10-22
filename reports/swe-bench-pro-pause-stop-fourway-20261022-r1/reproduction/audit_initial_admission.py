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

    result = json.loads(read('ablation-result-private.json'))
    groups = {}
    for block in result['blocks']:
        rows = [json.loads(line) for line in read(
            f'block-{block["index"]}-observations/observations-private.jsonl').splitlines()]
        groups[block['group']] = initial_snapshot(rows, block)
    reference = set(groups['thunderagent_original']['active_tasks'])
    product = set(groups['controller_original']['active_tasks'])
    return {'groups': groups, 'reference_product_initial_membership': {
        'common_active_count': len(reference & product),
        'different_membership_count': len(reference ^ product),
        'different_tasks': sorted(reference ^ product)}, 'source_sha256': sources,
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
