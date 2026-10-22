"""Audit archived server boundaries; CPU replay is not performance evidence."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from benchmarks.audit_ordered_http_parity import compare


def verify_archive(root):
    manifest = json.loads((root / 'evidence-files-private.json').read_text())
    for name, digest in manifest.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe evidence path')
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise ValueError('Evidence checksum mismatch: ' + name)
    return manifest


def initial_order(batch, expected_service, expected_source, expected_programs):
    summary, events = batch['summary'], batch['events']
    if (summary['service_id'] != expected_service
            or summary['observer_source_sha256'] != expected_source
            or summary['dropped'] or summary['snapshot_errors']):
        raise ValueError('Observer identity or completeness failure')
    if [row['sequence'] for row in events] != list(range(1, summary['total_seen'] + 1)):
        raise ValueError('Event sequence gap')
    if any(a['monotonic_ns'] > b['monotonic_ns'] for a, b in zip(events, events[1:])):
        raise ValueError('Event clock reversal')
    starts = [e for e in events if e['event'] == 'admission_start']
    if (len(starts) != len(expected_programs)
            or {e['program_id'] for e in starts} != set(expected_programs)):
        raise ValueError('Admission membership mismatch')
    for pid, rid in expected_programs.items():
        for name in ['request_received', 'admission_end', 'backend_body_received_end',
                     'response_headers_send', 'response_body_send', 'request_finished']:
            rows = [e for e in events if e['event'] == name and e.get('request_id') == rid]
            if len(rows) != 1:
                raise ValueError('Incomplete request boundary: ' + name)
        if not any(e['event'] == 'release_end' and e.get('program_id') == pid
                   and e.get('released') for e in events):
            raise ValueError('Missing program release')
    received = [e for e in events if e['event'] == 'request_received']
    selected = [e for e in events if e['event'] == 'new_program_selection_end'
                and e['selected_backend_index'] is not None]
    queue = next((e for e in events if e['event'] == 'queue_entry_start'), None)
    first_return = next(e for e in events if e['event'] == 'backend_body_received_start')
    if queue is None or any(e['sequence'] >= first_return['sequence'] for e in starts + selected):
        raise ValueError('Initial selection not isolated before first backend return')
    return {'received_order': [e['header_program_id'] for e in received],
            'admission_order': [e['program_id'] for e in starts],
            'initial_active': sorted(e['header_program_id'] for e in selected),
            'first_queue_event': queue,
            'first_backend_return_after_received_seconds': first_return['epoch'] - received[0]['epoch'],
            'ticks_before_first_backend_return_seconds': [e['epoch'] - received[0]['epoch']
                for e in events if e['event'] == 'scheduled_check_start'
                and received[0]['sequence'] <= e['sequence'] < first_return['sequence']]}


def audit(root):
    manifest = verify_archive(root)
    plan = json.loads((root / 'diagnostic-plan-r2-private.json').read_text())
    inputs = json.loads((root / plan['input_file']).read_text())
    if any(len(p['rounds']) != 1 for p in inputs):
        raise ValueError('Only first-round event protocol supported')
    by_task = {p['task']: p for p in inputs}
    result = json.loads((root / 'ablation-result-private.json').read_text())
    if result['status'] != 'COMPLETE_FIXED_WORK':
        raise ValueError('Incomplete diagnostic')
    groups = {}
    for label in ['thunderagent_original', 'controller_original']:
        block = next(b for b in result['blocks'] if b['group'] == label)
        observer = plan['groups'][label]['expected_identity']['server_event_observer']
        expected = {p['records'][0]['request_id'].rsplit(':query-', 1)[0]:
                    p['records'][0]['request_id'] for p in block['programs']}
        batch = json.loads((root / (label + '-terminal-events-private.json')).read_text())
        value = initial_order(batch, observer['service_id'], observer['source_sha256'], expected)
        ordered = []
        for pid in value['admission_order']:
            original = by_task[pid.rsplit(':', 1)[1]]['rounds'][0]
            payload = {**original, 'program_id': pid, 'max_tokens': plan['output_tokens'],
                       'ignore_eos': True, 'temperature': 0, 'seed': 17, 'stream': False}
            ordered.append({'task': pid, 'rounds': [payload], 'tool_wait_seconds': []})
        identity = block['identity_before']['capacity_evidence']
        # Capacity is also observed at the first queue boundary; no inferred engine limit.
        backend = value['first_queue_event']['router']['backends'][0]
        cpu = asyncio.run(compare(ordered, capacity=backend['capacity_tokens'],
                                  buffer_tokens=identity['per_program_buffer_tokens']))
        snapshot = cpu['official']['trace'][0]
        value['cpu'] = {'both_routes_equal': cpu['equal'],
            'active_matches_observed': snapshot['active'] == value['initial_active'],
            'remaining_capacity_tokens': snapshot['remaining_capacity'],
            'remaining_matches_observed': snapshot['remaining_capacity'] == backend['remaining_capacity_tokens']}
        if not all(value['cpu'][key] for key in ['both_routes_equal', 'active_matches_observed',
                                                'remaining_matches_observed']):
            raise ValueError('Observed initial order not explained by CPU semantics')
        groups[label] = value
    return {'groups': groups, 'verified_source_files': len(manifest),
            'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'Observed initial admission order and full payload CPU semantic replay. '
                     'Fixture returns, no timer/model/cache replay; no historical performance causality.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: {'initial_active': len(v['initial_active']), 'cpu': v['cpu']}
                      for k, v in result['groups'].items()}))


if __name__ == '__main__':
    main()
