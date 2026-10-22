"""Partition immutable worker traces at the actual request boundaries."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path('/root/gennova-evidence/20261019-midterm-aligned')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/workspace')
from integrate_r12_hardware import link_bounds
from derive_m01_worker_comm_flit_bound import reference_term


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def partition(db, table, lo, hi):
    # Disjoint buckets include zero-duration boundary events deterministically.
    row = db.execute(f'''SELECT count(*),
        sum(endNs < ?), sum(startNs > ?),
        sum(startNs >= ? AND endNs <= ?),
        min(startNs), max(endNs)
        FROM {table}''', (lo, hi, lo, hi)).fetchone()
    total, before, after, inside, first, last = row
    return dict(total=total, before=before, after=after, inside=inside,
                crossing=total-before-after-inside,
                first_ns=first, last_ns=last)


def counter_bounds(bound):
    devices = sorted(bound['devices'], key=lambda d: d['card'])
    assert [d['card'] for d in devices] == [4, 5]
    result, checks = {}, {}
    for name in ('lower', 'upper'):
        tx = [sum(d['intervals'][name]['tx_flits']) for d in devices]
        rx = [sum(d['intervals'][name]['rx_flits']) for d in devices]
        result[name] = sum(tx)
        checks[name] = dict(tx=tx, rx=rx, peer_equal=tx == rx[::-1])
    assert 0 <= result['lower'] <= result['upper']
    return dict(aggregate_tx=result, peer_checks=checks)


def main():
    audit_path = Path('/workspace/m01-worker-r1-audit.json')
    hardware_path = Path('/workspace/m01-worker-r1-envelope-hardware.json')
    comm_path = Path('/workspace/m01-worker-comm-flit-bound.json')
    audit, hardware, comm = [json.loads(p.read_text()) for p in
                             (audit_path, hardware_path, comm_path)]
    phase = hardware['request_phase']
    lo, hi = [round(phase[k] * 1e9) for k in ('start_wall', 'end_wall')]
    tolerance = audit['clock_bridge']['tolerance_ns']
    sources = {audit_path, hardware_path, comm_path, Path(__file__),
               ROOT / 'integrate_r12_hardware.py',
               Path('/workspace/derive_m01_worker_comm_flit_bound.py')}
    ranks = []
    for name, expected in audit['source_sha256'].items():
        path = Path(name)
        if path.suffix != '.db' or '/system/' in name:
            continue
        assert digest(path) == expected
        sources.add(path)
        with sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True) as db:
            rank = db.execute('select distinct deviceId from TASK').fetchall()
            assert len(rank) == 1
            counts = {}
            for table in ('TASK', 'CANN_API', 'COMMUNICATION_OP'):
                counts[table] = {
                    'nominal': partition(db, table, lo, hi),
                    'contracted_2ms': partition(db, table, lo+tolerance, hi-tolerance),
                    'expanded_2ms': partition(db, table, lo-tolerance, hi+tolerance),
                }
            ranks.append(dict(rank=rank[0][0], tables=counts))
    assert {r['rank'] for r in ranks} == {0, 1}
    raw = ROOT / 'm01-recovery-model-worker-r1/system/hccs64.jsonl'
    assert digest(raw) == comm['source_sha256'][str(raw)]
    sources.add(raw)
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    flits = {}
    for name, left, right in [('nominal', lo, hi),
                              ('contracted_2ms', lo+tolerance, hi-tolerance),
                              ('expanded_2ms', lo-tolerance, hi+tolerance)]:
        flits[name] = counter_bounds(link_bounds(rows, left, right))
    terms = reference_term(flits['expanded_2ms']['aggregate_tx'], comm['reference_seconds'],
                           comm['reference_aggregate_tx_flits'])
    window = next(w for w in hardware['windows']
                  if w['name'] == 'first_request_to_last_response')
    output = dict(status='request_interval_task_partition_and_counter_join',
                  request_phase=phase, request_start_ns=lo, request_end_ns=hi,
                  ranks=ranks, communication_flits=flits,
                  communication_reference_terms=terms,
                  hbm_existing_same_interval_bytes=window['hbm_bytes'],
                  formal_m01_attainment=None,
                  limitations=[
                      'Outside-request tasks must be identified before declaring arithmetic coverage.',
                      'Per-card sampling boundaries can differ; peer discrepancies are retained, not forced equal.',
                      'Clock tolerance is an observed engineering allowance, not a proven absolute bound.',
                      'Physical counters include observer traffic; this is not an uninstrumented score.'],
                  source_sha256={str(p): digest(p) for p in sorted(sources)})
    target = Path('/workspace/m01-request-task-boundaries.json')
    with target.open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
