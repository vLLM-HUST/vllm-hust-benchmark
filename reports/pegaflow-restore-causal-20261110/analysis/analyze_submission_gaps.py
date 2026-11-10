"""Relate complete device memcpy records to the batch API that submitted them."""

import argparse
import collections
import csv
from decimal import Decimal
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    database = next((args.root/'profiling').rglob('msprof_*.db'))
    task_csv = next((args.root/'profiling').rglob('task_time*.csv'))
    csv_keys = set()
    with task_csv.open() as stream:
        for row in csv.DictReader(stream):
            if row['kernel_type'] == 'MEMCPY_ASYNC':
                csv_keys.add((int(row['stream_id']), int(row['task_id']),
                              int(Decimal(row['task_start(us)'].strip())*1000)))
    with sqlite3.connect('file:'+database.resolve().as_posix()+'?mode=ro', uri=True) as db:
        rows = db.execute('''
            SELECT t.startNs,t.endNs,t.connectionId,a.startNs,a.endNs,
                   t.taskId,t.streamId,s.value
            FROM TASK t LEFT JOIN CANN_API a ON t.connectionId=a.connectionId
            LEFT JOIN STRING_IDS s ON a.name=s.id ORDER BY t.startNs
        ''').fetchall()
    tasks = [r for r in rows if (r[6], r[5], r[0]) in csv_keys]
    assert len(tasks) == len(csv_keys), 'Device CSV / database record coverage differs'
    assert all(t[7] == 'aclrtMemcpyBatchAsync' for t in tasks), 'Unexpected submission API'
    phases = []
    for line in (args.root/'operations.jsonl').read_text().splitlines():
        operation = json.loads(line)
        if operation.get('event') != 'restore':
            continue
        start = operation['start_epoch_ns']
        end = start+round(operation['complete_s']*1e9)
        selected = [r for r in tasks if start <= r[0] < end]
        counts = collections.Counter()
        durations = collections.Counter()
        examples = []
        for before, after in zip(selected, selected[1:]):
            gap = after[0]-before[1]
            if gap <= 1_000_000:
                continue
            lead = before[1]-after[4]
            category = ('batch_returned_at_least_1ms_before_gap' if lead >= 1_000_000
                        else 'batch_returned_before_gap' if lead >= 0
                        else 'batch_not_started_at_gap' if after[3] > before[1]
                        else 'during_batch_api')
            counts[category] += 1
            durations[category] += gap/1e6
            examples.append(dict(gap_ms=gap/1e6, category=category,
                                 same_batch=before[2] == after[2],
                                 next_api_return_lead_ms=lead/1e6,
                                 previous_task=before[5], next_task=after[5]))
        phases.append(dict(phase=operation['phase'], wall_ms=operation['complete_s']*1000,
                           memcpy_tasks=len(selected), expected=operation['descriptors'],
                           complete_task_coverage=len(selected) == operation['descriptors'],
                           counts=dict(counts), gap_duration_ms=dict(durations),
                           minimum_api_return_lead_ms=min(
                               (x['next_api_return_lead_ms'] for x in examples), default=None),
                           worst=sorted(examples, key=lambda x: x['gap_ms'], reverse=True)[:5]))
    result = dict(database=str(database), csv=str(task_csv), matched_memcpy_records=len(tasks),
                  gap_threshold_ms=1, phases=phases)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([p for p in phases if p['phase'] in ['stage-4', 'stage-6']]))


if __name__ == '__main__':
    main()
