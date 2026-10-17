#!/usr/bin/env python3
"""Build short, explicitly derived trajectory windows for iteration-speed checks.

Uses the canonical sessions.csv schema and the existing replay_sessions.py.
Windows preserve source prefix/input lengths; output and tool waits are capped.
This is not a complete-source-trajectory performance or quality evaluation.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def build(source, destination, programs=8, rounds=4, output_cap=32,
          tool_wait_cap_ms=200, minimum_prefix=8192, maximum_context=32768, exclude_session_ids=()):
    if destination.exists():
        raise ValueError('Destination already exists; choose a new workload name')
    if min(programs, rounds, output_cap) < 1 or min(tool_wait_cap_ms, minimum_prefix) < 0:
        raise ValueError('Invalid workload bounds')
    groups = collections.OrderedDict()
    with (source/'sessions.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        for row in reader:
            groups.setdefault(row['session_id'], []).append(row)
    chosen, windows = [], []
    for identifier, rows in groups.items():
        if identifier in exclude_session_ids:
            continue
        start = next((i for i in range(len(rows)-rounds+1)
                      if int(rows[i]['prefix_len']) >= minimum_prefix
                      and all(int(r['prefix_len'])+int(r['input_len'])+min(int(r['output_len']),output_cap)+32<=maximum_context
                              for r in rows[i:i+rounds])), None)
        if start is None:
            continue
        window = []
        for index, original in enumerate(rows[start:start+rounds]):
            row = dict(original)
            row.update(round_idx=str(index), arrival_time=str(len(windows)),
                       output_len=str(min(int(row['output_len']), output_cap)),
                       tool_wait_after_ms=str(0 if index == rounds-1 else
                                              min(int(row['tool_wait_after_ms']), tool_wait_cap_ms)))
            window.append(row)
        chosen.extend(window)
        windows.append({'session_id':identifier,'source_round_start':int(rows[start]['round_idx']),
                        'source_round_end':int(rows[start+rounds-1]['round_idx'])})
        if len(windows) == programs:
            break
    if len(windows) != programs:
        raise ValueError('Not enough eligible trajectory windows')
    destination.mkdir(parents=True)
    with (destination/'sessions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(chosen)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        'name':destination.name,
        'purpose':'iteration-speed and mechanism diagnostics; not full trajectory benchmark',
        'source_sessions_sha256':digest(source/'sessions.csv'),
        'sessions_sha256':digest(destination/'sessions.csv'),
        'source_manifest':json.loads((source/'manifest.json').read_text()),
        'selection':'First eligible windows in source order, without performance-based selection',
        'parameters':{'programs':programs,'rounds_per_window':rounds,'minimum_initial_prefix_tokens':minimum_prefix,
                      'excluded_source_session_ids':list(exclude_session_ids),
                      'output_cap_tokens':output_cap,'tool_wait_cap_ms':tool_wait_cap_ms,'maximum_context_tokens':maximum_context},
        'windows':windows,'rounds':len(chosen),
        'output_tokens_upper_bound':programs*rounds*output_cap,
        'maximum_tool_wait_seconds_per_window':(rounds-1)*tool_wait_cap_ms/1000,
        'runtime_target_seconds':[120,300],
        'runtime_measured_seconds':None,
        'boundary':'Source trajectories are windowed and generation/tool waits changed; do not merge with full-run gains. Runtime target is not measured runtime.'}
    (destination/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--programs',type=int,default=8)
    p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--output-cap',type=int,default=32)
    p.add_argument('--tool-wait-cap-ms',type=int,default=200)
    p.add_argument('--minimum-prefix',type=int,default=8192)
    p.add_argument('--maximum-context',type=int,default=32768)
    a=p.parse_args()
    m=build(a.source,a.output,a.programs,a.rounds,a.output_cap,a.tool_wait_cap_ms,a.minimum_prefix,a.maximum_context)
    print(json.dumps({k:v for k,v in m.items() if k not in ('source_manifest','windows')},ensure_ascii=False))
